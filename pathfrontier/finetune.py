"""Fireworks vision SFT (CONTRACTS.md section 5).

Builds the managed supervised-fine-tuning JSONL for a vision model, validates it, and drives the
Fireworks REST API (dataset -> SFT job -> on-demand deployment). Every REST step also exposes the
equivalent ``firectl`` command (``.cli_hint``) so the whole recipe can be run by hand.

Facts this module is built on (all researched 2026-10-29; URLs + notes in docs/FINETUNE.md):

* Vision SFT uses the ordinary chat ``messages`` JSONL. Image parts go on ``user`` messages only,
  inline as ``data:image/<type>;base64,...`` URIs (raw http URLs are not supported in training
  data). ``system``/``assistant`` turns must not contain images.
  https://docs.fireworks.ai/fine-tuning/fine-tuning-models#vision-training
* Dataset: 3 .. 3,000,000 lines; the single-request upload path ``:upload`` takes files up to
  150 MB, larger files go through ``:getUploadEndpoint`` (signed URL) + ``:validateUpload``.
* REST: ``POST /v1/accounts/{acct}/datasets``, ``POST .../datasets/{id}:upload`` (multipart),
  ``POST /v1/accounts/{acct}/supervisedFineTuningJobs`` (body: dataset, baseModel, outputModel,
  epochs, learningRate, loraRank, ...), ``GET .../supervisedFineTuningJobs/{id}`` (state
  ``JOB_STATE_*``), ``POST /v1/accounts/{acct}/deployments`` (baseModel = the fine-tuned model,
  deploymentShape "default"), ``GET .../deployments/{id}`` (state CREATING/READY/FAILED).
* ``loraRank`` MUST be set explicitly over REST (0/omitted == full-parameter training).
  firectl's default is 8, max 128.
* Trained LoRA models are served ONLY on an on-demand (dedicated) deployment, live-merged into the
  base model; the inference model id is ``accounts/<acct>/models/<output-model>``. Serverless LoRA
  serving is not supported.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image

from pathfrontier import config, schema

# ---------------------------------------------------------------------------------------------
# Base models (Fireworks managed SFT catalogue, https://docs.fireworks.ai/fine-tuning/models,
# 2026-10-29). Only models whose model page says "Supporting image input: Supported" AND whose
# catalogue entry has a LoRA training shape with managedSft=true are listed.
# ---------------------------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
VISION_SFT_BASE_MODELS: dict[str, dict[str, Any]] = {
    # id -> {params_b, tier_usd_per_m_tokens (LoRA SFT), lora_shape (smallest), notes}
    "accounts/fireworks/models/qwen3p5-9b": {
        "params_b": 9.4, "train_usd_per_m": 0.50, "lora_shape": "qwen3p5-9b-65k-lora (2 x B200)",
        "notes": "Qwen3.5 9B, native vision encoder; cheapest tier; default."},
    "accounts/fireworks/models/qwen3p5-27b": {
        "params_b": 27.3, "train_usd_per_m": 3.00, "lora_shape": "qwen3p5-27b-64k-lora (1 x B200)",
        "notes": "Qwen3.5 27B dense; better quality, 6x training price."},
    "accounts/fireworks/models/qwen3p6-35b-a3b": {
        "params_b": 35.1, "train_usd_per_m": 3.00, "lora_shape": "qwen3p6-35b-a3b-256k-lora (4 x B200)",
        "notes": "Qwen3.6 35B-A3B MoE; base has serverless inference (LoRA still needs on-demand)."},
    "accounts/fireworks/models/qwen3p8-27b": {
        "params_b": 27.3, "train_usd_per_m": 3.00, "lora_shape": "qwen3p8-27b-32k-b200-lora (4 x B200)",
        "notes": "Qwen3.8 27B VLM (newest Qwen); 4-GPU LoRA shapes only."},
    "accounts/fireworks/models/gemma-4-26b-a4b-it": {
        "params_b": 26.0, "train_usd_per_m": 3.00, "lora_shape": "gemma-4-26b-a4b-256k-b200-lora (4 x B200)",
        "notes": "Gemma 4 26B-A4B, text+image."},
    "accounts/fireworks/models/gemma-4-31b-it": {
        "params_b": 32.2, "train_usd_per_m": 3.00, "lora_shape": "gemma-4-31b-256k-b200-lora (4 x B200)",
        "notes": "Gemma 4 31B, text+image."},
    "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16": {
        "params_b": 124.0, "train_usd_per_m": 6.00,
        "lora_shape": "nvidia-nemotron-3p5-super-vl-120b-a12b-bf16-262k-b300-lora (8 x B300)",
        "notes": "120B-A12B MoE VLM; expensive to serve (8 x B300)."},
}
# Managed LoRA SFT price tiers, USD per 1M training tokens (https://fireworks.ai/pricing).
LORA_SFT_TIERS_USD_PER_M = [(16.0, 0.50), (80.0, 3.00), (300.0, 6.00), (math.inf, 10.00)]

# Image -> token geometry. Qwen3-family vision encoders use 16 px patches merged 2x2, i.e. one
# token per 32x32 px block; Qwen2.5-VL used 28 px (the Fireworks FAQ table: 336x336 -> 144 tokens
# = 28 px/token). We budget with the denser one so estimates are conservative.
PIXELS_PER_IMAGE_TOKEN_SIDE = 28
CHARS_PER_TEXT_TOKEN = 4.0

# Fireworks dataset limits (fine-tuning-models + upload-dataset-files docs).
MIN_EXAMPLES = 3
MAX_EXAMPLES = 3_000_000
SINGLE_UPLOAD_LIMIT_BYTES = 150 * 1024 * 1024

TERMINAL_JOB_STATES = {
    "JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    "JOB_STATE_EARLY_STOPPED", "JOB_STATE_DELETED",
}
SUCCESS_JOB_STATES = {"JOB_STATE_COMPLETED", "JOB_STATE_EARLY_STOPPED"}


class TileLike(Protocol):
    """Duck type of pathfrontier.data.TileRecord (so this module never imports data.py)."""

    tile_id: str
    image: Image.Image
    label: int


# ---------------------------------------------------------------------------------------------
# Rationale templates: 2-3 grounded, one-sentence explanations per class (<= 25 words each).
# ---------------------------------------------------------------------------------------------
RATIONALE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "pcam": {
        "normal": [
            ("Small uniform lymphocytes with round dark nuclei and preserved lymph-node "
             "architecture; no atypical epithelial clusters in the centre."),
            ("Centre shows benign lymphoid tissue, sinus histiocytes and stroma without "
             "pleomorphic or hyperchromatic tumour cells."),
            ("Monotonous population of mature lymphocytes and background stroma; no cohesive "
             "epithelial sheets or nuclear enlargement."),
        ],
        "tumor": [
            ("Sheets of cohesive cells with enlarged, hyperchromatic, pleomorphic nuclei and loss "
             "of normal lymph-node architecture."),
            ("Clusters of epithelial cells with prominent nucleoli and abundant cytoplasm "
             "displacing the lymphoid background, consistent with metastasis."),
            ("Centre contains crowded atypical cells with irregular nuclear membranes and high "
             "nuclear-to-cytoplasmic ratio, unlike surrounding lymphocytes."),
        ],
    },
    "crc": {
        "ADI": [
            "Large clear fat vacuoles with thin membranes and flattened peripheral nuclei.",
            "Optically empty adipocytes in a honeycomb pattern with scant intervening stroma.",
        ],
        "BACK": [
            "Empty glass or out-of-focus smear with no evaluable tissue.",
            "Essentially acellular background with no recognisable histologic structures.",
        ],
        "DEB": [
            "Amorphous eosinophilic granular necrotic material with scattered cell fragments.",
            "Necrotic debris with ghost outlines and karyorrhectic nuclear dust, no intact glands.",
        ],
        "LYM": [
            "Dense sheets of small round dark-blue nuclei with scant cytoplasm.",
            "Monotonous lymphocytic infiltrate with uniform nuclei and minimal cytoplasm.",
        ],
        "MUC": [
            "Pale wispy basophilic amorphous mucin pools, largely acellular.",
            "Extracellular mucus lakes with a few floating cells and no solid gland structure.",
        ],
        "MUS": [
            "Elongated spindle cells with cigar-shaped nuclei arranged in parallel bundles.",
            "Eosinophilic smooth-muscle fascicles with bland elongated nuclei and no glands.",
        ],
        "NORM": [
            "Regular tubular crypts with abundant goblet cells and basally oriented nuclei.",
            "Evenly spaced colonic crypts with preserved polarity and mature goblet cells.",
        ],
        "STR": [
            "Fibrous desmoplastic stroma with scattered fibroblasts and collagen, no epithelium.",
            "Loose to dense collagenous tissue with spindle fibroblasts and no glandular structures.",
        ],
        "TUM": [
            "Crowded irregular glands with hyperchromatic pleomorphic nuclei and loss of polarity.",
            ("Cribriform or solid growth of atypical epithelium with stratified nuclei and dirty "
             "necrosis."),
            "Back-to-back malignant glands with enlarged vesicular nuclei and prominent nucleoli.",
        ],
    },
}


def _fallback_prompt(dataset: str) -> tuple[str, str]:
    """Only used if vlm_client is unavailable; mirrors its v1 JSON-only instruction."""
    classes = schema.CLASSES[dataset]
    opts = " | ".join(f'"{c}"' for c in classes)
    system = ("You are a board-certified pathologist. Classify the H&E patch. Respond with ONLY a "
              f'single JSON object exactly of the form {{"label": {opts}, '
              '"confidence": <number between 0 and 1>, "reason": "<at most 25 words>"}}')
    user = f"Classify this H&E patch as one of {', '.join(classes)}. JSON only."
    return system, user


def build_instruction(dataset: str, prompt_version: str = "v1") -> tuple[str, str]:
    """(system, user_text) identical to the baseline VLM prompt (vlm_client.build_prompt)."""
    try:
        from pathfrontier.vlm_client import build_prompt
    except ImportError:  # pragma: no cover - vlm_client is a sibling slice; keep this importable
        return _fallback_prompt(dataset)
    return build_prompt(dataset, prompt_version)


def image_to_data_url(image: Image.Image, image_size: int = 448, quality: int = 90) -> str:
    """RGB PIL image -> square ``image_size`` JPEG -> ``data:image/jpeg;base64,...``."""
    img = image.convert("RGB")
    if img.size != (image_size, image_size):
        resample = Image.BICUBIC if max(img.size) < image_size else Image.LANCZOS
        img = img.resize((image_size, image_size), resample)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def pick_rationale(templates: list[str], tile_id: str, seed: int = 0) -> str:
    """Deterministic, seeded choice (stable across processes, unlike hash(str))."""
    h = hashlib.sha1(f"{seed}:{tile_id}".encode()).digest()
    return templates[int.from_bytes(h[:4], "big") % len(templates)]


def build_sft_example(record: TileLike, dataset: str,
                      rationale_templates: dict[str, list[str]] | None = None,
                      image_size: int = 448, seed: int = 0, confidence: float = 0.95,
                      prompt_version: str = "v1") -> dict:
    """One Fireworks chat-format training example.

    messages = [system (baseline prompt), user [image part, baseline user text], assistant JSON].
    The assistant target is the strict JSON the baseline parser expects:
    ``{"label": "<class>", "confidence": <float>, "reason": "<one sentence>"}``.
    """
    classes = schema.CLASSES[dataset]
    label_name = classes[int(record.label)]
    templates = (rationale_templates or RATIONALE_TEMPLATES[dataset])[label_name]
    system, user_text = build_instruction(dataset, prompt_version)
    answer = {"label": label_name, "confidence": float(confidence),
              "reason": pick_rationale(templates, record.tile_id, seed)}
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(record.image, image_size)}},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": json.dumps(answer)},
        ],
        # Extra top-level keys are ignored by Fireworks; handy for audits / eval joins.
        "tile_id": record.tile_id,
        "label": int(record.label),
    }


def write_sft_jsonl(records: Iterable[TileLike], dataset: str, path: Path, image_size: int = 448,
                    seed: int = 0, rationale_templates: dict[str, list[str]] | None = None,
                    progress: Callable[[Iterable], Iterable] | None = None) -> int:
    """Write one example per record; returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    it = progress(records) if progress else records
    n = 0
    with open(path, "w") as f:
        for rec in it:
            ex = build_sft_example(rec, dataset, rationale_templates, image_size=image_size, seed=seed)
            f.write(json.dumps(ex) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
_DATA_URL_RE = re.compile(r"^data:image/(jpeg|jpg|png|webp|gif);base64,([A-Za-z0-9+/=\s]+)$")


def _image_tokens_from_data_url(url: str) -> int:
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ValueError("image_url must be a data:image/<type>;base64,... URI")
    raw = base64.b64decode(m.group(2))
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
    return max(1, math.ceil(w / PIXELS_PER_IMAGE_TOKEN_SIDE) * math.ceil(h / PIXELS_PER_IMAGE_TOKEN_SIDE))


def _text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TEXT_TOKEN))


@dataclass
class ValidationReport:
    path: str
    n_examples: int = 0
    n_images: int = 0
    text_tokens: int = 0
    image_tokens: int = 0
    assistant_tokens: int = 0
    bytes: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.text_tokens + self.image_tokens

    @property
    def ok(self) -> bool:
        return not self.errors

    def training_cost_usd(self, epochs: int = 1, usd_per_m_tokens: float = 0.50) -> float:
        return self.total_tokens * epochs * usd_per_m_tokens / 1e6

    def summary(self) -> str:
        mb = self.bytes / 1e6
        lines = [
            f"{self.path}: {self.n_examples} examples, {self.n_images} images, {mb:.1f} MB",
            (f"  ~{self.total_tokens:,} tokens/epoch "
             f"(text {self.text_tokens:,} + image {self.image_tokens:,}; "
             f"assistant {self.assistant_tokens:,})"),
            f"  labels: {json.dumps(self.label_counts)}",
        ]
        if self.bytes > SINGLE_UPLOAD_LIMIT_BYTES:
            lines.append("  NOTE: > 150 MB -> REST client uses the signed-URL upload path; "
                         "firectl handles it automatically")
        if self.errors:
            lines.append(f"  ERRORS ({len(self.errors)}):")
            lines += [f"    - {e}" for e in self.errors[:20]]
        return "\n".join(lines)


def validate_sft_jsonl(path: Path, max_errors: int = 50) -> ValidationReport:
    """Schema check against the Fireworks vision-SFT rules + rough token/image accounting."""
    path = Path(path)
    rep = ValidationReport(path=str(path))
    if not path.exists():
        rep.errors.append("file does not exist")
        return rep
    rep.bytes = path.stat().st_size
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if len(rep.errors) >= max_errors:
                rep.errors.append("... (truncated)")
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                rep.errors.append(f"line {ln}: invalid JSON ({e.msg})")
                continue
            msgs = ex.get("messages") if isinstance(ex, dict) else None
            if not isinstance(msgs, list) or not msgs:
                rep.errors.append(f"line {ln}: missing non-empty 'messages' list")
                continue
            rep.n_examples += 1
            has_assistant = False
            for mi, m in enumerate(msgs):
                role = m.get("role") if isinstance(m, dict) else None
                content = m.get("content") if isinstance(m, dict) else None
                if role not in ("system", "user", "assistant", "tool"):
                    rep.errors.append(f"line {ln} msg {mi}: bad role {role!r}")
                    continue
                parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
                for part in parts:
                    if not isinstance(part, dict):
                        rep.errors.append(f"line {ln} msg {mi}: content part must be an object")
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        txt = part.get("text")
                        if not isinstance(txt, str):
                            rep.errors.append(f"line {ln} msg {mi}: text part without string text")
                            continue
                        t = _text_tokens(txt)
                        rep.text_tokens += t
                        if role == "assistant":
                            rep.assistant_tokens += t
                    elif ptype in ("image_url", "image"):
                        if role not in ("user", "tool"):
                            rep.errors.append(f"line {ln} msg {mi}: images only allowed on user "
                                              f"(or tool) messages, found on {role}")
                            continue
                        url = part.get("image_url") if ptype == "image_url" else part.get("image")
                        if isinstance(url, dict):
                            url = url.get("url")
                        try:
                            rep.image_tokens += _image_tokens_from_data_url(str(url))
                            rep.n_images += 1
                        except Exception as e:  # noqa: BLE001
                            rep.errors.append(f"line {ln} msg {mi}: bad image ({e})")
                    else:
                        rep.errors.append(f"line {ln} msg {mi}: unknown part type {ptype!r}")
                if role == "assistant":
                    has_assistant = True
                    if isinstance(content, str):
                        try:
                            obj = json.loads(content)
                            lab = obj.get("label") if isinstance(obj, dict) else None
                            if lab is not None:
                                rep.label_counts[str(lab)] = rep.label_counts.get(str(lab), 0) + 1
                        except json.JSONDecodeError:
                            rep.errors.append(f"line {ln}: assistant content is not JSON")
            if not has_assistant:
                rep.errors.append(f"line {ln}: no assistant message (nothing to learn)")
    if rep.n_examples < MIN_EXAMPLES:
        rep.errors.append(f"need at least {MIN_EXAMPLES} examples, found {rep.n_examples}")
    if rep.n_examples > MAX_EXAMPLES:
        rep.errors.append(f"too many examples ({rep.n_examples} > {MAX_EXAMPLES})")
    return rep


def estimate_training_cost(tokens_per_epoch: int, epochs: int, base_model: str) -> float:
    """USD at the managed LoRA SFT tier for the base model's parameter count."""
    info = VISION_SFT_BASE_MODELS.get(base_model)
    params_b = info["params_b"] if info else 16.0
    price = next(p for cap, p in LORA_SFT_TIERS_USD_PER_M if params_b <= cap)
    return tokens_per_epoch * epochs * price / 1e6


# ---------------------------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------------------------
class StepResult(dict):
    """A JSON response dict that also carries the equivalent firectl command."""

    cli_hint: str = ""

    def __init__(self, data: dict | None = None, cli_hint: str = ""):
        super().__init__(data or {})
        self.cli_hint = cli_hint


def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return s[:63] or "model"


class FireworksFT:
    """Thin httpx wrapper around the Fireworks fine-tuning REST API.

    Every method returns a :class:`StepResult` (dict + ``.cli_hint``). Static ``cli_*`` helpers
    produce the firectl commands without needing credentials (used for the manual recipe).
    """

    API_BASE = "https://api.fireworks.ai/v1"

    def __init__(self, api_key: str | None = None, account_id: str | None = None,
                 transport: httpx.BaseTransport | None = None, timeout: float = 120.0,
                 base_url: str | None = None):
        self.api_key = api_key if api_key is not None else config.FIREWORKS_API_KEY
        self.account_id = account_id if account_id is not None else config.FIREWORKS_ACCOUNT_ID
        self.base_url = (base_url or os.getenv("FIREWORKS_API_BASE", self.API_BASE)).rstrip("/")
        self._client = httpx.Client(timeout=timeout, transport=transport,
                                    headers={"Authorization": f"Bearer {self.api_key}"})

    # -- helpers ------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.api_key and self.account_id)

    def _require(self) -> None:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        if not self.account_id:
            raise RuntimeError("FIREWORKS_ACCOUNT_ID is not set (firectl whoami shows it)")

    def _acct(self, suffix: str) -> str:
        return f"{self.base_url}/accounts/{self.account_id}{suffix}"

    def _json(self, method: str, url: str, **kw) -> dict:
        r = self._client.request(method, url, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:500]}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    def dataset_name(self, dataset_id: str) -> str:
        return f"accounts/{self.account_id}/datasets/{dataset_id}"

    def model_name(self, output_model: str) -> str:
        return output_model if output_model.startswith("accounts/") else \
            f"accounts/{self.account_id}/models/{output_model}"

    # -- firectl equivalents (no credentials needed) -------------------------------------------
    @staticmethod
    def cli_create_dataset(path: Path | str, dataset_id: str) -> str:
        return f"firectl dataset create {dataset_id} {path}"

    @staticmethod
    def cli_create_job(base_model: str, dataset_id: str, name: str, lora_rank: int = 16,
                       epochs: int = 2, lr: float | None = None,
                       eval_dataset_id: str | None = None) -> str:
        cmd = (f"firectl sftj create --base-model {base_model} --dataset {dataset_id} "
               f"--output-model {name} --lora-rank {lora_rank} --epochs {epochs}")
        if lr is not None:
            cmd += f" --learning-rate {lr:g}"
        if eval_dataset_id:
            cmd += f" --evaluation-dataset {eval_dataset_id}"
        return cmd

    @staticmethod
    def cli_get_job(job_id: str) -> str:
        return f"firectl sftj get {job_id}"

    @staticmethod
    def cli_deploy(model_id: str, deployment_id: str | None = None, shape: str = "default") -> str:
        cmd = f"firectl deployment create {model_id} --deployment-shape {shape} --wait"
        if deployment_id:
            cmd += f" --deployment-id {deployment_id}"
        return cmd

    # -- datasets -----------------------------------------------------------------------------
    def create_dataset(self, path: Path | str, dataset_id: str, wait: bool = True,
                       poll_s: float = 5.0, timeout_s: float = 1800.0) -> StepResult:
        """Create the dataset record, upload the JSONL, wait for state READY."""
        self._require()
        path = Path(path)
        hint = self.cli_create_dataset(path, dataset_id)
        with open(path) as fh:
            n_lines = sum(1 for line in fh if line.strip())
        body = {"datasetId": dataset_id,
                "dataset": {"userUploaded": {}, "exampleCount": str(n_lines),
                            "displayName": dataset_id, "format": "CHAT"}}
        created = self._json("POST", self._acct("/datasets"), json=body)
        size = path.stat().st_size
        if size <= SINGLE_UPLOAD_LIMIT_BYTES:
            with open(path, "rb") as fh:
                self._json("POST", self._acct(f"/datasets/{dataset_id}:upload"),
                           files={"file": (path.name, fh, "application/jsonl")})
        else:  # signed-URL path for big files
            ep = self._json("POST", self._acct(f"/datasets/{dataset_id}:getUploadEndpoint"),
                            json={"filenameToSize": {path.name: str(size)}})
            signed = ep.get("filenameToSignedUrls", {}).get(path.name)
            if not signed:
                raise RuntimeError(f"no signed URL returned: {ep}")
            with open(path, "rb") as fh:
                r = httpx.put(signed, content=fh.read(), timeout=timeout_s,
                              headers={"Content-Type": "application/octet-stream"})
                if r.status_code >= 400:
                    raise RuntimeError(f"signed upload failed HTTP {r.status_code}: {r.text[:300]}")
            self._json("POST", self._acct(f"/datasets/{dataset_id}:validateUpload"), json={})
        info = dict(created)
        if wait:
            t0 = time.time()
            while True:
                info = self._json("GET", self._acct(f"/datasets/{dataset_id}"))
                if info.get("state") == "READY":
                    break
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"dataset {dataset_id} not READY after {timeout_s}s: {info}")
                time.sleep(poll_s)
        info.setdefault("name", self.dataset_name(dataset_id))
        return StepResult(info, hint)

    # -- jobs ---------------------------------------------------------------------------------
    def create_job(self, base_model: str, dataset_id: str, lora_rank: int = 16, epochs: int = 2,
                   lr: float | None = None, name: str | None = None,
                   eval_dataset_id: str | None = None, display_name: str | None = None,
                   extra: dict | None = None) -> StepResult:
        """POST supervisedFineTuningJobs. ``lora_rank`` must be > 0 (0 = full-parameter)."""
        self._require()
        if lora_rank <= 0:
            raise ValueError("lora_rank must be > 0 (0 / omitted means full-parameter training)")
        name = name or f"pathfrontier-{_slug(_short(base_model))}-{int(time.time())}"
        body: dict[str, Any] = {
            "dataset": self.dataset_name(dataset_id),
            "baseModel": base_model,
            "outputModel": self.model_name(name),
            "displayName": display_name or name,
            "epochs": int(epochs),
            "loraRank": int(lora_rank),
        }
        if lr is not None:
            body["learningRate"] = float(lr)
        if eval_dataset_id:
            body["evaluationDataset"] = self.dataset_name(eval_dataset_id)
        if extra:
            body.update(extra)
        res = self._json("POST", self._acct("/supervisedFineTuningJobs"), json=body)
        res.setdefault("outputModel", body["outputModel"])
        res["jobId"] = _short(res.get("name", ""))
        hint = self.cli_create_job(base_model, dataset_id, name, lora_rank, epochs, lr, eval_dataset_id)
        return StepResult(res, hint)

    def get_job(self, job_id: str) -> StepResult:
        self._require()
        job_id = _short(job_id)
        res = self._json("GET", self._acct(f"/supervisedFineTuningJobs/{job_id}"))
        res["jobId"] = job_id
        return StepResult(res, self.cli_get_job(job_id))

    def wait(self, job_id: str, poll_s: float = 30.0, timeout_s: float = 6 * 3600,
             on_update: Callable[[dict], None] | None = None) -> StepResult:
        """Poll until a terminal state; raises on FAILED/CANCELLED/EXPIRED."""
        t0 = time.time()
        last_state = None
        while True:
            job = self.get_job(job_id)
            state = job.get("state", "")
            if on_update and state != last_state:
                on_update(job)
            last_state = state
            if state in TERMINAL_JOB_STATES:
                if state not in SUCCESS_JOB_STATES:
                    msg = (job.get("status") or {}).get("message", "")
                    raise RuntimeError(f"job {job_id} ended in {state}: {msg}")
                return job
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"job {job_id} still {state} after {timeout_s}s")
            time.sleep(poll_s)

    # -- deployment ---------------------------------------------------------------------------
    def deploy(self, model_id: str, deployment_id: str | None = None, shape: str = "default",
               wait: bool = True, poll_s: float = 20.0, timeout_s: float = 3600.0,
               min_replicas: int = 0, max_replicas: int = 1,
               scale_to_zero_window: str = "600s") -> StepResult:
        """Create an on-demand deployment of the fine-tuned model (live-merged LoRA).

        Returns the INFERENCE model id to use with the OpenAI-compatible endpoint
        (``accounts/<acct>/models/<name>``) under key ``inference_model``.
        """
        self._require()
        model_id = self.model_name(model_id)
        deployment_id = deployment_id or _slug(_short(model_id))
        body: dict[str, Any] = {
            "baseModel": model_id,
            "deploymentShape": shape,
            "displayName": deployment_id,
            "minReplicaCount": int(min_replicas),
            "maxReplicaCount": int(max_replicas),
        }
        if min_replicas == 0:
            body["autoscalingPolicy"] = {"scaleToZeroWindow": scale_to_zero_window}
        res = self._json("POST", self._acct("/deployments"), params={"deploymentId": deployment_id},
                         json=body)
        dep_name = res.get("name") or f"accounts/{self.account_id}/deployments/{deployment_id}"
        if wait:
            t0 = time.time()
            while True:
                res = self._json("GET", self._acct(f"/deployments/{_short(dep_name)}"))
                state = res.get("state", "")
                if state == "READY":
                    break
                if state == "FAILED":
                    raise RuntimeError(f"deployment failed: {(res.get('status') or {}).get('message')}")
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"deployment {dep_name} still {state} after {timeout_s}s")
                time.sleep(poll_s)
        res["name"] = dep_name
        res["inference_model"] = model_id
        return StepResult(res, self.cli_deploy(model_id, deployment_id, shape))

    def delete_deployment(self, deployment_id: str) -> StepResult:
        self._require()
        res = self._json("DELETE", self._acct(f"/deployments/{_short(deployment_id)}"))
        return StepResult(res, f"firectl deployment delete {_short(deployment_id)}")

    def close(self) -> None:
        self._client.close()


def manual_recipe(dataset_file: Path | str, base_model: str, name: str, lora_rank: int = 16,
                  epochs: int = 2, lr: float | None = None, eval_file: Path | str | None = None,
                  dataset_id: str | None = None) -> str:
    """The full firectl recipe as a shell snippet (used when no API key is present)."""
    dataset_file = Path(dataset_file)
    dataset_id = dataset_id or _slug(dataset_file.stem)
    eval_id = f"{dataset_id}-eval" if eval_file else None
    lines = [
        "# 0. one-time: install + sign in (https://docs.fireworks.ai/tools-sdks/firectl/firectl)",
        "brew tap fw-ai/firectl && brew install firectl   # or the curl download in docs",
        "firectl signin && firectl whoami                  # note the account id",
        "",
        "# 1. upload the dataset(s)",
        FireworksFT.cli_create_dataset(dataset_file, dataset_id),
    ]
    if eval_file:
        lines.append(FireworksFT.cli_create_dataset(Path(eval_file), eval_id))
    lines += [
        "",
        "# 2. launch the LoRA SFT job (learning rate auto-set by Fireworks unless given)",
        FireworksFT.cli_create_job(base_model, dataset_id, name, lora_rank, epochs, lr, eval_id),
        "",
        "# 3. poll until state COMPLETED (expect ~15-45 min at 2k images incl. queue)",
        "firectl sftj get <JOB_ID>",
        "",
        "# 4. deploy the LoRA (on-demand only; live-merged into the base model)",
        FireworksFT.cli_deploy(f"accounts/<ACCOUNT_ID>/models/{name}"),
        "",
        "# 5. run the fine-tuned evaluation",
        f"export FIREWORKS_FT_MODEL=accounts/<ACCOUNT_ID>/models/{name}",
        ".venv/bin/python scripts/run_vlm_ft.py --dataset pcam --split test --limit 500",
        "",
        "# 6. stop paying for the GPU when done",
        "firectl deployment delete <DEPLOYMENT_ID>",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BASE_MODEL",
    "RATIONALE_TEMPLATES",
    "VISION_SFT_BASE_MODELS",
    "FireworksFT",
    "StepResult",
    "ValidationReport",
    "build_instruction",
    "build_sft_example",
    "estimate_training_cost",
    "image_to_data_url",
    "manual_recipe",
    "pick_rationale",
    "validate_sft_jsonl",
    "write_sft_jsonl",
]
