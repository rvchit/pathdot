"""VLM baseline client (CONTRACTS.md section 4).

OpenAI-compatible async client (Fireworks serverless, a fine-tuned Fireworks deployment, or
local vLLM) plus an optional Anthropic path. Every call returns the same result dict and
`classify_many` emits `schema.Prediction` records with a resumable JSONL cache.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from pathfrontier import config, schema

# ---------------------------------------------------------------------------------------------
# Prices: USD per 1M tokens, (input, output).
#
# Researched 2026-10-29 from https://docs.fireworks.ai/serverless/pricing (the per-model
# "Text and vision models" table + parameter-size tiers) and the individual model pages at
# https://fireworks.ai/models/<owner>/<id> (which state Serverless / Fine-tuning / image-input
# support). Only models whose model page says "Supporting image input: Supported" are listed.
#
# NOTE: as of this date the classic open VLMs (Qwen2.5-VL-32B, Qwen3-VL-8B/30B/235B, Llama 4
# Maverick/Scout, Kimi-VL) are "Serverless: Not Supported" on Fireworks -- on-demand (GPU-hour)
# deployment only. They are kept below with the published parameter-size tier price as a proxy so
# token-cost bookkeeping still works if someone points the client at an on-demand deployment.
# Fine-tuned Fireworks deployments are billed "the same price as base models".
# ---------------------------------------------------------------------------------------------
PRICES: dict[str, tuple[float, float]] = {
    # --- Serverless, vision-capable (fireworks.ai/models/* pages, 2026-10-29) ---------------
    "accounts/fireworks/models/kimi-k3": (3.00, 15.00),               # open MoE, FT: Supported
    "accounts/fireworks/models/ember-1": (3.00, 15.00),               # Fireworks' own, on Kimi K3
    "accounts/fireworks/models/qwen3p8-max": (2.00, 6.00),            # FT: Not Supported
    "accounts/fireworks/models/muse-spark-open": (1.25, 4.25),        # FT: Not Supported
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.30, 1.20),    # FT: Supported
    "accounts/fireworks/models/deepseek-v4p1-flash-1017": (0.30, 1.20),
    "accounts/fireworks/models/minimax-m3p1-flash": (0.15, 0.60),     # FT: Not Supported
    "accounts/fireworks/models/glm-5p3-flash": (0.15, 0.50),          # FT: Supported
    # --- On-demand only (not serverless) -- size-tier proxy prices from the same pricing page:
    #     <4B $0.10, 4-16B $0.20, >16B $0.90, MoE 56-176B $1.20 (same for input and output) -----
    "accounts/fireworks/models/qwen3-vl-8b-instruct": (0.20, 0.20),
    "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct": (0.90, 0.90),   # FT: Supported
    "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct": (1.20, 1.20),
    "accounts/fireworks/models/qwen2p5-vl-32b-instruct": (0.90, 0.90),
    "accounts/fireworks/models/llama4-maverick-instruct-basic": (1.20, 1.20),
    "accounts/fireworks/models/llama4-scout-instruct-basic": (1.20, 1.20),
    "accounts/fireworks/models/mimo-v2p6-pro-rl": (0.90, 0.90),
    "accounts/fireworks/models/qwen3p8-flash-next-fp8": (0.90, 0.90),
}

# Strongest serverless *open-weights* VLM on Fireworks today (Kimi K3: image input + serverless +
# fine-tunable). Budget alternative with the same capabilities: deepseek-v4p1-flash (10x cheaper).
DEFAULT_BASELINE_MODEL = "accounts/fireworks/models/kimi-k3"
BUDGET_BASELINE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"

# Anthropic first-party list prices (claude-api skill table, cached 2026-10-06), USD per 1M tokens.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"

# Size-tier fallback when a model id is unknown (Fireworks ">16B parameters" tier).
UNKNOWN_MODEL_PRICE = (0.90, 0.90)

MIN_IMAGE_PX = 448


def price_for(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for a model id; fine-tuned ids inherit their base price."""
    if model in PRICES:
        return PRICES[model]
    if model in ANTHROPIC_PRICES:
        return ANTHROPIC_PRICES[model]
    # Fine-tuned deployments look like accounts/<acct>/models/<base>-<suffix> or
    # accounts/<acct>/deployedModels/<base>-<hash>; match on the base model's short name.
    short = model.rsplit("/", 1)[-1].lower()
    best: tuple[int, tuple[float, float]] | None = None
    for known, p in PRICES.items():
        k = known.rsplit("/", 1)[-1].lower()
        if short.startswith(k) and (best is None or len(k) > best[0]):
            best = (len(k), p)
    return best[1] if best else UNKNOWN_MODEL_PRICE


@dataclass
class VLMConfig:
    model: str = DEFAULT_BASELINE_MODEL
    base_url: str = config.FIREWORKS_BASE_URL
    api_key: str = config.FIREWORKS_API_KEY
    max_tokens: int = 200
    temperature: float = 0.0
    concurrency: int = 8
    price_in_per_m: float = -1.0   # -1 -> looked up from PRICES in __post_init__
    price_out_per_m: float = -1.0
    provider: str = "openai"       # "openai" (any OpenAI-compatible host) | "anthropic"
    max_retries: int = 6
    timeout_s: float = 90.0
    min_image_px: int = MIN_IMAGE_PX
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.price_in_per_m < 0 or self.price_out_per_m < 0:
            pin, pout = price_for(self.model)
            if self.price_in_per_m < 0:
                self.price_in_per_m = pin
            if self.price_out_per_m < 0:
                self.price_out_per_m = pout
        if self.provider == "anthropic" and self.api_key == config.FIREWORKS_API_KEY:
            self.api_key = config.ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------
CRC_DESCRIPTIONS = {
    "ADI": "adipose tissue: large clear fat vacuoles with thin membranes and peripheral nuclei",
    "BACK": "background: empty glass, no tissue, or out-of-focus smear",
    "DEB": "debris: necrotic material, amorphous eosinophilic granular material, cell fragments",
    "LYM": "lymphocytes: dense sheets of small round dark-blue nuclei with scant cytoplasm",
    "MUC": "mucus: pale, wispy, basophilic amorphous pools, often acellular",
    "MUS": "smooth muscle: elongated spindle cells with cigar-shaped nuclei in parallel bundles",
    "NORM": "normal colon mucosa: regular tubular crypts with goblet cells and basal nuclei",
    "STR": "cancer-associated stroma: fibrous desmoplastic tissue with scattered fibroblasts",
    "TUM": "colorectal adenocarcinoma epithelium: crowded irregular glands, hyperchromatic "
    "pleomorphic nuclei, loss of polarity, cribriform or solid growth",
}


def _json_schema_hint(classes: list[str]) -> str:
    opts = " | ".join(f'"{c}"' for c in classes)
    return (
        "Respond with ONLY a single JSON object, no prose before or after, exactly of the form "
        f'{{"label": {opts}, "confidence": <number between 0 and 1>, '
        '"reason": "<at most 25 words>"}}'
    )


def build_prompt(dataset: str, prompt_version: str = "v1") -> tuple[str, str]:
    """Return (system_prompt, user_text) for a dataset + prompt version."""
    if prompt_version != "v1":
        raise ValueError(f"unknown prompt_version {prompt_version!r}; only 'v1' exists")
    classes = schema.CLASSES[dataset]
    if dataset in ("pcam", "camelyon17"):
        source = ("one small H&E-stained patch (96x96 um at 10x) taken from a sentinel lymph node "
                  "section of a breast cancer patient (PatchCamelyon)") if dataset == "pcam" else (
                  "one small 96x96 px H&E-stained patch taken from a sentinel lymph node whole-slide "
                  "image of a breast cancer patient (Camelyon17)")
        system = (
            "You are a board-certified pathologist specialising in breast cancer staging. "
            f"You are shown {source}. Decide whether the "
            "CENTRE region of the patch contains metastatic tumour cells. Tumour: sheets or "
            "clusters of epithelial cells with enlarged, pleomorphic, hyperchromatic nuclei, "
            "prominent nucleoli, abundant cytoplasm, displacing the normal lymphoid "
            "architecture. Normal: small uniform lymphocytes, germinal centres, sinus "
            "histiocytes, fat, vessels, stroma or background. Calibrate your confidence "
            "honestly; it will be used as a probability. " + _json_schema_hint(classes)
        )
        user = "Classify this H&E lymph node patch as normal or tumor. JSON only."
    elif dataset == "crc":
        desc = "\n".join(f"- {c}: {CRC_DESCRIPTIONS[c]}" for c in classes)
        system = (
            "You are a board-certified gastrointestinal pathologist. You are shown one H&E-stained "
            "224x224 px patch (0.5 um/px) from a human colorectal cancer or normal colon section "
            "(NCT-CRC-HE). Assign the single dominant tissue class using these codes:\n"
            f"{desc}\n"
            "Calibrate your confidence honestly; it will be used as a probability. "
            + _json_schema_hint(classes)
        )
        user = "Classify the dominant tissue type in this H&E colorectal patch. JSON only."
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return system, user


# ---------------------------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------------------------
def prepare_image(image: Image.Image, min_px: int = MIN_IMAGE_PX) -> Image.Image:
    """RGB copy upscaled (bicubic) so the shorter side is >= min_px; never downscaled."""
    img = image.convert("RGB") if image.mode != "RGB" else image
    w, h = img.size
    short = min(w, h)
    if short < min_px:
        f = min_px / short
        img = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BICUBIC)
    return img


def encode_image(image: Image.Image, min_px: int = MIN_IMAGE_PX, fmt: str = "PNG") -> str:
    """Base64 data URL for the (upscaled) tile."""
    img = prepare_image(image, min_px)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
_LABEL_RE = re.compile(r"[\"']?label[\"']?\s*[:=]\s*[\"']?([A-Za-z_]+)", re.IGNORECASE)
_CONF_RE = re.compile(r"[\"']?confidence[\"']?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)
_REASON_RE = re.compile(r"[\"']?reason[\"']?\s*[:=]\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


def _candidate_json_blobs(text: str) -> Iterable[str]:
    for m in _FENCE_RE.finditer(text):
        yield m.group(1).strip()
    yield text.strip()
    # every balanced-looking {...} span, outermost first
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in starts:
        depth = 0
        for j in range(s, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    yield text[s : j + 1]
                    break


def _match_label(raw_label: Any, classes: list[str]) -> int | None:
    if raw_label is None:
        return None
    s = str(raw_label).strip().strip("\"'").lower()
    lower = [c.lower() for c in classes]
    if s in lower:
        return lower.index(s)
    # integer index
    if s.isdigit() and int(s) < len(classes):
        return int(s)
    # synonyms / prefixes
    syn = {
        "pcam": {"tumour": "tumor", "malignant": "tumor", "metastasis": "tumor",
                 "metastatic": "tumor", "cancer": "tumor", "positive": "tumor",
                 "benign": "normal", "negative": "normal", "non-tumor": "normal",
                 "nontumor": "normal"},
    }
    for table in syn.values():
        if s in table and table[s] in lower:
            return lower.index(table[s])
    for i, c in enumerate(lower):
        if s.startswith(c) or c.startswith(s) and len(s) >= 3:
            return i
    return None


def _clamp_conf(v: Any, pct: bool = False) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    if pct or f >= 10.0:          # "85" / "85%" -> 0.85
        f = f / 100.0
    return min(1.0, max(0.0, f))  # 1.7 -> 1.0 (mis-scaled), negatives -> 0


def parse_response(raw: str, dataset: str) -> dict:
    """Robustly parse the model's text.

    Returns {"label": int|None, "confidence": float, "reason": str, "parse_error": bool}.
    Order: fenced JSON -> whole text as JSON -> any {...} span -> regex key fallback.
    """
    classes = schema.CLASSES[dataset]
    text = raw or ""
    for blob in _candidate_json_blobs(text):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        label = _match_label(obj.get("label", obj.get("class", obj.get("prediction"))), classes)
        if label is None:
            continue
        conf = _clamp_conf(obj.get("confidence", obj.get("probability")))
        reason = obj.get("reason", obj.get("rationale", ""))
        return {
            "label": label,
            "confidence": 0.5 if conf is None else conf,
            "reason": str(reason or "").strip(),
            "parse_error": False,
        }
    # Regex fallback on free text
    label = None
    m = _LABEL_RE.search(text)
    if m:
        label = _match_label(m.group(1), classes)
    if label is None:
        # bare class mention, longest match first so "tumor" beats "normal" in "no tumor"? No:
        # prefer the LAST class word mentioned, which is usually the verdict.
        hits = [(mm.start(), i) for i, c in enumerate(classes)
                for mm in re.finditer(rf"\b{re.escape(c)}\b", text, re.IGNORECASE)]
        if hits:
            label = max(hits)[1]
    conf = None
    m = _CONF_RE.search(text)
    if m:
        conf = _clamp_conf(m.group(1), pct=bool(m.group(2)))
    reason = ""
    m = _REASON_RE.search(text)
    if m:
        reason = m.group(1).strip()
    if label is None:
        return {"label": None, "confidence": 0.0, "reason": text.strip()[:200], "parse_error": True}
    return {"label": label, "confidence": 0.5 if conf is None else conf, "reason": reason,
            "parse_error": False}


def probs_from(label: int | None, confidence: float, n_classes: int) -> list[float]:
    """Probability vector: label gets `confidence`, remainder spread uniformly over the rest.

    Binary case reduces to p_tumor = conf if tumor else 1 - conf. Unknown label -> uniform.
    """
    if label is None or not (0 <= label < n_classes):
        return [1.0 / n_classes] * n_classes
    c = min(1.0, max(0.0, float(confidence)))
    if n_classes == 1:
        return [1.0]
    rest = (1.0 - c) / (n_classes - 1)
    return [c if i == label else rest for i in range(n_classes)]


def cost_usd(tokens_in: int, tokens_out: int, cfg: VLMConfig) -> float:
    return (tokens_in * cfg.price_in_per_m + tokens_out * cfg.price_out_per_m) / 1_000_000.0


# ---------------------------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------------------------
class FakeAsyncClient:
    """Stand-in for openai.AsyncOpenAI used by --dry-run and tests.

    Mimics `client.chat.completions.create(...)` and returns canned answers in a mix of formats
    (plain JSON, fenced JSON, JSON + trailing prose) so the parser is exercised end to end.
    `oracle` may map a sha1 of the image bytes -> class index to make answers "correct".
    """

    def __init__(self, dataset: str = "pcam", latency_s: float = 0.0, seed: int = 0,
                 fail_first_n: int = 0, oracle: dict[str, int] | None = None):
        self.dataset = dataset
        self.latency_s = latency_s
        self.rng = random.Random(seed)
        self.fail_first_n = fail_first_n
        self.oracle = oracle or {}
        self.calls = 0
        self.chat = _FakeChat(self)

    async def _create(self, **kw) -> Any:
        self.calls += 1
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise _FakeRateLimit("fake 429")
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        classes = schema.CLASSES[self.dataset]
        img_hash = _image_hash_from_messages(kw.get("messages", []))
        if img_hash in self.oracle:
            label = classes[self.oracle[img_hash]]
        else:
            h = int(img_hash[:8], 16) if img_hash else self.rng.randrange(1 << 32)
            label = classes[h % len(classes)]
        conf = round(0.55 + 0.45 * self.rng.random(), 2)
        obj = {"label": label, "confidence": conf, "reason": f"canned {label} answer"}
        style = self.calls % 3
        if style == 0:
            content = json.dumps(obj)
        elif style == 1:
            content = "```json\n" + json.dumps(obj, indent=1) + "\n```"
        else:
            content = json.dumps(obj) + "\n\nI hope this helps."
        return _obj(
            choices=[_obj(message=_obj(content=content), finish_reason="stop")],
            usage=_obj(prompt_tokens=900, completion_tokens=40, total_tokens=940),
            model=kw.get("model", "fake"),
        )


class _FakeChat:
    def __init__(self, parent: FakeAsyncClient):
        self.completions = _obj(create=parent._create)


class _FakeRateLimit(Exception):
    status_code = 429


class _obj:  # tiny attribute bag
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _image_hash_from_messages(messages: list) -> str:
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    return hashlib.sha1(url.encode()).hexdigest()
    return ""


def image_hash(image: Image.Image, min_px: int = MIN_IMAGE_PX) -> str:
    """Same hash FakeAsyncClient sees, so tests/dry runs can build an oracle."""
    return hashlib.sha1(encode_image(image, min_px).encode()).hexdigest()


def make_client(cfg: VLMConfig) -> Any:
    if cfg.provider == "anthropic":
        import anthropic

        return anthropic.AsyncAnthropic(api_key=cfg.api_key or None, timeout=cfg.timeout_s,
                                        max_retries=0)
    import openai

    return openai.AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "missing",
                              timeout=cfg.timeout_s, max_retries=0)


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status == 408 or status >= 500
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "RateLimit", "InternalServer",
                                   "ServiceUnavailable", "APIError"))


async def _with_retries(fn: Callable[[], Awaitable[Any]], cfg: VLMConfig,
                        sleep: Callable[[float], Awaitable[None]] | None = None) -> Any:
    sleep = sleep or asyncio.sleep
    last: BaseException | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == cfg.max_retries:
                raise
            last = exc
            delay = min(60.0, (2.0**attempt) + random.uniform(0, 0.5))
            await sleep(delay)
    raise last  # pragma: no cover


async def _call_openai(client: Any, image: Image.Image, dataset: str, cfg: VLMConfig,
                       prompt_version: str) -> tuple[str, int, int]:
    system, user = build_prompt(dataset, prompt_version)
    data_url = encode_image(image, cfg.min_image_px)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user},
        ]},
    ]
    resp = await client.chat.completions.create(
        model=cfg.model, messages=messages, max_tokens=cfg.max_tokens,
        temperature=cfg.temperature, **cfg.extra,
    )
    choice = resp.choices[0]
    raw = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0)
    tout = int(getattr(usage, "completion_tokens", 0) or 0)
    return raw, tin, tout


async def _call_anthropic(client: Any, image: Image.Image, dataset: str, cfg: VLMConfig,
                          prompt_version: str) -> tuple[str, int, int]:
    system, user = build_prompt(dataset, prompt_version)
    data_url = encode_image(image, cfg.min_image_px)
    b64 = data_url.split(",", 1)[1]
    messages = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        {"type": "text", "text": user},
    ]}]
    # Thinking is always on for current Claude models (omit the param); effort low keeps it
    # cheap for a short classification; server-side refusal fallback on by default.
    resp = await client.beta.messages.create(
        model=cfg.model, max_tokens=max(cfg.max_tokens, 256), system=system, messages=messages,
        output_config={"effort": cfg.extra.get("effort", "low")},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raw = ""
    else:
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = getattr(resp, "usage", None)
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    return raw, tin, tout


async def classify_tile_async(image: Image.Image, dataset: str, cfg: VLMConfig,
                              prompt_version: str = "v1", client: Any = None) -> dict:
    """One tile -> result dict (see CONTRACTS section 4). Never raises on parse problems."""
    client = client or make_client(cfg)
    call = _call_anthropic if cfg.provider == "anthropic" else _call_openai
    t0 = time.perf_counter()
    try:
        raw, tin, tout = await _with_retries(
            lambda: call(client, image, dataset, cfg, prompt_version), cfg)
        api_error = None
    except Exception as exc:  # noqa: BLE001 - exhausted retries/non-retryable: record it
        raw, tin, tout, api_error = "", 0, 0, f"{type(exc).__name__}: {exc}"[:300]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    parsed = parse_response(raw, dataset)
    n = len(schema.CLASSES[dataset])
    probs = probs_from(parsed["label"], parsed["confidence"], n)
    label = parsed["label"] if parsed["label"] is not None else int(max(range(n), key=probs.__getitem__))
    out = {
        "label": int(label),
        "probs": probs,
        "reason": parsed["reason"],
        "latency_ms": latency_ms,
        "cost_usd": cost_usd(tin, tout, cfg),
        "raw": raw,
        "tokens_in": tin,
        "tokens_out": tout,
        "parse_error": bool(parsed["parse_error"]),
        "confidence": float(parsed["confidence"]),
    }
    if api_error:
        out["api_error"] = api_error
        out["parse_error"] = True
    return out


def classify_tile(image: Image.Image, dataset: str, cfg: VLMConfig,
                  prompt_version: str = "v1", client: Any = None) -> dict:
    """Synchronous convenience wrapper around classify_tile_async."""
    return asyncio.run(classify_tile_async(image, dataset, cfg, prompt_version, client))


# ---------------------------------------------------------------------------------------------
# Batch + cache
# ---------------------------------------------------------------------------------------------
def _cache_key(model: str, prompt_version: str, tile_id: str) -> str:
    return f"{model}|{prompt_version}|{tile_id}"


def load_cache(cache_path: Path | None) -> dict[str, dict]:
    """Read a JSONL cache -> {key: result dict}. Tolerates partial/corrupt trailing lines."""
    out: dict[str, dict] = {}
    if not cache_path or not Path(cache_path).exists():
        return out
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _cache_key(rec.get("model", ""), rec.get("prompt_version", ""),
                             rec.get("tile_id", ""))
            if rec.get("result"):
                out[key] = rec["result"]
    return out


def _append_cache_line(path: Path, rec: dict) -> None:
    # open/append/close per line: no awaits in between, so lines never interleave, and a crash
    # mid-run leaves at most one partial trailing line (which load_cache skips).
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def split_from_tile_id(tile_id: str) -> str:
    parts = tile_id.split(":")
    return parts[1] if len(parts) >= 3 else "test"


def to_prediction(rec: Any, result: dict, dataset: str, model: str, stage: str,
                  prompt_version: str, split: str | None = None) -> schema.Prediction:
    extra = {
        "prompt_version": prompt_version,
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "parse_error": bool(result.get("parse_error", False)),
        "raw": (result.get("raw") or "")[:500],
    }
    if result.get("api_error"):
        extra["api_error"] = result["api_error"]
    return schema.Prediction(
        tile_id=rec.tile_id,
        dataset=dataset,
        split=split or split_from_tile_id(rec.tile_id),
        y_true=int(rec.label),
        y_prob=list(result["probs"]),
        y_pred=int(result["label"]),
        model=model,
        stage=stage,
        latency_ms=float(result.get("latency_ms", 0.0)),
        cost_usd=float(result.get("cost_usd", 0.0)),
        confidence=max(result["probs"]) if not result.get("parse_error") else None,
        reason=result.get("reason") or None,
        extra=extra,
    )


async def classify_many(records: list, dataset: str, cfg: VLMConfig, prompt_version: str = "v1",
                        cache_path: Path | None = None, stage: str = "baseline",
                        split: str | None = None, client: Any = None,
                        progress: bool = True) -> list[schema.Prediction]:
    """Classify TileRecord-like objects (.tile_id/.image/.label) concurrently.

    Concurrency via asyncio.Semaphore(cfg.concurrency); exponential backoff per call; results
    appended to `cache_path` (JSONL keyed by model+prompt_version+tile_id) as they complete so an
    interrupted run resumes without re-paying for finished tiles. Output order == input order.
    """
    client = client or make_client(cfg)
    cache = load_cache(cache_path)
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    results: dict[str, dict] = {}
    todo = []
    for rec in records:
        key = _cache_key(cfg.model, prompt_version, rec.tile_id)
        if key in cache:
            results[rec.tile_id] = cache[key]
        else:
            todo.append(rec)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(records), initial=len(records) - len(todo), unit="tile",
                       desc=cfg.model.rsplit("/", 1)[-1])
        except ImportError:  # pragma: no cover
            bar = None

    async def one(rec) -> None:
        async with sem:
            res = await classify_tile_async(rec.image, dataset, cfg, prompt_version, client)
        results[rec.tile_id] = res
        if cache_path:
            _append_cache_line(cache_path, {"model": cfg.model, "prompt_version": prompt_version,
                                            "tile_id": rec.tile_id, "result": res})
        if bar is not None:
            bar.update(1)

    try:
        await asyncio.gather(*(one(r) for r in todo))
    finally:
        if bar is not None:
            bar.close()

    return [to_prediction(r, results[r.tile_id], dataset, cfg.model, stage, prompt_version, split)
            for r in records]


def summarize(preds: list[schema.Prediction]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "mean_latency_ms": float("nan"),
                "total_cost_usd": 0.0, "parse_errors": 0}
    correct = sum(p.y_pred == p.y_true for p in preds)
    return {
        "n": n,
        "accuracy": correct / n,
        "mean_latency_ms": sum(p.latency_ms for p in preds) / n,
        "total_cost_usd": sum(p.cost_usd for p in preds),
        "parse_errors": sum(bool(p.extra.get("parse_error")) for p in preds),
    }


# ---------------------------------------------------------------------------------------------
# Shared runner used by scripts/run_baseline.py and scripts/run_vlm_ft.py
# ---------------------------------------------------------------------------------------------
def run_split(model: str, dataset: str, split: str, limit: int | None, concurrency: int,
              prompt_version: str, provider: str, stage: str, dry_run: bool = False,
              seed: int = 0, results_dir: Path | None = None, cache_dir: Path | None = None,
              out_name: str | None = None) -> tuple[Path, dict]:
    """Load tiles, classify, write preds JSONL, return (path, summary)."""
    results_dir = results_dir or config.RESULTS_DIR
    cache_dir = cache_dir or config.CACHE_DIR
    cfg = VLMConfig(model=model, provider=provider, concurrency=concurrency)
    if not dry_run and not cfg.api_key:
        env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "FIREWORKS_API_KEY"
        raise SystemExit(f"{env} is not set; use --dry-run to exercise the pipeline offline.")

    records = _load_records(dataset, split, limit, seed, dry_run)
    client = None
    if dry_run:
        oracle = {image_hash(r.image, cfg.min_image_px): int(r.label) for r in records}
        # make ~1 in 5 answers wrong so the summary isn't a boring 100%
        for i, k in enumerate(list(oracle)):
            if i % 5 == 0:
                oracle[k] = (oracle[k] + 1) % len(schema.CLASSES[dataset])
        client = FakeAsyncClient(dataset=dataset, oracle=oracle)

    safe = cfg.model.replace("/", "_").replace(":", "_")
    cache_path = cache_dir / f"vlm_{safe}_{dataset}_{split}_{prompt_version}.jsonl"
    if dry_run:
        cache_path = cache_dir / f"vlm_dryrun_{safe}_{dataset}_{split}_{prompt_version}.jsonl"
    preds = asyncio.run(classify_many(records, dataset, cfg, prompt_version, cache_path,
                                      stage=stage, split=split, client=client))
    out = schema.preds_path(out_name or cfg.model, split, results_dir)
    schema.write_predictions(preds, out)
    return out, summarize(preds)


def _load_records(dataset: str, split: str, limit: int | None, seed: int, dry_run: bool) -> list:
    try:
        from pathfrontier import data
    except ImportError as exc:  # pragma: no cover - data.py missing
        if not dry_run:
            raise
        print(f"[vlm] pathfrontier.data unavailable ({exc}); using built-in synthetic tiles")
        return _local_synthetic(dataset, split, limit or 16, seed)
    if dry_run:
        try:
            return data.make_synthetic(dataset, n=limit or 16, seed=seed)
        except Exception as exc:  # noqa: BLE001 - dry-run must never die on a fixture bug
            print(f"[vlm] make_synthetic failed ({exc}); using built-in synthetic tiles")
            return _local_synthetic(dataset, split, limit or 16, seed)
    return data.load_split(dataset, split, limit=limit, seed=seed)


@dataclass
class _Tile:
    tile_id: str
    image: Image.Image
    label: int


def _local_synthetic(dataset: str, split: str, n: int, seed: int) -> list[_Tile]:
    rng = random.Random(seed)
    k = len(schema.CLASSES[dataset])
    out = []
    for i in range(n):
        label = i % k
        col = tuple(rng.randrange(60, 220) for _ in range(3))
        out.append(_Tile(f"{dataset}:{split}:{i}", Image.new("RGB", (96, 96), col), label))
    return out
