"""Curated demo gallery (judge-facing demo page): manifest loading + the /scan decision logic.

The gallery lives in ``data/demo_gallery/`` (``PATHFRONTIER_GALLERY_DIR``) and is built by
``scripts/make_demo_gallery.py``: a ``manifest.json`` with two sections plus the image files.

    scan items: Camelyon17 (or PCam fallback) tiles with a truth label -> POST /scan
    ask  items: PathVQA images with their reference questions      -> POST /vqa {"gallery_id"}

/scan runs two independent classifiers on the same tile and reports both:

    probe: the service's Phikon-v2 encoder + the Camelyon17 linear probe
           (``PATHFRONTIER_SCAN_PROBE``, default results/camelyon17/probe_phikon-v2_camelyon17_
           c17train.joblib; falls back to the service's own probe if that file is missing)
    vlm:   a separate fine-tuned one-word classifier on Fireworks (``PATHFRONTIER_SCAN_MODEL``,
           key ``PATHFRONTIER_SCAN_API_KEY`` defaulting to FIREWORKS_API_KEY), prompt v2 exactly
           as serve._vlm_classify does, p_tumor from logprobs, verdict = p_tumor >= cutoff
           (``PATHFRONTIER_SCAN_CUTOFF``, default results/camelyon17/cutoff_qwen3p5-9b_ft_v2.json)

Decision order (``scan_backend`` in GET /health):

    camelyon-lora  PATHFRONTIER_SCAN_MODEL set (+ key): the Camelyon17 one-word classifier above
    pathvqa-lora   else the PathVQA fine-tune (``Settings.vqa_model``, vqa.DEFAULT_VQA_MODEL) is
                   asked the fixed question "is tumor present?" through the same client/prompt
                   as POST /vqa, with first-token logprobs: p_tumor = P("yes") over {yes, no},
                   verdict = tumor iff the answer starts with "yes"
    probe          else (or PATHFRONTIER_FAKE=1 -> "fake")

The chosen VLM decides when it succeeds; on any VLM error the probe decides (the probe always
runs as the second opinion).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from pathfrontier import config

log = logging.getLogger("pathfrontier.gallery")

MANIFEST_NAME = "manifest.json"
BUILD_CMD = ".venv/bin/python scripts/make_demo_gallery.py"
DEFAULT_SCAN_PROBE = config.RESULTS_DIR / "camelyon17" / "probe_phikon-v2_camelyon17_c17train.joblib"
DEFAULT_SCAN_CUTOFF = 0.1068   # results/camelyon17/cutoff_qwen3p5-9b_ft_v2.json (slide-29 dev set)
PROBE_THRESHOLD = 0.5
VQA_SCAN_QUESTION = "is tumor present?"
VQA_SCAN_MAX_TOKENS = 4
VQA_SCAN_TOP_LOGPROBS = 5
VQA_SCAN_SOURCE = "pathvqa-lora"
SCAN_DATASET = "camelyon17"    # binary normal/tumor label space for the v2 prompt
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp"}


def default_gallery_dir() -> Path:
    return Path(os.getenv("PATHFRONTIER_GALLERY_DIR") or config.DATA_DIR / "demo_gallery")


def manifest_path(gallery_dir: Path | str | None = None) -> Path:
    return Path(gallery_dir or default_gallery_dir()) / MANIFEST_NAME


def load_manifest(gallery_dir: Path | str | None = None) -> dict | None:
    """The manifest dict ({"scan": [...], "ask": [...], ...}) or None when it is not built."""
    p = manifest_path(gallery_dir)
    if not p.is_file():
        return None
    with open(p) as f:
        m = json.load(f)
    m.setdefault("scan", [])
    m.setdefault("ask", [])
    return m


def items_by_id(manifest: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for section in ("scan", "ask"):
        for it in manifest.get(section, []):
            out[it["id"]] = it
    return out


def get_item(gallery_dir: Path | str | None, item_id: str) -> dict | None:
    m = load_manifest(gallery_dir)
    if m is None:
        return None
    return items_by_id(m).get(item_id)


def image_path(gallery_dir: Path | str | None, item: dict) -> Path:
    name = Path(str(item.get("file", ""))).name   # never escape the gallery dir
    return Path(gallery_dir or default_gallery_dir()) / name


def media_type(path: Path | str) -> str:
    return MEDIA_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def open_image(gallery_dir: Path | str | None, item: dict):
    from PIL import Image

    with Image.open(image_path(gallery_dir, item)) as im:
        out = im.convert("RGB")
        out.load()
    return out


def public_manifest(manifest: dict) -> dict:
    """Manifest as served by GET /gallery: every item gains an image_url."""
    out = {k: v for k, v in manifest.items() if k not in ("scan", "ask")}
    for section in ("scan", "ask"):
        out[section] = [dict(it, image_url=f"/gallery/image/{it['id']}")
                        for it in manifest.get(section, [])]
    return out


def counts(manifest: dict | None) -> dict[str, int]:
    if manifest is None:
        return {"scan": 0, "ask": 0}
    return {"scan": len(manifest.get("scan", [])), "ask": len(manifest.get("ask", []))}


# --------------------------------------------------------------------------------------
# Scan: probe + separate fine-tuned VLM
# --------------------------------------------------------------------------------------
def load_cutoff(path: Path | None = None) -> float:
    env = os.getenv("PATHFRONTIER_SCAN_CUTOFF")
    if env:
        return float(env)
    p = path or config.RESULTS_DIR / "camelyon17" / "cutoff_qwen3p5-9b_ft_v2.json"
    try:
        with open(p) as f:
            return float(json.load(f)["cutoff"])
    except (OSError, KeyError, ValueError, TypeError):
        return DEFAULT_SCAN_CUTOFF


def fake_p_tumor(item_id: str, label: int) -> float:
    """Deterministic per id and on the right side of 0.5 for the truth label (offline demo)."""
    h = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) / float(1 << 32)
    return 0.80 + 0.17 * h if int(label) == 1 else 0.03 + 0.17 * h


def verdict_of(p_tumor: float, cutoff: float) -> str:
    return "tumor" if p_tumor >= cutoff else "normal"


def vlm_scan_config(model: str, api_key: str | None = None):
    """VLMConfig identical to serve._vlm_config with prompt v2 + thinking off."""
    from pathfrontier import vlm_client

    cfg = vlm_client.VLMConfig(model=model, api_key=api_key or config.FIREWORKS_API_KEY)
    cfg.min_image_px, cfg.max_tokens = vlm_client.V2_IMAGE_PX, vlm_client.V2_MAX_TOKENS
    cfg.extra.update({"logprobs": True, "top_logprobs": vlm_client.V2_TOP_LOGPROBS})
    cfg.extra["extra_body"] = {"reasoning_effort": "none"}
    return cfg


def vlm_scan(image, model: str, api_key: str | None, cutoff: float) -> dict:
    """One tile through the scan VLM. Raises on any API/parse failure (caller reports it)."""
    from pathfrontier.vlm_client import classify_tile

    cfg = vlm_scan_config(model, api_key)
    r = classify_tile(image, SCAN_DATASET, cfg, prompt_version="v2")
    if r.get("api_error"):
        raise RuntimeError(r["api_error"])
    probs = [float(p) for p in r.get("probs", [])]
    if len(probs) != 2:
        raise RuntimeError(f"unexpected VLM probs {probs!r}")
    words = (r.get("raw") or "").strip().split()
    answer = words[0].strip(".,!\"'").lower() if words else None
    if r.get("parse_error") and answer not in ("tumor", "normal") and "logprob_p_tumor" not in r:
        raise RuntimeError(f"unparseable VLM answer {r.get('raw')!r}")
    p = probs[1]
    return {
        "verdict": verdict_of(p, cutoff),
        "p_tumor": p,
        "answer": answer,
        "latency_ms": float(r.get("latency_ms", 0.0)),
        "cost_usd": float(r.get("cost_usd", 0.0)),
        "model": model,
        "cutoff": cutoff,
    }


def vqa_scan(image, model: str, api_key: str | None) -> dict:
    """One tile through the PathVQA fine-tune: "is tumor present?" -> yes/no with logprobs.

    p_tumor = P("yes") renormalised over {yes, no} from the first token's top_logprobs; when the
    API returns no logprobs it is 0.95 for "yes" / 0.05 for "no". verdict = tumor iff the answer
    starts with "yes" (p_tumor >= 0.5 when the answer is neither). Raises on any API failure.
    """
    from pathfrontier import vqa

    client = vqa.make_client(api_key=api_key) if api_key else None
    r = vqa.answer_question(image, VQA_SCAN_QUESTION, model=model, client=client, logprobs=True,
                            top_logprobs=VQA_SCAN_TOP_LOGPROBS, max_tokens=VQA_SCAN_MAX_TOKENS)
    answer = (r.get("answer") or "").strip()
    word = answer.lower()
    says_yes = word.startswith("yes")
    says_no = word.startswith("no")
    p = r.get("p_yes")
    if p is None:
        p = 0.95 if says_yes else 0.05 if says_no else 0.5
    p = float(p)
    verdict = "tumor" if (says_yes or (not says_no and p >= 0.5)) else "normal"
    return {
        "verdict": verdict,
        "p_tumor": p,
        "answer": answer,
        "latency_ms": float(r.get("latency_ms", 0.0)),
        "cost_usd": float(r.get("cost_usd", 0.0)),
        "model": model,
        "question": VQA_SCAN_QUESTION,
        "source": VQA_SCAN_SOURCE,
    }


class ScanEngine:
    """Lazy holder for the scan probe + per-id result cache. One per app."""

    def __init__(self, settings: Any, scorer_fn):
        self.settings = settings
        self._scorer_fn = scorer_fn      # () -> serve.Scorer (raises 503 until loaded)
        self._probe = None
        self._probe_model: str | None = None
        self._probe_loaded = False
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.cutoff = float(getattr(settings, "scan_cutoff", None) or load_cutoff())

    # -- probe ---------------------------------------------------------------------------
    def _ensure_probe(self, scorer) -> tuple[Any, str]:
        with self._lock:
            if self._probe_loaded:
                return self._probe, self._probe_model or ""
            s = self.settings
            path = Path(getattr(s, "scan_probe", None) or DEFAULT_SCAN_PROBE)
            probe, name = None, None
            if not getattr(s, "fake", False) and path.is_file():
                try:
                    from pathfrontier.probe import LinearProbe

                    cand = LinearProbe.load(path)
                    enc_ok = getattr(cand, "encoder_name", s.encoder) == s.encoder
                    if enc_ok:
                        probe = cand
                        name = f"probe/{cand.encoder_name} ({cand.dataset})"
                    else:
                        log.warning("scan probe %s was trained on %s, service encoder is %s; "
                                    "using the service probe", path, cand.encoder_name, s.encoder)
                except Exception as e:  # noqa: BLE001 - never block the demo on a bad joblib
                    log.warning("could not load scan probe %s: %s", path, e)
            if probe is None:
                probe = scorer.probe
                name = (f"{scorer.model_name} ({s.dataset} fallback; scan probe missing at "
                        f"{path})")
            self._probe, self._probe_model, self._probe_loaded = probe, name, True
            return probe, name

    def probe_scan(self, item: dict, image) -> dict:
        s = self.settings
        t0 = time.perf_counter()
        if getattr(s, "fake", False):
            p = fake_p_tumor(item["id"], int(item.get("label", 0)))
            model = "fake/probe"
        else:
            scorer = self._scorer_fn()
            probe, model = self._ensure_probe(scorer)
            X = scorer.embed([image])
            probs = probe.predict_proba(X)[0]
            p = float(probs[1]) if len(probs) > 1 else float(probs[0])
        return {
            "verdict": verdict_of(p, PROBE_THRESHOLD),
            "p_tumor": p,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "model": model,
        }

    # -- full decision -------------------------------------------------------------------
    def vlm_enabled(self) -> bool:
        s = self.settings
        if getattr(s, "fake", False):
            return False
        return bool(getattr(s, "scan_model", None)) and bool(self.scan_api_key())

    def scan_api_key(self) -> str:
        return (getattr(self.settings, "scan_api_key", None)
                or os.getenv("FIREWORKS_API_KEY", config.FIREWORKS_API_KEY) or "")

    def vqa_api_key(self) -> str:
        return os.getenv("FIREWORKS_API_KEY", config.FIREWORKS_API_KEY) or ""

    def vqa_enabled(self) -> bool:
        """The PathVQA fine-tune decides when no Camelyon scan model is configured."""
        s = self.settings
        if getattr(s, "fake", False) or self.vlm_enabled():
            return False
        return bool(getattr(s, "vqa_model", None)) and bool(self.vqa_api_key())

    def scan_backend(self) -> str:
        if getattr(self.settings, "fake", False):
            return "fake"
        if self.vlm_enabled():
            return "camelyon-lora"
        if self.vqa_enabled():
            return VQA_SCAN_SOURCE
        return "probe"

    def scan(self, item: dict, image, vlm_fn=None, vqa_fn=None) -> dict:
        """Decision dict for one gallery item (no caching here; see `cached_scan`)."""
        s = self.settings
        probe = self.probe_scan(item, image)
        vlm: dict | None = None
        if self.vlm_enabled():
            fn = vlm_fn or vlm_scan
            try:
                vlm = fn(image, s.scan_model, self.scan_api_key(), self.cutoff)
            except Exception as e:  # noqa: BLE001 - VLM failure -> probe decides, report it
                log.warning("scan VLM failed for %s: %s", item["id"], e)
                vlm = {"error": f"{type(e).__name__}: {e}"[:300]}
        elif self.vqa_enabled():
            fn = vqa_fn or vqa_scan
            try:
                vlm = fn(image, s.vqa_model, self.vqa_api_key())
            except Exception as e:  # noqa: BLE001 - VQA failure -> probe decides, report it
                log.warning("scan VQA failed for %s: %s", item["id"], e)
                vlm = {"error": f"{type(e).__name__}: {e}"[:300], "source": VQA_SCAN_SOURCE}
        decider = vlm if vlm and "error" not in vlm else probe
        decided_by = "vlm" if decider is vlm else "probe"
        p = float(decider["p_tumor"])
        verdict = decider["verdict"]
        return {
            "id": item["id"],
            "verdict": verdict,
            "p_tumor": p,
            "confidence": p if verdict == "tumor" else 1.0 - p,
            "decided_by": decided_by,
            "model": decider["model"],
            "latency_ms": float(probe["latency_ms"]) + float((vlm or {}).get("latency_ms", 0.0)),
            "cost_usd": float((vlm or {}).get("cost_usd", 0.0)),
            "truth": {"label": int(item.get("label", 0)),
                      "label_name": item.get("label_name", "tumor" if item.get("label") else "normal")},
            "probe": probe,
            "vlm": vlm,
        }

    def cached_scan(self, item: dict, image_loader, vlm_fn=None, vqa_fn=None) -> dict:
        hit = self.cache.get(item["id"])
        if hit is not None:
            return dict(hit, cached=True)
        out = self.scan(item, image_loader(), vlm_fn, vqa_fn)
        self.cache[item["id"]] = out
        return dict(out, cached=False)
