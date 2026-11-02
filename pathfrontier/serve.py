"""ML inference service (docs/CONTRACTS.md section 8). FastAPI on port 8008.

    GET  /health
    POST /predict/tile      multipart image file, or JSON {"image_b64": ...}
    POST /predict/slide     JSON {"image_id", "level", "max_tiles", "low", "high", "fallback"}
    GET  /jobs/{job_id}
    GET  /heatmap/{image_id}
    POST /vqa               JSON {"image_id", "level"|null, "x", "y", "question"}
                            or   {"image_b64", "question"}  (PathVQA fine-tune, see vqa.py)
                            or   {"gallery_id", "question"} (image from the demo gallery)
    GET  /gallery           curated demo items (scripts/make_demo_gallery.py; see gallery.py)
    GET  /gallery/image/{id}
    POST /scan              JSON {"id": "scan_1"}: probe + separate fine-tuned VLM verdicts

Configuration (environment):

    PATHFRONTIER_ENCODER     encoder name (default "phikon-v2")
    PATHFRONTIER_PROBE       probe joblib (default results/probe_<encoder>_<dataset>.joblib)
    PATHFRONTIER_DATASET     label space, "pcam" (default) or "crc"
    PATHFRONTIER_TILES_DIR   local DZI pyramids (default tiles/)
    PATHFRONTIER_VLM_MODEL   optional Fireworks model id; used for the uncertain band only when
                             FIREWORKS_API_KEY is also set, otherwise band tiles -> "pathologist"
    PATHFRONTIER_VLM_PROMPT_VERSION  prompt for the band VLM: "v1" (default; JSON label/confidence/
                             reason) or "v2" (fine-tuned one-word tumor/normal classifier: p_tumor
                             from first-token logprobs, `reason` = the one-word answer)
    PATHFRONTIER_VLM_NO_THINKING  "1" (default) sends reasoning_effort="none" so hybrid-reasoning
                             models (Qwen3.5) answer instead of thinking; "0" leaves it to the model
    PATHFRONTIER_VQA_MODEL   Fireworks model id for POST /vqa (default vqa.DEFAULT_VQA_MODEL, the
                             PathVQA fine-tune); needs FIREWORKS_API_KEY unless PATHFRONTIER_FAKE=1
    PATHFRONTIER_LOW/HIGH    cascade thresholds (defaults 0.01 / 0.90)
    PATHFRONTIER_BG_SAT      mean HSV saturation below which a tile is glass/background (default 0.04)
    PATHFRONTIER_FAKE=1      "fake" encoder + deterministic random probe-like scorer, no downloads;
                             /vqa answers offline with a canned answer
    PATHFRONTIER_VLM_MAX_TILES  cap on VLM calls per slide job (default 200)
    PATHFRONTIER_GALLERY_DIR     demo gallery dir (default data/demo_gallery)
    PATHFRONTIER_SCAN_MODEL      Fireworks model id of the fine-tuned Camelyon17 classifier used by
                                 POST /scan; PATHFRONTIER_SCAN_API_KEY (default FIREWORKS_API_KEY),
                                 PATHFRONTIER_SCAN_CUTOFF (default 0.1068), PATHFRONTIER_SCAN_PROBE
                                 (default the camelyon17 Phikon-v2 probe). Unset -> the PathVQA
                                 fine-tune (PATHFRONTIER_VQA_MODEL) decides via "is tumor present?"
                                 when a key is present, else the probe alone (/health scan_backend)

Slide `level`: an int DZI level, "max" (full res), "auto" (max_level - 2, ~1 um/px for a 40x
slide) or null -> max_level - 1 (half res; 4x fewer tiles than full res, a good speed/detail
trade-off for the demo).

Run: `python -m pathfrontier.serve` (or scripts/serve.py).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathfrontier import config, dzi, gallery, vqa
from pathfrontier.schema import CLASSES, DEFAULT_COST_PER_1K, POSITIVE_INDEX, save_json

log = logging.getLogger("pathfrontier.serve")

PROBE_COST_PER_TILE = DEFAULT_COST_PER_1K["probe"] / 1000.0
BACKGROUND_SAT_THRESHOLD = float(os.getenv("PATHFRONTIER_BG_SAT", "0.04"))  # mean HSV saturation below this => glass
BACKGROUND_VAL_THRESHOLD = 0.80   # ...and mean brightness above this (white glass)
EMBED_BATCH = 64
VQA_CACHE_SIZE = 512


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclass
class Settings:
    encoder: str = "phikon-v2"
    dataset: str = "pcam"
    probe: Path | None = None
    tiles_dir: Path = field(default_factory=lambda: config.ROOT / "tiles")
    results_dir: Path = field(default_factory=lambda: config.RESULTS_DIR)
    vlm_model: str | None = None
    vlm_prompt_version: str = "v1"
    vlm_no_thinking: bool = True
    vqa_model: str | None = vqa.DEFAULT_VQA_MODEL
    low: float = 0.01
    high: float = 0.90
    fake: bool = False
    vlm_max_tiles: int = 200
    gallery_dir: Path = field(default_factory=gallery.default_gallery_dir)
    scan_model: str | None = None
    scan_api_key: str | None = None
    scan_cutoff: float | None = None      # None -> gallery.load_cutoff() (results json / 0.1068)
    scan_probe: Path | None = None        # None -> gallery.DEFAULT_SCAN_PROBE

    @classmethod
    def from_env(cls) -> Settings:
        enc = os.getenv("PATHFRONTIER_ENCODER", "phikon-v2")
        dataset = os.getenv("PATHFRONTIER_DATASET", "pcam")
        probe = os.getenv("PATHFRONTIER_PROBE")
        probe_path = Path(probe) if probe else config.RESULTS_DIR / f"probe_{enc}_{dataset}.joblib"
        truthy = {"1", "true", "yes"}
        fake = os.getenv("PATHFRONTIER_FAKE", "0").lower() in truthy
        return cls(
            encoder="fake" if fake else enc,
            dataset=dataset,
            probe=probe_path,
            tiles_dir=Path(os.getenv("PATHFRONTIER_TILES_DIR", config.ROOT / "tiles")),
            results_dir=config.RESULTS_DIR,
            vlm_model=os.getenv("PATHFRONTIER_VLM_MODEL") or None,
            vlm_prompt_version=os.getenv("PATHFRONTIER_VLM_PROMPT_VERSION", "v1").strip() or "v1",
            vlm_no_thinking=os.getenv("PATHFRONTIER_VLM_NO_THINKING", "1").lower() in truthy,
            vqa_model=os.getenv("PATHFRONTIER_VQA_MODEL", vqa.DEFAULT_VQA_MODEL).strip() or None,
            low=float(os.getenv("PATHFRONTIER_LOW", "0.01")),
            high=float(os.getenv("PATHFRONTIER_HIGH", "0.90")),
            fake=fake,
            vlm_max_tiles=int(os.getenv("PATHFRONTIER_VLM_MAX_TILES", "200")),
            gallery_dir=gallery.default_gallery_dir(),
            scan_model=os.getenv("PATHFRONTIER_SCAN_MODEL", "").strip() or None,
            scan_api_key=os.getenv("PATHFRONTIER_SCAN_API_KEY", "").strip() or None,
            scan_cutoff=float(os.getenv("PATHFRONTIER_SCAN_CUTOFF")) if os.getenv(
                "PATHFRONTIER_SCAN_CUTOFF") else None,
            scan_probe=Path(os.getenv("PATHFRONTIER_SCAN_PROBE")) if os.getenv(
                "PATHFRONTIER_SCAN_PROBE") else None,
        )

    @property
    def classes(self) -> list[str]:
        return CLASSES[self.dataset]

    @property
    def positive_index(self) -> int:
        return POSITIVE_INDEX[self.dataset]

    def heatmap_path(self, image_id: str) -> Path:
        return self.results_dir / "heatmaps" / f"{image_id}.json"


# --------------------------------------------------------------------------------------
# Scorer: encoder + probe (real or fake)
# --------------------------------------------------------------------------------------
class _FallbackFakeEncoder:
    """Deterministic, download-free encoder used when encoders.py has no 'fake' entry."""

    name = "fake"
    dim = 192

    def embed(self, images, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            small = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            v = small.reshape(-1)
            out[i] = (v - v.mean()) / (v.std() + 1e-6)
        return out


class _FakeProbe:
    """Untrained but deterministic logistic scorer so the UI works with no model download."""

    def __init__(self, dim: int, n_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((dim, n_classes)).astype(np.float32) / np.sqrt(dim)
        self.b = rng.standard_normal(n_classes).astype(np.float32) * 0.1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X.astype(np.float32) @ self.W + self.b
        z = z * 3.0  # spread scores so the fake heatmap has clear/flag/review variety
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class Scorer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.device = config.device()
        self.encoder = self._load_encoder()
        self.probe = self._load_probe()
        self.model_name = f"probe/{self.settings.encoder}"

    def _load_encoder(self):
        s = self.settings
        try:
            from pathfrontier.encoders import get_encoder
        except ImportError as e:
            if s.fake:
                log.warning("encoders.py unavailable (%s); using built-in fake encoder", e)
                return _FallbackFakeEncoder()
            raise RuntimeError(
                "pathfrontier.encoders could not be imported; set PATHFRONTIER_FAKE=1 "
                "to run without models"
            ) from e
        try:
            return get_encoder("fake" if s.fake else s.encoder, self.device)
        except Exception as e:
            if s.fake:
                log.warning("get_encoder('fake') failed (%s); using built-in fake encoder", e)
                return _FallbackFakeEncoder()
            raise RuntimeError(f"failed to load encoder {s.encoder!r}: {e}") from e

    def _load_probe(self):
        s = self.settings
        if s.fake:
            return _FakeProbe(int(getattr(self.encoder, "dim", 192)), len(s.classes))
        if s.probe is None or not Path(s.probe).is_file():
            raise RuntimeError(
                f"probe artifact not found at {s.probe}; train one with scripts/train_probe.py "
                f"or set PATHFRONTIER_PROBE (or PATHFRONTIER_FAKE=1 for a demo)"
            )
        try:
            from pathfrontier.probe import LinearProbe
        except ImportError as e:
            raise RuntimeError("pathfrontier.probe could not be imported") from e
        return LinearProbe.load(Path(s.probe))

    def embed(self, images) -> np.ndarray:
        if not images:
            return np.zeros((0, int(getattr(self.encoder, "dim", 1))), dtype=np.float32)
        with self.lock:
            X = self.encoder.embed(images, batch_size=EMBED_BATCH)
        return np.asarray(X, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros((0, len(self.settings.classes)), dtype=np.float32)
        return np.asarray(self.probe.predict_proba(X), dtype=np.float32)


# --------------------------------------------------------------------------------------
# VLM fallback (lazy)
# --------------------------------------------------------------------------------------
def vlm_enabled(settings: Settings) -> bool:
    return bool(settings.vlm_model) and bool(os.getenv("FIREWORKS_API_KEY", config.FIREWORKS_API_KEY))


def _vlm_config(settings: Settings):
    """VLMConfig for the band model, mirroring vlm_client.run_split's prompt-version handling."""
    from pathfrontier import vlm_client

    cfg = vlm_client.VLMConfig(model=settings.vlm_model)
    if settings.vlm_prompt_version in vlm_client.ONE_WORD_PROMPTS:
        cfg.min_image_px, cfg.max_tokens = vlm_client.V2_IMAGE_PX, vlm_client.V2_MAX_TOKENS
        cfg.extra.update({"logprobs": True, "top_logprobs": vlm_client.V2_TOP_LOGPROBS})
    if settings.vlm_no_thinking:  # hybrid-reasoning models (Qwen3.5) otherwise think, not answer
        cfg.extra["extra_body"] = {"reasoning_effort": "none"}
    return cfg


def _vlm_classify(image, settings: Settings) -> dict:
    """classify_tile wrapper; returns the vlm_client dict. Monkeypatched in tests.

    v1: unchanged baseline JSON prompt. v2 (fine-tuned one-word classifier): p_tumor comes from
    the first-token logprobs (classify_tile already folds them into `probs`) and `reason` is the
    model's one-word answer.
    """
    try:
        from pathfrontier.vlm_client import classify_tile
    except ImportError as e:
        raise RuntimeError(f"VLM fallback requested but vlm_client unavailable: {e}") from e
    cfg = _vlm_config(settings)
    r = classify_tile(image, settings.dataset, cfg, prompt_version=settings.vlm_prompt_version)
    if settings.vlm_prompt_version != "v1":
        word = (r.get("raw") or "").strip().split()
        r["reason"] = word[0].strip(".,!\"'").lower() if word else (r.get("reason") or None)
    return r


def _scan_vlm(image, model: str, api_key: str | None, cutoff: float) -> dict:
    """gallery.vlm_scan wrapper for POST /scan (same prompt v2 setup as _vlm_classify).
    Monkeypatched in tests."""
    return gallery.vlm_scan(image, model, api_key, cutoff)


def _scan_vqa(image, model: str, api_key: str | None) -> dict:
    """gallery.vqa_scan wrapper for POST /scan when no scan model is set: the PathVQA fine-tune
    answers "is tumor present?" with logprobs. Monkeypatched in tests."""
    return gallery.vqa_scan(image, model, api_key)


# --------------------------------------------------------------------------------------
# PathVQA (lazy; see vqa.py)
# --------------------------------------------------------------------------------------
def vqa_enabled(settings: Settings) -> bool:
    if settings.fake:
        return True
    return bool(settings.vqa_model) and bool(
        os.getenv("FIREWORKS_API_KEY", config.FIREWORKS_API_KEY))


def vqa_model_name(settings: Settings) -> str | None:
    if not vqa_enabled(settings):
        return None
    return "fake/vqa" if settings.fake else settings.vqa_model


def _vqa_answer(image, question: str, settings: Settings) -> dict:
    """vqa.answer_question wrapper; raises vqa.VQAError. Monkeypatched in tests."""
    return vqa.answer_question(image, question, model=settings.vqa_model or vqa.DEFAULT_VQA_MODEL,
                               fake=settings.fake)


class ScanRequest(BaseModel):
    id: str


class VQARequest(BaseModel):
    question: str
    image_id: str | None = None
    gallery_id: str | None = None
    level: int | None = None
    x: int | None = None
    y: int | None = None
    image_b64: str | None = None


class _VQACache:
    """Tiny thread-safe LRU so repeated demo clicks on the same tile+question are instant."""

    def __init__(self, size: int = VQA_CACHE_SIZE):
        self.size = size
        self._d: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: tuple, value: dict) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


# --------------------------------------------------------------------------------------
# Tile decisions
# --------------------------------------------------------------------------------------
def is_background(image) -> bool:
    """Mostly-white / unstained tile: low mean saturation and high brightness."""
    hsv = np.asarray(image.convert("RGB").resize((32, 32)).convert("HSV"), dtype=np.float32)
    sat = hsv[..., 1].mean() / 255.0
    val = hsv[..., 2].mean() / 255.0
    return sat < BACKGROUND_SAT_THRESHOLD and val > BACKGROUND_VAL_THRESHOLD


def decide(p_tumor: float, low: float, high: float) -> str:
    if p_tumor < low:
        return "negative"
    if p_tumor > high:
        return "positive"
    return "review"


def resolve_band(image, probs: np.ndarray, settings: Settings, fallback: str,
                 vlm_budget: list[int] | None = None) -> dict:
    """Route a tile in the uncertain band to the VLM (if configured) or the pathologist."""
    pos = settings.positive_index
    use_vlm = fallback == "vlm_ft" and vlm_enabled(settings)
    if use_vlm and vlm_budget is not None:
        if vlm_budget[0] <= 0:
            use_vlm = False
        else:
            vlm_budget[0] -= 1
    if use_vlm:
        try:
            r = _vlm_classify(image, settings)
            vprobs = [float(p) for p in r.get("probs", [])] or list(map(float, probs))
            label = int(r.get("label", int(np.argmax(vprobs))))
            return {
                "p_tumor": float(vprobs[pos]) if len(vprobs) > pos else float(probs[pos]),
                "probs": vprobs,
                "decision": "positive" if label == pos else "negative",
                "routed_to": "vlm_ft",
                "reason": r.get("reason"),
                "latency_ms": float(r.get("latency_ms", 0.0)),
                "cost_usd": float(r.get("cost_usd", 0.0)),
            }
        except Exception as e:  # noqa: BLE001 - any VLM failure -> human path, never crash the job
            log.warning("VLM call failed, routing to pathologist: %s", e)
    return {
        "p_tumor": float(probs[pos]),
        "probs": list(map(float, probs)),
        "decision": "review",
        "routed_to": "pathologist",
        "reason": None,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }


def score_tile(scorer: Scorer, image, low: float, high: float, fallback: str = "vlm_ft") -> dict:
    t0 = time.perf_counter()
    s = scorer.settings
    X = scorer.embed([image])
    probs = scorer.predict_proba(X)[0]
    probe_ms = (time.perf_counter() - t0) * 1000.0
    p = float(probs[s.positive_index])
    decision = decide(p, low, high)
    out = {
        "p_tumor": p,
        "probs": list(map(float, probs)),
        "classes": s.classes,
        "decision": decision,
        "routed_to": "probe",
        "reason": None,
        "latency_ms": probe_ms,
        "cost_usd": PROBE_COST_PER_TILE,
    }
    if decision == "review":
        r = resolve_band(image, probs, s, fallback)
        out.update({k: r[k] for k in ("p_tumor", "probs", "decision", "routed_to", "reason")})
        out["latency_ms"] += r["latency_ms"]
        out["cost_usd"] += r["cost_usd"]
    return out


# --------------------------------------------------------------------------------------
# Slide jobs
# --------------------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    image_id: str
    status: str = "running"      # running | done | error
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = {"job_id": self.job_id, "image_id": self.image_id, "status": self.status,
             "progress": round(self.progress, 4), "result": self.result}
        if self.error:
            d["error"] = self.error
        return d


class SlideRequest(BaseModel):
    image_id: str
    level: int | str | None = None
    max_tiles: int | None = None
    low: float | None = None
    high: float | None = None
    fallback: str = "vlm_ft"      # vlm_ft | pathologist | none


def choose_level(level: int | str | None, max_level: int) -> int:
    if level is None:
        return max(0, max_level - 1)
    if isinstance(level, str):
        key = level.strip().lower()
        if key in {"max", "full"}:
            return max_level
        if key == "auto":
            return max(0, max_level - 2)
        if key.lstrip("-").isdigit():
            level = int(key)
        else:
            raise ValueError(f"unknown level {level!r}; use an int, 'max', 'auto' or null")
    if level < 0 or level > max_level:
        raise ValueError(f"level {level} outside [0, {max_level}]")
    return int(level)


def run_slide(scorer: Scorer, req: SlideRequest, job: Job | None = None) -> dict:
    """Score every DZI tile of a slide at one level and build the heatmap JSON."""
    from PIL import Image

    s = scorer.settings
    low = s.low if req.low is None else req.low
    high = s.high if req.high is None else req.high
    fallback = req.fallback if req.fallback in {"vlm_ft", "pathologist", "none"} else "vlm_ft"
    dzi_file = dzi.find_dzi(s.tiles_dir, req.image_id)
    if dzi_file is None:
        raise FileNotFoundError(f"no DZI pyramid for {req.image_id!r} in {s.tiles_dir}")
    info = dzi.parse_dzi(dzi_file)
    max_level = info["max_level"]
    level = choose_level(req.level, max_level)
    cols, rows = dzi.level_grid(info["width"], info["height"], level, max_level, info["tile_size"])

    def set_progress(p: float):
        if job is not None:
            job.progress = min(max(p, 0.0), 1.0)

    t_start = time.perf_counter()
    paths = list(dzi.iter_tiles(s.tiles_dir, req.image_id, level, info))
    if req.max_tiles is not None:
        paths = paths[: max(0, int(req.max_tiles))]
    n_total = max(len(paths), 1)

    # Pass 1: load, drop background, embed in batches.
    tiles: list[dict] = []
    tissue_idx: list[int] = []
    images: list[Any] = []
    for i, (x, y, p) in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.load()
        if is_background(im):
            tiles.append({"x": x, "y": y, "p_tumor": 0.0, "decision": "background",
                          "routed_to": None, "reason": None})
        else:
            tiles.append({"x": x, "y": y, "p_tumor": None, "decision": None,
                          "routed_to": "probe", "reason": None})
            tissue_idx.append(len(tiles) - 1)
            images.append(im)
        if i % 50 == 0:
            set_progress(0.05 * i / n_total)

    embs = []
    for b in range(0, len(images), EMBED_BATCH):
        embs.append(scorer.embed(images[b: b + EMBED_BATCH]))
        set_progress(0.05 + 0.75 * min(1.0, (b + EMBED_BATCH) / max(len(images), 1)))
    X = np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)
    probs = scorer.predict_proba(X) if len(X) else np.zeros((0, len(s.classes)))
    probe_ms = (time.perf_counter() - t_start) * 1000.0
    set_progress(0.82)

    # Pass 2: thresholds + band resolution.
    pos = s.positive_index
    cost = PROBE_COST_PER_TILE * len(tissue_idx)
    latency_total = probe_ms
    counts = {"auto_cleared": 0, "auto_flagged": 0, "review": 0, "vlm_decided": 0, "background": 0}
    vlm_budget = [s.vlm_max_tiles]
    band = []
    for k, ti in enumerate(tissue_idx):
        p = float(probs[k][pos])
        d = decide(p, low, high)
        tiles[ti]["p_tumor"] = p
        tiles[ti]["decision"] = d
        if d == "negative":
            counts["auto_cleared"] += 1
        elif d == "positive":
            counts["auto_flagged"] += 1
        else:
            band.append((ti, k))
    counts["background"] = len(tiles) - len(tissue_idx)

    for j, (ti, k) in enumerate(band):
        r = resolve_band(images[k], probs[k], s, fallback, vlm_budget)
        tiles[ti].update({kk: r[kk] for kk in ("p_tumor", "decision", "routed_to", "reason")})
        cost += r["cost_usd"]
        latency_total += r["latency_ms"]
        if r["routed_to"] == "vlm_ft":
            counts["vlm_decided"] += 1
        else:
            counts["review"] += 1
        if j % 5 == 0:
            set_progress(0.82 + 0.17 * (j + 1) / max(len(band), 1))

    n_tissue = len(tissue_idx)
    model = scorer.model_name + (
        " + vlm_ft" if fallback == "vlm_ft" and vlm_enabled(s) else " + pathologist"
    )
    summary = {
        "n_tiles": n_tissue,
        "n_tiles_total": len(tiles),
        "background": counts["background"],
        "auto_cleared": counts["auto_cleared"],
        "auto_flagged": counts["auto_flagged"],
        "review": counts["review"],
        "vlm_decided": counts["vlm_decided"],
        "frac_auto_cleared": (counts["auto_cleared"] / n_tissue) if n_tissue else 0.0,
        "frac_automated": ((n_tissue - counts["review"]) / n_tissue) if n_tissue else 0.0,
        "cost_usd": cost,
        "latency_ms_total": latency_total,
        "pathologist_minutes_saved": counts["auto_cleared"] * config.PATHOLOGIST_SEC_PER_TILE / 60.0,
        "model": model,
        "low": low,
        "high": high,
        "fallback": fallback,
    }
    heatmap = {
        "image_id": req.image_id,
        "width": info["width"],
        "height": info["height"],
        "tile_size": info["tile_size"],
        "level": level,
        "max_level": max_level,
        "scale": dzi.level_scale(level, max_level),
        "cols": cols,
        "rows": rows,
        "tiles": tiles,
        "summary": summary,
        "created_at": time.time(),
    }
    save_json(heatmap, s.heatmap_path(req.image_id))
    set_progress(1.0)
    return heatmap


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def _decode_image_bytes(data: bytes):
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode image: {e}") from e


def _safe_id(image_id: str) -> str:
    if not image_id or "/" in image_id or "\\" in image_id or image_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid image_id")
    return image_id


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.time()
        app.state.scorer = Scorer(settings)
        log.info("loaded %s (%s) in %.1fs; vlm=%s", settings.encoder, app.state.scorer.device,
                 time.time() - t0, settings.vlm_model if vlm_enabled(settings) else None)
        yield

    app = FastAPI(title="pathfrontier inference", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.state.settings = settings
    app.state.jobs = jobs

    def scorer() -> Scorer:
        sc = getattr(app.state, "scorer", None)
        if sc is None:
            raise HTTPException(status_code=503, detail="model not loaded yet")
        return sc

    @app.get("/health")
    def health():
        sc = getattr(app.state, "scorer", None)
        return {
            "status": "ok" if sc is not None else "loading",
            "encoder": settings.encoder,
            "probe": "fake" if settings.fake else str(settings.probe),
            "vlm": settings.vlm_model if vlm_enabled(settings) else None,
            "vlm_prompt_version": settings.vlm_prompt_version,
            "vqa": vqa_model_name(settings),
            "device": sc.device if sc is not None else config.device(),
            "dataset": settings.dataset,
            "classes": settings.classes,
            "fake": settings.fake,
            "tiles_dir": str(settings.tiles_dir),
            "images": dzi.list_image_ids(settings.tiles_dir),
            "low": settings.low,
            "high": settings.high,
            "scan_model": settings.scan_model if scan_engine.vlm_enabled() else None,
            "scan_backend": scan_engine.scan_backend(),
            "gallery": gallery.counts(gallery.load_manifest(settings.gallery_dir)),
        }

    @app.post("/predict/tile")
    async def predict_tile(request: Request):
        ctype = request.headers.get("content-type", "")
        low, high, fallback = settings.low, settings.high, "vlm_ft"
        data: bytes | None = None
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            for key in ("file", "image", "tile"):
                if key in form and hasattr(form[key], "read"):
                    data = await form[key].read()
                    break
            if data is None:
                for v in form.values():
                    if hasattr(v, "read"):
                        data = await v.read()
                        break
            low = float(form.get("low", low))
            high = float(form.get("high", high))
            fallback = str(form.get("fallback", fallback))
        else:
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(status_code=400, detail="send multipart 'file' or JSON "
                                    "{'image_b64': ...}") from e
            b64 = body.get("image_b64") if isinstance(body, dict) else None
            if not b64:
                raise HTTPException(status_code=400, detail="missing image_b64")
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                data = base64.b64decode(b64)
            except Exception as e:
                raise HTTPException(status_code=400, detail="image_b64 is not base64") from e
            low = float(body.get("low", low))
            high = float(body.get("high", high))
            fallback = str(body.get("fallback", fallback))
        if not data:
            raise HTTPException(status_code=400, detail="no image data received")
        image = _decode_image_bytes(data)
        return score_tile(scorer(), image, low, high, fallback)

    @app.post("/predict/slide")
    def predict_slide(req: SlideRequest):
        _safe_id(req.image_id)
        sc = scorer()
        dzi_file = dzi.find_dzi(settings.tiles_dir, req.image_id)
        if dzi_file is None:
            raise HTTPException(status_code=404, detail=f"no DZI pyramid for {req.image_id!r} "
                                f"in {settings.tiles_dir}")
        try:
            choose_level(req.level, dzi.parse_dzi(dzi_file)["max_level"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        job = Job(job_id=uuid.uuid4().hex[:12], image_id=req.image_id)
        with jobs_lock:
            jobs[job.job_id] = job

        def _work():
            try:
                job.result = run_slide(sc, req, job)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:  # surfaced via /jobs
                log.exception("slide job %s failed", job.job_id)
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"

        threading.Thread(target=_work, name=f"slide-{job.job_id}", daemon=True).start()
        return {"job_id": job.job_id, "status": "running", "image_id": req.image_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.public()

    @app.get("/jobs")
    def list_jobs():
        return [dict(j.public(), result=None) for j in jobs.values()]

    @app.get("/heatmap/{image_id}")
    def get_heatmap(image_id: str):
        p = settings.heatmap_path(_safe_id(image_id))
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"no heatmap for {image_id!r}; "
                                "POST /predict/slide first")
        with open(p) as f:
            return json.load(f)

    @app.get("/images")
    def list_images():
        return {"images": dzi.list_image_ids(settings.tiles_dir)}

    scan_engine = gallery.ScanEngine(settings, scorer)
    app.state.scan_engine = scan_engine

    def gallery_item(item_id: str) -> dict:
        _safe_id(item_id)
        m = gallery.load_manifest(settings.gallery_dir)
        if m is None:
            raise HTTPException(status_code=404, detail=f"demo gallery not built at "
                                f"{settings.gallery_dir}; run `{gallery.BUILD_CMD}`")
        it = gallery.items_by_id(m).get(item_id)
        if it is None:
            raise HTTPException(status_code=404, detail=f"unknown gallery item {item_id!r}")
        return it

    @app.get("/gallery")
    def get_gallery():
        m = gallery.load_manifest(settings.gallery_dir)
        if m is None:
            raise HTTPException(status_code=404, detail=f"demo gallery not built at "
                                f"{settings.gallery_dir}; run `{gallery.BUILD_CMD}`")
        return gallery.public_manifest(m)

    @app.get("/gallery/image/{item_id}")
    def get_gallery_image(item_id: str):
        it = gallery_item(item_id)
        p = gallery.image_path(settings.gallery_dir, it)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"image file missing for {item_id!r}")
        return FileResponse(p, media_type=gallery.media_type(p),
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/scan")
    def scan_endpoint(req: ScanRequest):
        it = gallery_item(req.id)
        if it.get("section") != "scan":
            raise HTTPException(status_code=400, detail=f"{req.id!r} is not a scan item")
        if not settings.fake:
            scorer()  # 503 until the encoder is loaded
        return scan_engine.cached_scan(it, lambda: gallery.open_image(settings.gallery_dir, it),
                                       vlm_fn=_scan_vlm, vqa_fn=_scan_vqa)

    vqa_cache = _VQACache()
    app.state.vqa_cache = vqa_cache

    @app.post("/vqa")
    def vqa_endpoint(req: VQARequest):
        from PIL import Image

        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="missing question")
        if not vqa_enabled(settings):
            raise HTTPException(status_code=503, detail={"error": "vqa_not_configured"})
        model = vqa_model_name(settings)

        image_id = level = x = y = None
        gallery_id = None
        if req.gallery_id is not None:
            it = gallery_item(req.gallery_id)
            gallery_id = req.gallery_id
            key = (model, "gallery:" + gallery_id, None, None, None, question)
            hit = vqa_cache.get(key)
            if hit is not None:
                return dict(hit, cached=True)
            try:
                image = gallery.open_image(settings.gallery_dir, it)
            except OSError as e:
                raise HTTPException(status_code=404, detail=f"image file missing for "
                                    f"{gallery_id!r}: {e}") from e
        elif req.image_id is not None:
            if req.x is None or req.y is None:
                raise HTTPException(status_code=400, detail="image_id needs x and y")
            image_id = _safe_id(req.image_id)
            dzi_file = dzi.find_dzi(settings.tiles_dir, image_id)
            if dzi_file is None:
                raise HTTPException(status_code=404, detail=f"no DZI pyramid for {image_id!r} "
                                    f"in {settings.tiles_dir}")
            info = dzi.parse_dzi(dzi_file)
            level = info["max_level"] if req.level is None else int(req.level)
            x, y = int(req.x), int(req.y)
            key = (model, image_id, level, x, y, question)
            hit = vqa_cache.get(key)
            if hit is not None:
                return dict(hit, cached=True)
            tile = dzi.tile_path(settings.tiles_dir, image_id, level, x, y, info["format"])
            if level < 0 or level > info["max_level"] or not tile.is_file():
                raise HTTPException(status_code=404, detail=f"no tile {image_id}/{level}/{x}_{y}")
            with Image.open(tile) as im:
                image = im.convert("RGB")
                image.load()
        elif req.image_b64:
            b64 = req.image_b64.strip()
            if b64.startswith("data:") and "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                data = base64.b64decode(b64)
            except Exception as e:
                raise HTTPException(status_code=400, detail="image_b64 is not base64") from e
            image = _decode_image_bytes(data)
            key = (model, "b64:" + hashlib.sha1(data).hexdigest(), None, None, None, question)
            hit = vqa_cache.get(key)
            if hit is not None:
                return dict(hit, cached=True)
        else:
            raise HTTPException(status_code=400, detail="send {image_id, level, x, y, question} "
                                "or {image_b64, question}")

        try:
            r = _vqa_answer(image, question, settings)
        except vqa.VQAError as e:
            raise HTTPException(status_code=502, detail=f"vqa upstream error: {e}") from e
        out = {
            "answer": r["answer"],
            "question": question,
            "model": r.get("model", model),
            "latency_ms": float(r.get("latency_ms", 0.0)),
            "cost_usd": float(r.get("cost_usd", 0.0)),
            "tokens_in": int(r.get("tokens_in", 0)),
            "tokens_out": int(r.get("tokens_out", 0)),
            "image_id": image_id,
            "level": level,
            "x": x,
            "y": y,
            "gallery_id": gallery_id,
        }
        vqa_cache.put(key, out)
        return dict(out, cached=False)

    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 8008, reload: bool = False) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("pathfrontier.serve:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main(port=int(os.getenv("PORT", "8008")))
