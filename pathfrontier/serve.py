"""ML inference service (docs/CONTRACTS.md section 8). FastAPI on port 8008.

    GET  /health
    POST /predict/tile      multipart image file, or JSON {"image_b64": ...}
    POST /predict/slide     JSON {"image_id", "level", "max_tiles", "low", "high", "fallback"}
    GET  /jobs/{job_id}
    GET  /heatmap/{image_id}

Configuration (environment):

    PATHFRONTIER_ENCODER     encoder name (default "phikon-v2")
    PATHFRONTIER_PROBE       probe joblib (default results/probe_<encoder>_<dataset>.joblib)
    PATHFRONTIER_DATASET     label space, "pcam" (default) or "crc"
    PATHFRONTIER_TILES_DIR   local DZI pyramids (default tiles/)
    PATHFRONTIER_VLM_MODEL   optional Fireworks model id; used for the uncertain band only when
                             FIREWORKS_API_KEY is also set, otherwise band tiles -> "pathologist"
    PATHFRONTIER_LOW/HIGH    cascade thresholds (defaults 0.01 / 0.90)
    PATHFRONTIER_BG_SAT      mean HSV saturation below which a tile is glass/background (default 0.04)
    PATHFRONTIER_FAKE=1      "fake" encoder + deterministic random probe-like scorer, no downloads
    PATHFRONTIER_VLM_MAX_TILES  cap on VLM calls per slide job (default 200)

Slide `level`: an int DZI level, "max" (full res), "auto" (max_level - 2, ~1 um/px for a 40x
slide) or null -> max_level - 1 (half res; 4x fewer tiles than full res, a good speed/detail
trade-off for the demo).

Run: `python -m pathfrontier.serve` (or scripts/serve.py).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathfrontier import config, dzi
from pathfrontier.schema import CLASSES, DEFAULT_COST_PER_1K, POSITIVE_INDEX, save_json

log = logging.getLogger("pathfrontier.serve")

PROBE_COST_PER_TILE = DEFAULT_COST_PER_1K["probe"] / 1000.0
BACKGROUND_SAT_THRESHOLD = float(os.getenv("PATHFRONTIER_BG_SAT", "0.04"))  # mean HSV saturation below this => glass
BACKGROUND_VAL_THRESHOLD = 0.80   # ...and mean brightness above this (white glass)
EMBED_BATCH = 64


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
    low: float = 0.01
    high: float = 0.90
    fake: bool = False
    vlm_max_tiles: int = 200

    @classmethod
    def from_env(cls) -> Settings:
        enc = os.getenv("PATHFRONTIER_ENCODER", "phikon-v2")
        dataset = os.getenv("PATHFRONTIER_DATASET", "pcam")
        probe = os.getenv("PATHFRONTIER_PROBE")
        probe_path = Path(probe) if probe else config.RESULTS_DIR / f"probe_{enc}_{dataset}.joblib"
        fake = os.getenv("PATHFRONTIER_FAKE", "0").lower() in {"1", "true", "yes"}
        return cls(
            encoder="fake" if fake else enc,
            dataset=dataset,
            probe=probe_path,
            tiles_dir=Path(os.getenv("PATHFRONTIER_TILES_DIR", config.ROOT / "tiles")),
            results_dir=config.RESULTS_DIR,
            vlm_model=os.getenv("PATHFRONTIER_VLM_MODEL") or None,
            low=float(os.getenv("PATHFRONTIER_LOW", "0.01")),
            high=float(os.getenv("PATHFRONTIER_HIGH", "0.90")),
            fake=fake,
            vlm_max_tiles=int(os.getenv("PATHFRONTIER_VLM_MAX_TILES", "200")),
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


def _vlm_classify(image, settings: Settings) -> dict:
    """classify_tile wrapper; returns the vlm_client dict. Monkeypatched in tests."""
    try:
        from pathfrontier.vlm_client import VLMConfig, classify_tile
    except ImportError as e:
        raise RuntimeError(f"VLM fallback requested but vlm_client unavailable: {e}") from e
    cfg = VLMConfig(model=settings.vlm_model)
    return classify_tile(image, settings.dataset, cfg)


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
            "device": sc.device if sc is not None else config.device(),
            "dataset": settings.dataset,
            "classes": settings.classes,
            "fake": settings.fake,
            "tiles_dir": str(settings.tiles_dir),
            "images": dzi.list_image_ids(settings.tiles_dir),
            "low": settings.low,
            "high": settings.high,
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

    return app


app = create_app()


def main(host: str = "0.0.0.0", port: int = 8008, reload: bool = False) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("pathfrontier.serve:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main(port=int(os.getenv("PORT", "8008")))
