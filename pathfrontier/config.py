"""Central paths, constants and environment handling. Every module imports from here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in CWD if present; silently no-op otherwise

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PATHFRONTIER_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.getenv("PATHFRONTIER_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.getenv("PATHFRONTIER_RESULTS_DIR", ROOT / "results"))
for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Business framing constants (documented in GUIDE.md; tune to your story).
TILES_PER_SLIDE = 10_000          # typical tissue tiles in one 40x whole-slide image
PATHOLOGIST_SEC_PER_TILE = 2.0    # rough seconds a reader spends per tile in a manual screen
TARGET_SENSITIVITY = 0.99         # clinical floor: missed tumor >> false alarm
FIXED_SPECIFICITY_POINTS = (0.90, 0.95, 0.99)


def device() -> str:
    """Best available torch device string without importing torch at module import time."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
