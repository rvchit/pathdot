"""Shared offline fixtures. No network, no API keys, fast.

Reusable by every engineer:
    synthetic_records        64 synthetic pcam TileRecords (balanced)
    synthetic_records_crc    90 synthetic crc TileRecords (10 per class)
    synthetic_preds          Prediction list for the pcam records from a good-but-imperfect
                             simulated probe (AUROC ~0.95, no pathologist routing)
    make_preds               factory: make_preds(records, dataset, quality=..., model=..., stage=...)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathfrontier.data import TileRecord, make_synthetic
from pathfrontier.schema import CLASSES, POSITIVE_INDEX, Prediction


def simulate_predictions(records: list[TileRecord], dataset: str = "pcam", quality: float = 0.8,
                         seed: int = 0, model: str = "probe/synthetic", stage: str = "probe",
                         latency_ms: float = 2.0, cost_per_tile: float = 0.0000005,
                         split: str = "test") -> list[Prediction]:
    """Simulated predictor: probability mass `quality` on the true class blended with noise.
    quality=1.0 -> perfect ranking; quality=0.0 -> pure noise."""
    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES[dataset])
    out = []
    for rec in records:
        noise = rng.dirichlet(np.ones(n_classes))
        onehot = np.zeros(n_classes)
        onehot[rec.label] = 1.0
        probs = quality * onehot + (1.0 - quality) * noise
        probs = probs / probs.sum()
        out.append(Prediction(
            tile_id=rec.tile_id, dataset=dataset, split=split, y_true=rec.label,
            y_prob=probs.tolist(), y_pred=int(np.argmax(probs)), model=model, stage=stage,
            latency_ms=float(latency_ms * rng.uniform(0.5, 1.5)), cost_usd=cost_per_tile,
        ))
    return out


def binary_pred(y_true: int, p_pos: float, i: int = 0, dataset: str = "pcam", **kw) -> Prediction:
    """Hand-built binary Prediction from a positive-class probability (pcam by default)."""
    n_classes = len(CLASSES[dataset])
    pos = POSITIVE_INDEX[dataset]
    probs = [(1.0 - p_pos) / (n_classes - 1)] * n_classes
    probs[pos] = p_pos
    defaults = {"tile_id": f"{dataset}:test:{i}", "dataset": dataset, "split": "test",
                "y_true": y_true, "y_prob": probs, "y_pred": int(np.argmax(probs)),
                "model": "hand", "stage": "probe"}
    defaults.update(kw)
    return Prediction(**defaults)


@pytest.fixture(scope="session")
def synthetic_records() -> list[TileRecord]:
    return make_synthetic("pcam", n=64, seed=0)


@pytest.fixture(scope="session")
def synthetic_records_crc() -> list[TileRecord]:
    return make_synthetic("crc", n=90, seed=0)


@pytest.fixture(scope="session")
def synthetic_preds(synthetic_records) -> list[Prediction]:
    return simulate_predictions(synthetic_records, "pcam", quality=0.7, seed=1)


@pytest.fixture(scope="session")
def make_preds():
    return simulate_predictions


@pytest.fixture(scope="session")
def make_binary_pred():
    return binary_pred
