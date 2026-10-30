"""THE CONTRACT. Every model (baseline VLM, probe, fine-tuned VLM, cascade) emits
`Prediction` records to a JSONL file in RESULTS_DIR; the eval harness, sweep,
frontier chart and demo all consume that format. Do not add required fields.

JSONL filename convention:  results/preds_<model_name>_<split>.jsonl
Embedding cache convention: cache/emb_<encoder>_<dataset>_<split>.npy  (+ _labels.npy, _ids.json)
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DatasetName = Literal["pcam", "crc"]
Split = Literal["train", "valid", "test"]

# Label spaces. Binary PCam: index 1 == tumor (positive class for sensitivity).
PCAM_CLASSES = ["normal", "tumor"]
CRC_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CRC_POSITIVE_CLASS = "TUM"  # for sensitivity/specificity we treat TUM-vs-rest as the clinical question
CLASSES = {"pcam": PCAM_CLASSES, "crc": CRC_CLASSES}
POSITIVE_INDEX = {"pcam": 1, "crc": CRC_CLASSES.index(CRC_POSITIVE_CLASS)}

# Cost model in USD per 1,000 tiles. Baseline/fine-tuned VLM costs are filled from the
# live price list by the vlm_client (tokens * price). Probe cost is amortised GPU time:
# e.g. ViT-L on an H100 at ~$2.5/h processes ~1,500 tiles/s -> ~$0.0005 / 1k tiles.
# These are defaults; scripts may override with measured numbers.
DEFAULT_COST_PER_1K = {
    "probe": 0.0005,
    "vlm_baseline": 2.00,     # placeholder; overwritten by measured token cost
    "vlm_finetuned": 0.60,    # placeholder; overwritten by measured token cost
    "pathologist": 0.0,       # time cost tracked separately in seconds, not USD
}


class Prediction(BaseModel):
    """One model's output for one tile."""

    tile_id: str                       # stable id: f"{dataset}:{split}:{row_index}"
    dataset: DatasetName
    split: Split
    y_true: int                        # class index into CLASSES[dataset]
    y_prob: list[float]                # probability per class; len == len(CLASSES[dataset]); sums ~1
    y_pred: int                        # argmax or thresholded decision
    model: str                         # e.g. "fireworks/qwen2.5-vl-32b", "probe/phikon-v2", "cascade"
    stage: Literal["baseline", "probe", "vlm_ft", "cascade", "pathologist"] = "probe"
    latency_ms: float = 0.0            # wall time for this tile (batch time / batch size is fine)
    cost_usd: float = 0.0              # dollars attributed to this tile
    confidence: float | None = None    # max(y_prob) unless the model defines its own
    reason: str | None = None          # VLM free-text rationale, if any
    routed_to: str | None = None       # for cascade: which stage made the final call
    extra: dict = Field(default_factory=dict)

    def model_post_init(self, context, /) -> None:
        if self.confidence is None:
            self.confidence = max(self.y_prob) if self.y_prob else None


def preds_path(model_name: str, split: str, results_dir: Path) -> Path:
    safe = model_name.replace("/", "_").replace(":", "_")
    return results_dir / f"preds_{safe}_{split}.jsonl"


def write_predictions(preds: Iterable[Prediction], path: Path, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w") as f:
        for p in preds:
            f.write(p.model_dump_json() + "\n")
            n += 1
    return n


def read_predictions(path: Path) -> Iterator[Prediction]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield Prediction.model_validate_json(line)


def load_predictions(path: Path) -> list[Prediction]:
    return list(read_predictions(path))


def embedding_paths(encoder: str, dataset: str, split: str, cache_dir: Path) -> dict[str, Path]:
    safe = encoder.replace("/", "_")
    stem = cache_dir / f"emb_{safe}_{dataset}_{split}"
    return {"X": stem.with_suffix(".npy"), "y": Path(f"{stem}_labels.npy"), "ids": Path(f"{stem}_ids.json")}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
