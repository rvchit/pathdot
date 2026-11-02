"""Train a linear probe on cached embeddings and evaluate on the test split (CONTRACTS.md section 3).

    uv run python scripts/train_probe.py --encoder phikon-v2 --dataset pcam

Reads cache/emb_<encoder>_<dataset>_<split>.npy (+ labels/ids/meta) written by scripts/embed.py,
fits `LinearProbe`, saves results/probe_<encoder>_<dataset>.joblib, writes
results/preds_probe_<encoder>_<test-split>.jsonl and prints AUROC / balanced accuracy.

Cost model: per-tile GPU cost = (1 / tiles_per_s) * GPU_USD_PER_HOUR / 3600 using the throughput
recorded in the embedding cache meta (override with --tiles-per-s). GPU_USD_PER_HOUR defaults to an
H100 at ~$2.5/h. NOTE: the throughput measured on this Mac (MPS) is far lower than an H100, so
pass --tiles-per-s with an H100 number (e.g. ~1500 for a ViT-L) to get a realistic cloud cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.probe import LinearProbe
from pathfrontier.schema import (
    CLASSES,
    POSITIVE_INDEX,
    embedding_paths,
    preds_path,
    save_json,
    write_predictions,
)

GPU_USD_PER_HOUR = 2.5  # H100 on-demand, rough public cloud price (2026)
DEFAULT_TRAIN_SPLIT = {"pcam": "valid", "crc": "train"}

app = typer.Typer(add_completion=False)


def load_cache(encoder: str, dataset: str, split: str, cache_dir: Path | None = None
               ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    cache_dir = cache_dir or config.CACHE_DIR
    paths = embedding_paths(encoder, dataset, split, cache_dir)
    meta_p = cache_dir / f"emb_{encoder.replace('/', '_')}_{dataset}_{split}.meta.json"
    missing = [str(p) for p in (*paths.values(), meta_p) if not p.exists()]
    if missing:
        raise typer.BadParameter(
            f"embedding cache for {encoder}/{dataset}/{split} is missing ({missing[0]}). Run:\n"
            f"  .venv/bin/python scripts/embed.py --encoder {encoder} --dataset {dataset} "
            f"--split {split}"
        )
    X = np.load(paths["X"]).astype(np.float32)
    y = np.load(paths["y"]).astype(np.int64)
    with open(paths["ids"]) as f:
        ids = json.load(f)
    meta = json.loads(meta_p.read_text())
    return X, y, ids, meta


def auroc(y: np.ndarray, p: np.ndarray, dataset: str) -> float | None:
    """Binary: positive-class AUROC. Multiclass: tumour-vs-rest AUROC (clinical question)."""
    pos = POSITIVE_INDEX[dataset]
    y_bin = (y == pos).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    return float(roc_auc_score(y_bin, p[:, pos]))


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    if p.shape[1] <= 2 or len(np.unique(y)) < 2:
        return None
    present = np.unique(y)
    try:
        return float(roc_auc_score(y, p[:, present] / p[:, present].sum(1, keepdims=True),
                                   multi_class="ovr", average="macro", labels=present))
    except ValueError:
        return None


@app.command()
def main(
    encoder: str = typer.Option("phikon-v2"),
    dataset: str = typer.Option("pcam", help="test dataset (and train dataset unless overridden)"),
    train_dataset: str | None = typer.Option(None, help="train on another dataset with the same "
                                             "label space, e.g. pcam -> camelyon17"),
    train_split: str | None = typer.Option(None, help="default: valid for pcam, train for crc"),
    test_split: str = typer.Option("test"),
    c: float = typer.Option(1.0, "--C", help="inverse L2 regularisation strength"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate"),
    class_weight: str | None = typer.Option("balanced", help="'balanced' or 'none'"),
    gpu_usd_per_hour: float = typer.Option(GPU_USD_PER_HOUR),
    tiles_per_s: float | None = typer.Option(None, help="override measured encoder throughput"),
    seed: int = typer.Option(0),
    tag: str = typer.Option("", help="suffix for output names, to keep several probes apart"),
    results_dir: str = typer.Option("", help="default: results/"),
) -> None:
    train_dataset = train_dataset or dataset
    if CLASSES[train_dataset] != CLASSES[dataset]:
        raise typer.BadParameter(f"{train_dataset} and {dataset} have different label spaces")
    out_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    suffix = f"_{tag}" if tag else ""
    train_split = train_split or DEFAULT_TRAIN_SPLIT.get(train_dataset, "train")
    Xtr, ytr, _, meta_tr = load_cache(encoder, train_dataset, train_split)
    Xte, yte, ids_te, meta_te = load_cache(encoder, dataset, test_split)

    cw = None if class_weight in (None, "none", "None", "") else class_weight
    probe = LinearProbe(encoder, dataset, C=c, calibrate=calibrate, class_weight=cw, seed=seed)
    probe.fit(Xtr, ytr)
    model_path = out_dir / f"probe_{encoder}_{train_dataset}{suffix}.joblib"
    probe.save(model_path)

    tps = tiles_per_s or meta_te.get("tiles_per_s") or meta_tr.get("tiles_per_s")
    cost_per_tile = (1.0 / tps) * gpu_usd_per_hour / 3600.0
    latency_ms = 1000.0 / tps
    preds = probe.predict_records(Xte, yte, ids_te, test_split, latency_ms, cost_per_tile)
    out = preds_path(probe.model_name + suffix, test_split, out_dir)
    write_predictions(preds, out)

    p = probe.predict_proba(Xte)
    y_hat = p.argmax(1)
    summary = {
        "encoder": encoder, "dataset": dataset, "train_dataset": train_dataset,
        "train_split": train_split,
        "test_split": test_split, "n_train": len(ytr), "n_test": len(yte),
        "C": c, "calibrated": probe.is_calibrated,
        "auroc": auroc(yte, p, dataset), "auroc_macro_ovr": macro_auroc(yte, p),
        "balanced_accuracy": float(balanced_accuracy_score(yte, y_hat)),
        "accuracy": float(accuracy_score(yte, y_hat)),
        "tiles_per_s": float(tps), "device": meta_te.get("device"),
        "gpu_usd_per_hour": gpu_usd_per_hour, "cost_per_tile_usd": cost_per_tile,
        "cost_per_1k_usd": cost_per_tile * 1000, "latency_ms": latency_ms,
        "model_path": str(model_path), "preds_path": str(out),
    }
    save_json(summary, out_dir / f"probe_{encoder}_{dataset}{suffix}.json")

    typer.echo(f"probe/{encoder} on {dataset}: train={train_dataset}/{train_split} (n={len(ytr)}), "
               f"test={test_split} (n={len(yte)}), calibrated={probe.is_calibrated}")
    a = summary["auroc"]
    typer.echo(f"  AUROC (positive vs rest): {a:.4f}" if a is not None else "  AUROC: n/a")
    if summary["auroc_macro_ovr"] is not None:
        typer.echo(f"  AUROC (macro OvR):        {summary['auroc_macro_ovr']:.4f}")
    typer.echo(f"  balanced accuracy:        {summary['balanced_accuracy']:.4f}")
    typer.echo(f"  accuracy:                 {summary['accuracy']:.4f}")
    typer.echo(f"  throughput {tps:.1f} tiles/s on {meta_te.get('device')} -> "
               f"{latency_ms:.2f} ms/tile, ${cost_per_tile * 1000:.5f} per 1k tiles "
               f"at ${gpu_usd_per_hour}/h")
    if meta_te.get("device") != "cuda" and tiles_per_s is None:
        typer.echo("  note: throughput was measured on this Mac; an H100 is much faster. "
                   "Pass --tiles-per-s to use a GPU figure.")
    typer.echo(f"  saved {model_path}\n  wrote {out} ({len(preds)} predictions)")


if __name__ == "__main__":
    app()
