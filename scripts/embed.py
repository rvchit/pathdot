"""Embed a dataset split with an encoder and write the embedding cache (CONTRACTS.md section 2).

    uv run python scripts/embed.py --encoder phikon-v2 --dataset pcam --split test --limit 4096

Writes `schema.embedding_paths(...)` (X.npy float32, _labels.npy int64, _ids.json) and
`cache/emb_<encoder>_<dataset>_<split>.meta.json` with tiles_per_s / device / n / dim / seconds.
Existing caches are reused unless --force.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.encoders import ENCODERS, Encoder, get_encoder
from pathfrontier.schema import embedding_paths, save_json

app = typer.Typer(add_completion=False)


def meta_path(encoder: str, dataset: str, split: str, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or config.CACHE_DIR
    return cache_dir / f"emb_{encoder.replace('/', '_')}_{dataset}_{split}.meta.json"


def cache_exists(encoder: str, dataset: str, split: str, cache_dir: Path | None = None) -> bool:
    cache_dir = cache_dir or config.CACHE_DIR
    paths = embedding_paths(encoder, dataset, split, cache_dir)
    return all(p.exists() for p in paths.values()) and meta_path(encoder, dataset, split,
                                                                  cache_dir).exists()


def embed_records(enc: Encoder, records: Iterable, batch_size: int = 64
                  ) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Embed TileRecord-like objects (.tile_id/.image/.label). Returns X, y, ids, wall seconds."""
    records = list(records)
    ids = [r.tile_id for r in records]
    y = np.asarray([int(r.label) for r in records], dtype=np.int64)
    t0 = time.perf_counter()
    X = enc.embed([r.image for r in records], batch_size=batch_size)
    seconds = time.perf_counter() - t0
    return X, y, ids, seconds


def write_cache(encoder: str, dataset: str, split: str, X: np.ndarray, y: np.ndarray,
                ids: list[str], seconds: float, device: str, cache_dir: Path | None = None,
                tiles_per_s: float | None = None) -> dict:
    cache_dir = cache_dir or config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = embedding_paths(encoder, dataset, split, cache_dir)
    np.save(paths["X"], X.astype(np.float32, copy=False))
    np.save(paths["y"], y.astype(np.int64, copy=False))
    with open(paths["ids"], "w") as f:
        json.dump(list(ids), f)
    meta = {
        "tiles_per_s": float(tiles_per_s if tiles_per_s is not None
                             else len(ids) / max(seconds, 1e-9)),
        "device": device, "n": len(ids), "dim": int(X.shape[1]), "seconds": float(seconds),
        "encoder": encoder, "dataset": dataset, "split": split,
    }
    save_json(meta, meta_path(encoder, dataset, split, cache_dir))
    return meta


def load_records(dataset: str, split: str, limit: int | None, seed: int, balanced: bool,
                 synthetic: int | None):
    from pathfrontier import data

    if synthetic:
        return data.make_synthetic(dataset, n=synthetic, seed=seed)
    return data.load_split(dataset, split, limit=limit, seed=seed, balanced=balanced)


@app.command()
def main(
    encoder: str = typer.Option("phikon-v2", help=f"one of {list(ENCODERS)}"),
    dataset: str = typer.Option("pcam", help="pcam | crc"),
    split: str = typer.Option("test", help="train | valid | test"),
    limit: int | None = typer.Option(None, help="first N tiles after a seeded shuffle"),
    batch_size: int = typer.Option(64),
    seed: int = typer.Option(0),
    balanced: bool = typer.Option(False, help="equal number of tiles per class"),
    force: bool = typer.Option(False, help="recompute even if the cache exists"),
    device: str | None = typer.Option(None, help="cuda | mps | cpu (default: auto)"),
    synthetic: int | None = typer.Option(None, help="use N synthetic tiles instead of the real split"),
) -> None:
    if encoder not in ENCODERS:
        raise typer.BadParameter(f"unknown encoder {encoder}; choose from {list(ENCODERS)}")
    if not force and cache_exists(encoder, dataset, split):
        meta = json.loads(meta_path(encoder, dataset, split).read_text())
        typer.echo(f"cache exists for {encoder}/{dataset}/{split} (n={meta['n']}, dim={meta['dim']}, "
                   f"{meta['tiles_per_s']:.1f} tiles/s on {meta['device']}); use --force to redo")
        return

    typer.echo(f"loading {dataset}/{split} (limit={limit}, balanced={balanced}, seed={seed})")
    records = load_records(dataset, split, limit, seed, balanced, synthetic)
    typer.echo(f"loaded {len(records)} tiles; loading encoder {encoder}")
    enc = get_encoder(encoder, device=device)
    # warm-up so the first batch's kernel compilation does not pollute the throughput number
    enc.embed([records[0].image] * min(8, len(records)), batch_size=batch_size)
    X, y, ids, seconds = embed_records(enc, records, batch_size=batch_size)
    meta = write_cache(encoder, dataset, split, X, y, ids, seconds, enc.device)
    paths = embedding_paths(encoder, dataset, split, config.CACHE_DIR)
    typer.echo(f"wrote {paths['X']} {X.shape} in {seconds:.1f}s "
               f"({meta['tiles_per_s']:.1f} tiles/s on {enc.device})")


if __name__ == "__main__":
    app()
