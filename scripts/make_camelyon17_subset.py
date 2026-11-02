"""Build slide-disjoint Camelyon17-WILDS subsets from the webdataset shards.

Source: huggingface.co/datasets/djghosh/wds_wilds-camelyon17_test (the WILDS test split, which is
all of hospital 2, slides 20-29). Each sample is a 96x96 PNG + .cls label + .npy metadata
[hospital, slide, y, from_source_domain]. Download the 9 shards to data/camelyon17_wilds_test/.

    test  = balanced eval set from EVAL_SLIDES (frozen; every baseline / fine-tune is scored on it)
    train = balanced fine-tuning pool from the remaining slides (never overlaps test)

Writes data/camelyon17/{test,train}.parquet (data.py schema) + {test,train}_manifest.csv with the
row -> (shard, key, slide) mapping so results can be broken down per slide.

    .venv/bin/python scripts/make_camelyon17_subset.py --n-test 6000 --n-train 10000
"""
from __future__ import annotations

import csv
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.data import ROW_GROUP_SIZE

# Slides 22-26 hold 8.7k tumor / 19.7k normal patches; 20, 21, 27-29 (33.8k / 22.9k) are left for
# fine-tuning. Slide 28 alone has 27k tumor patches, so it goes to the larger training side.
EVAL_SLIDES = (22, 23, 24, 25, 26)

app = typer.Typer(add_completion=False)


def index_shards(shard_dir: Path) -> list[tuple[int, str, int, int]]:
    """(shard, key, slide, label) for every sample, read from the .npy metadata."""
    rows = []
    for shard in sorted(shard_dir.glob("*.tar"), key=lambda p: int(p.stem)):
        with tarfile.open(shard) as tar:
            for m in tar:
                if m.name.endswith(".npy"):
                    _hospital, slide, y, _ = np.load(io.BytesIO(tar.extractfile(m).read()))
                    rows.append((int(shard.stem), m.name[:-4], int(slide), int(y)))
    return rows


def balanced_sample(rows: list, n: int, rng: np.random.Generator) -> list:
    out = []
    for label in (0, 1):
        pool = [r for r in rows if r[3] == label]
        if len(pool) < n // 2:
            raise SystemExit(f"only {len(pool)} rows with label {label}, need {n // 2}")
        out += [pool[i] for i in rng.choice(len(pool), n // 2, replace=False)]
    return [out[i] for i in rng.permutation(len(out))]


def write_split(selected: list, shard_dir: Path, out_dir: Path, split: str) -> None:
    wanted = {(s, k) for s, k, _, _ in selected}
    images: dict[tuple[int, str], bytes] = {}
    for shard in sorted({s for s, _ in wanted}):
        with tarfile.open(shard_dir / f"{shard}.tar") as tar:
            for m in tar:
                key = (shard, m.name[:-4])
                if m.name.endswith(".png") and key in wanted:
                    img = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    images[key] = buf.getvalue()
    table = pa.table({
        "image": [{"bytes": images[(s, k)], "path": f"{s}/{k}.png"} for s, k, _, _ in selected],
        "label": pa.array([bool(y) for *_, y in selected]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / f"{split}.parquet", row_group_size=ROW_GROUP_SIZE,
                   compression="zstd")
    with open(out_dir / f"{split}_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "shard", "key", "hospital", "slide", "label"])
        for row, (s, k, slide, y) in enumerate(selected):
            w.writerow([f"camelyon17:{split}:{row}", s, k, 2, slide, y])
    by_slide = {}
    for *_, slide, y in selected:
        by_slide.setdefault(slide, [0, 0])[y] += 1
    typer.echo(f"{split}: {len(selected)} tiles -> {out_dir / f'{split}.parquet'}")
    for slide in sorted(by_slide):
        typer.echo(f"  slide {slide}: normal={by_slide[slide][0]} tumor={by_slide[slide][1]}")


@app.command()
def main(
    shard_dir: str = typer.Option("", help="default: data/camelyon17_wilds_test"),
    out_dir: str = typer.Option("", help="default: data/camelyon17"),
    n_test: int = typer.Option(6000, help="Balanced eval tiles from EVAL_SLIDES"),
    n_train: int = typer.Option(10000, help="Balanced fine-tuning tiles from the other slides"),
    seed: int = typer.Option(0),
) -> None:
    shard_dir = Path(shard_dir) if shard_dir else config.DATA_DIR / "camelyon17_wilds_test"
    out_dir = Path(out_dir) if out_dir else config.DATA_DIR / "camelyon17"
    rows = index_shards(shard_dir)
    typer.echo(f"indexed {len(rows)} samples from {shard_dir}")
    rng = np.random.default_rng(seed)
    test = balanced_sample([r for r in rows if r[2] in EVAL_SLIDES], n_test, rng)
    train = balanced_sample([r for r in rows if r[2] not in EVAL_SLIDES], n_train, rng)
    write_split(test, shard_dir, out_dir, "test")
    write_split(train, shard_dir, out_dir, "train")


if __name__ == "__main__":
    app()
