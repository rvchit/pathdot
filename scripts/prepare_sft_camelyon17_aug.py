"""Round-2 SFT set for Camelyon17: tumor examples from two slides instead of one, plus flips/rotations.

Reads the WILDS webdataset shards directly (data/camelyon17_wilds_test/*.tar) so it can draw beyond
the 10k pool. Only slides 20, 21, 27, 28, 29 are eligible; the eval slides 22-26 are never read.

    dev:   --dev-per-class tumor + normal tiles from slide 29 (cutoff selection), written first
    train: --n-train tiles, half tumor (split evenly between slides 28 and 29, plus the few on
           20/21/27) and half normal (all five slides), disjoint from dev; each image gets one
           random dihedral transform (rotation by 0/90/180/270, optionally mirrored)

    .venv/bin/python scripts/prepare_sft_camelyon17_aug.py
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import typer
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.finetune import estimate_training_cost, validate_sft_jsonl, write_sft_jsonl
from pathfrontier.vlm_client import V2_IMAGE_PX

app = typer.Typer(add_completion=False)
ELIGIBLE = {20, 21, 27, 28, 29}
EVAL_SLIDES = {22, 23, 24, 25, 26}
DEV_SLIDE = 29
BASE_MODEL = "accounts/fireworks/models/qwen3p5-9b"
DIHEDRAL = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
            Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]


@dataclass
class Tile:
    tile_id: str
    image: Image.Image
    label: int


def load_images(shard_dir: Path, wanted: set[tuple[int, str]]) -> dict[tuple[int, str], Image.Image]:
    out = {}
    for shard in sorted({s for s, _ in wanted}):
        with tarfile.open(shard_dir / f"{shard}.tar") as tar:
            for m in tar:
                key = (shard, m.name[:-4])
                if m.name.endswith(".png") and key in wanted:
                    out[key] = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
    return out


@app.command()
def main(
    n_train: int = typer.Option(7600),
    dev_per_class: int = typer.Option(500),
    epochs: int = typer.Option(2, help="Only used for the cost estimate"),
    seed: int = typer.Option(1),
) -> None:
    index = json.loads((config.CACHE_DIR / "camelyon17_index.json").read_text())
    rows = [(s, k, slide, y) for s, k, _h, slide, y, _d in index if slide in ELIGIBLE]
    assert not {r[2] for r in rows} & EVAL_SLIDES
    rng = random.Random(seed)
    rng.shuffle(rows)

    dev = []
    for label in (1, 0):
        dev += [r for r in rows if r[2] == DEV_SLIDE and r[3] == label][:dev_per_class]
    dev_keys = {(s, k) for s, k, _, _ in dev}
    rest = [r for r in rows if (r[0], r[1]) not in dev_keys]

    half = n_train // 2
    rare_tumor = [r for r in rest if r[3] == 1 and r[2] not in (28, 29)]
    per_big = (half - len(rare_tumor)) // 2
    tumor = rare_tumor + [r for r in rest if r[3] == 1 and r[2] == 28][:per_big] \
        + [r for r in rest if r[3] == 1 and r[2] == 29][:half - len(rare_tumor) - per_big]
    normal_by_slide = {s: [r for r in rest if r[3] == 0 and r[2] == s] for s in sorted(ELIGIBLE)}
    normal, i = [], 0
    while len(normal) < half:  # round-robin over slides for stain/tissue variety
        for s in sorted(ELIGIBLE):
            if i < len(normal_by_slide[s]) and len(normal) < half:
                normal.append(normal_by_slide[s][i])
        i += 1
    train = tumor + normal
    rng.shuffle(train)

    images = load_images(config.DATA_DIR / "camelyon17_wilds_test",
                         {(s, k) for s, k, _, _ in train + dev})
    tid = lambda s, k: f"c17wilds:{s}:{k}"

    dev_manifest = config.DATA_DIR / "camelyon17" / "dev2_manifest.csv"
    with open(dev_manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "shard", "key", "slide", "label"])
        for s, k, slide, y in dev:
            w.writerow([tid(s, k), s, k, slide, y])
    typer.echo(f"dev: {len(dev)} tiles from slide {DEV_SLIDE} -> {dev_manifest}")

    def augmented(r):
        s, k, _, y = r
        op = DIHEDRAL[rng.randrange(len(DIHEDRAL))]
        img = images[(s, k)] if op is None else images[(s, k)].transpose(op)
        return Tile(tid(s, k), img, y)

    path = config.DATA_DIR / "sft_camelyon17_aug_train.jsonl"
    n = write_sft_jsonl([augmented(r) for r in train], "camelyon17", path, image_size=V2_IMAGE_PX,
                        seed=seed, prompt_version="v2")
    by = Counter((r[2], "tumor" if r[3] else "normal") for r in train)
    typer.echo(f"train: {n} examples -> {path} ({path.stat().st_size / 1e6:.0f} MB)")
    typer.echo("  per slide: " + ", ".join(f"{s}/{c}={v}" for (s, c), v in sorted(by.items())))
    rep = validate_sft_jsonl(path)
    typer.echo(rep.summary())
    typer.echo(f"  est. cost for {epochs} epochs: "
               f"${estimate_training_cost(rep.total_tokens, epochs, BASE_MODEL):.2f}")
    if not rep.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
