"""Stitch a seeded grid of real test tiles into one PNG "slide" + ground truth, then tile it
(docs/CONTRACTS.md section 9).

    uv run python scripts/make_mosaic_slide.py --dataset pcam --grid 24        # real PCam test tiles
    uv run python scripts/make_mosaic_slide.py --synthetic --grid 8            # offline fixture

Outputs:
    data/slides/pcam_mosaic_<grid>.png        the stitched image (grid*tile_px square)
    data/slides/pcam_mosaic_<grid>.gt.json    {"grid", "tile_px", "width", "height", "cells":
                                               [{"row","col","label","tile_id"}, ...]}
    tiles/pcam_mosaic_<grid>/...              the DZI pyramid (via pathfrontier.dzi.tile_file)

Geometry note: source tiles are 96 px (PCam) while DZI tiles are 256 px, so the two grids do not
align. That is expected: the service scores DZI tiles; the frontend draws truth cells from the
gt.json using `tile_px` (cell rect in image px = [col*tile_px, row*tile_px, tile_px, tile_px]).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config, dzi
from pathfrontier.schema import CLASSES

app = typer.Typer(add_completion=False)


def _fallback_synthetic(dataset: str, n: int, seed: int) -> list:
    """Offline fixture used only if pathfrontier.data.make_synthetic is unavailable."""
    from dataclasses import dataclass

    import numpy as np
    from PIL import Image

    @dataclass
    class Rec:
        tile_id: str
        image: Image.Image
        label: int

    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES[dataset])
    px = 96 if dataset == "pcam" else 224
    out = []
    for i in range(n):
        label = int(rng.integers(n_classes))
        base = np.array([150, 60, 170] if label == 1 else [235, 170, 205], dtype=np.float32)
        arr = base + rng.normal(0, 25, size=(px, px, 3))
        out.append(Rec(f"{dataset}:synthetic:{i}", Image.fromarray(
            np.clip(arr, 0, 255).astype("uint8"), "RGB"), label))
    return out


def load_records(dataset: str, n: int, seed: int, synthetic: bool) -> list:
    if synthetic:
        try:
            from pathfrontier.data import make_synthetic
        except ImportError:
            typer.echo("pathfrontier.data not importable; using built-in synthetic tiles")
            return _fallback_synthetic(dataset, n, seed)
        return make_synthetic(dataset=dataset, n=n, seed=seed)
    try:
        from pathfrontier.data import load_split
    except ImportError as e:
        raise typer.BadParameter(
            f"pathfrontier.data is not available ({e}); pass --synthetic for an offline mosaic"
        ) from e
    return load_split(dataset, "test", limit=n, seed=seed)


def build_mosaic(records: list, grid: int, cell_px: int | None = None):
    """Stitch records into a grid. ``cell_px`` resizes every tile to that size (default: native
    tile size). Using 256 makes each full-resolution DZI tile coincide with exactly one cell, so
    the heatmap and the ground-truth overlay line up 1:1 in the viewer."""
    from PIL import Image

    tile_px = cell_px or records[0].image.width
    canvas = Image.new("RGB", (grid * tile_px, grid * tile_px), (255, 255, 255))
    cells = []
    for i in range(grid * grid):
        rec = records[i % len(records)]
        row, col = divmod(i, grid)
        im = rec.image.convert("RGB")
        if im.size != (tile_px, tile_px):
            im = im.resize((tile_px, tile_px))
        canvas.paste(im, (col * tile_px, row * tile_px))
        cells.append({"row": row, "col": col, "label": int(rec.label), "tile_id": rec.tile_id})
    return canvas, tile_px, cells


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="pcam or crc")] = "pcam",
    grid: Annotated[int, typer.Option(help="grid side; grid*grid tiles are stitched")] = 24,
    seed: Annotated[int, typer.Option(help="seed for the test-split shuffle")] = 0,
    out_dir: Annotated[Path | None, typer.Option(help="png + gt.json dir (default data/slides)")] = None,
    tiles_dir: Annotated[Path, typer.Option(help="DZI output root")] = Path("tiles"),
    synthetic: Annotated[bool, typer.Option("--synthetic", help="use make_synthetic (offline)")] = False,
    image_id: Annotated[str | None, typer.Option(help="override id (default <dataset>_mosaic_<grid>)")] = None,
    no_tile: Annotated[bool, typer.Option("--no-tile", help="skip the DZI step")] = False,
    cell_px: Annotated[int, typer.Option(help="resize each tile to this many px (0 = native). "
                                             "256 aligns cells with full-res DZI tiles")] = 256,
):
    out_dir = out_dir or (config.DATA_DIR / "slides")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = image_id or f"{dataset}_mosaic_{grid}"
    n = grid * grid

    typer.echo(f"loading {n} {'synthetic' if synthetic else 'real test'} {dataset} tiles (seed={seed})")
    records = load_records(dataset, n, seed, synthetic)
    if not records:
        raise typer.BadParameter("no records loaded")
    if len(records) < n:
        typer.echo(f"only {len(records)} tiles available; cells will repeat")

    canvas, tile_px, cells = build_mosaic(records, grid, cell_px or None)
    png = out_dir / f"{image_id}.png"
    canvas.save(png, "PNG")
    gt = {
        "image_id": image_id,
        "dataset": dataset,
        "classes": CLASSES[dataset],
        "grid": grid,
        "tile_px": tile_px,
        "width": canvas.width,
        "height": canvas.height,
        "seed": seed,
        "synthetic": synthetic,
        "cells": cells,
    }
    gt_path = out_dir / f"{image_id}.gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=1)
    pos = sum(1 for c in cells if c["label"] == 1) if dataset == "pcam" else None
    typer.echo(f"wrote {png} ({canvas.width}x{canvas.height}) and {gt_path}"
               + (f"; {pos}/{len(cells)} tumor cells" if pos is not None else ""))

    if no_tile:
        return
    dzi_file = dzi.tile_image_pillow(canvas, tiles_dir, image_id)
    info = dzi.parse_dzi(dzi_file)
    n_files = sum(1 for _ in dzi.files_dir(tiles_dir, image_id).rglob("*.jpeg"))
    typer.echo(f"tiled -> {dzi_file} (max_level={info['max_level']}, {n_files} tiles)")


if __name__ == "__main__":
    app()
