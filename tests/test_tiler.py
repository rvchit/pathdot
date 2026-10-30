"""Offline tests for the DZI helpers and the pure-Pillow pyramid writer."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathfrontier import dzi

ROOT = Path(__file__).resolve().parent.parent


def gradient_image(w: int = 600, h: int = 400) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([
        (xx * 255 // max(w - 1, 1)),
        (yy * 255 // max(h - 1, 1)),
        np.full_like(xx, 120),
    ], axis=-1).astype("uint8")
    return Image.fromarray(arr, "RGB")


def test_geometry_helpers():
    assert dzi.max_level_for(600, 400) == 10        # 2**10 = 1024 >= 600 > 512
    assert dzi.max_level_for(1, 1) == 0
    assert dzi.max_level_for(256, 256) == 8
    assert dzi.level_dims(600, 400, 10, 10) == (600, 400)
    assert dzi.level_dims(600, 400, 9, 10) == (300, 200)
    assert dzi.level_dims(600, 400, 0, 10) == (1, 1)
    assert dzi.level_grid(600, 400, 10, 10) == (3, 2)
    assert dzi.level_grid(600, 400, 9, 10) == (2, 1)
    assert dzi.level_grid(600, 400, 8, 10) == (1, 1)
    assert dzi.level_scale(8, 10) == 4
    with pytest.raises(ValueError):
        dzi.level_scale(11, 10)
    assert dzi.tile_path("tiles", "img", 10, 2, 1) == Path("tiles/img_files/10/2_1.jpeg")


def test_pillow_pyramid_600x400(tmp_path: Path):
    img = gradient_image(600, 400)
    dzi_file = dzi.tile_image_pillow(img, tmp_path, "synth")

    # Descriptor at the contract location and at the vips-style sibling location.
    assert dzi_file == tmp_path / "synth" / "synth.dzi"
    assert (tmp_path / "synth.dzi").is_file()
    xml = dzi_file.read_text()
    assert 'xmlns="http://schemas.microsoft.com/deepzoom/2008"' in xml
    assert 'Format="jpeg"' in xml and 'Overlap="0"' in xml and 'TileSize="256"' in xml
    assert 'Width="600"' in xml and 'Height="400"' in xml

    info = dzi.parse_dzi(dzi_file)
    assert info == {"width": 600, "height": 400, "tile_size": 256, "overlap": 0,
                    "format": "jpeg", "max_level": 10}
    assert dzi.find_dzi(tmp_path, "synth") == dzi_file
    assert dzi.list_image_ids(tmp_path) == ["synth"]

    files = tmp_path / "synth_files"
    expected = {10: 6, 9: 2, 8: 1, 7: 1, 6: 1, 5: 1, 4: 1, 3: 1, 2: 1, 1: 1, 0: 1}
    for level, n in expected.items():
        got = sorted((files / str(level)).glob("*.jpeg"))
        assert len(got) == n, f"level {level}: {len(got)} tiles, expected {n}"
    assert len(list(files.rglob("*.jpeg"))) == sum(expected.values())

    # Edge tiles are cropped, not padded; interior tiles are full size.
    with Image.open(files / "10" / "0_0.jpeg") as t:
        assert t.size == (256, 256)
    with Image.open(files / "10" / "2_1.jpeg") as t:
        assert t.size == (600 - 512, 400 - 256)
    with Image.open(files / "9" / "1_0.jpeg") as t:
        assert t.size == (300 - 256, 200)
    with Image.open(files / "0" / "0_0.jpeg") as t:
        assert t.size == (1, 1)

    # Content survives: top-left tile is dark-red-ish corner, bottom-right is bright.
    with Image.open(files / "10" / "0_0.jpeg") as t:
        tl = np.asarray(t.convert("RGB"))[:8, :8].mean(axis=(0, 1))
    with Image.open(files / "10" / "2_1.jpeg") as t:
        br = np.asarray(t.convert("RGB"))[-8:, -8:].mean(axis=(0, 1))
    assert tl[0] < 40 and br[0] > 200

    tiles = list(dzi.iter_tiles(tmp_path, "synth", 10))
    assert [(x, y) for x, y, _ in tiles] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_tile_file_pillow_path(tmp_path: Path):
    src = tmp_path / "small.png"
    gradient_image(300, 100).save(src)
    out = tmp_path / "tiles"
    dzi_file = dzi.tile_file(src, out, prefer_vips=False)
    info = dzi.parse_dzi(dzi_file)
    assert info["max_level"] == math.ceil(math.log2(300)) == 9
    assert dzi.level_grid(300, 100, 9, 9) == (2, 1)
    assert dzi.list_image_ids(out) == ["small"]
    with pytest.raises(RuntimeError):
        # whole-slide formats need libvips; the Pillow path must refuse clearly
        dzi.tile_file(tmp_path / "x.svs", out, prefer_vips=False)


def test_tile_slide_cli(tmp_path: Path):
    """Run scripts/tile_slide.py's typer app in-process on a png."""
    from typer.testing import CliRunner

    spec = importlib.util.spec_from_file_location("tile_slide", ROOT / "scripts" / "tile_slide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "pic.jpg"
    gradient_image(520, 260).save(src)
    res = CliRunner().invoke(mod.app, [str(src), "--out", str(tmp_path / "t"), "--no-vips"])
    assert res.exit_code == 0, res.output
    assert "max_level=10" in res.output
    assert (tmp_path / "t" / "pic" / "pic.dzi").is_file()
    assert len(list((tmp_path / "t" / "pic_files" / "10").glob("*.jpeg"))) == 3 * 2
