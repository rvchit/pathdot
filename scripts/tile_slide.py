"""DZI pyramid writer (docs/CONTRACTS.md section 9).

    uv run python scripts/tile_slide.py data/slides/CMU-1-Small-Region.svs --out tiles/
    uv run python scripts/tile_slide.py data/slides/pcam_mosaic_24.png --out tiles/ --image-id pcam_mosaic_24
    uv run python scripts/tile_slide.py --download-sample          # fetch OpenSlide's CMU-1-Small-Region.svs

pyvips (`dzsave`, handles .svs/.tiff/.ndpi) is used when importable; otherwise a pure-Pillow
pyramid (png/jpg/tif). Output: tiles/<id>/<id>.dzi (+ tiles/<id>.dzi) and
tiles/<id>_files/<level>/<x>_<y>.jpeg with DZI level numbering (0 = 1x1 px, max = full res).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config, dzi

SAMPLE_URL = ("https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"
              "CMU-1-Small-Region.svs")
SAMPLE_NAME = "CMU-1-Small-Region.svs"
VIPS_HINT = ("pyvips/libvips not available: `brew install vips && uv pip install pyvips` "
             "to tile .svs files (Pillow handles png/jpg/tif only).")

app = typer.Typer(add_completion=False)


def download_sample(dest_dir: Path) -> Path:
    import httpx

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / SAMPLE_NAME
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        typer.echo(f"sample already present: {dest}")
        return dest
    typer.echo(f"downloading {SAMPLE_URL} -> {dest}")
    with httpx.stream("GET", SAMPLE_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.writelines(r.iter_bytes())
    typer.echo(f"downloaded {dest.stat().st_size / 1e6:.1f} MB")
    return dest


@app.command()
def main(
    image: Annotated[Path | None, typer.Argument(help="slide/image to tile (.svs/.tiff/.png/.jpg)")] = None,
    out: Annotated[Path, typer.Option(help="tiles root directory")] = Path("tiles"),
    image_id: Annotated[str | None, typer.Option(help="id to use (default: file stem)")] = None,
    tile_size: Annotated[int, typer.Option(help="DZI tile size in px")] = 256,
    no_vips: Annotated[bool, typer.Option("--no-vips", help="force the pure-Pillow writer")] = False,
    download_sample_: Annotated[bool, typer.Option(
        "--download-sample",
        help=f"fetch OpenSlide's {SAMPLE_NAME} into data/slides/ and tile it if pyvips is available",
    )] = False,
):
    """Make a DZI pyramid (tile 256, overlap 0, jpeg) for OpenSeadragon / the inference service."""
    if download_sample_:
        image = download_sample(config.DATA_DIR / "slides")
        if not dzi.pyvips_available():
            typer.echo(VIPS_HINT)
            typer.echo(f"sample saved at {image}; re-run `tile_slide.py {image}` once vips is installed.")
            raise typer.Exit(code=0)
    if image is None:
        typer.echo("give an image path or --download-sample", err=True)
        raise typer.Exit(code=2)
    if not image.is_file():
        typer.echo(f"not a file: {image}", err=True)
        raise typer.Exit(code=2)

    image_id = image_id or image.stem
    use_vips = dzi.pyvips_available() and not no_vips
    typer.echo(f"tiling {image} -> {out}/{image_id} via {'pyvips' if use_vips else 'Pillow'}")

    def progress(level: int, max_level: int, n: int):
        typer.echo(f"  level {level:>2}/{max_level}: {n} tiles")

    try:
        dzi_file = dzi.tile_file(image, out, image_id, tile_size=tile_size, prefer_vips=use_vips,
                                 progress=progress)
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        typer.echo(VIPS_HINT, err=True)
        raise typer.Exit(code=1) from e
    info = dzi.parse_dzi(dzi_file)
    n_files = sum(1 for _ in dzi.files_dir(out, image_id).rglob("*.jpeg"))
    typer.echo(f"wrote {dzi_file} ({info['width']}x{info['height']}, max_level={info['max_level']}, "
               f"{n_files} tiles)")


if __name__ == "__main__":
    app()
