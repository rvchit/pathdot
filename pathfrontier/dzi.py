"""Deep Zoom Image (DZI) helpers shared by the tiler scripts and the inference service.

Layout (see docs/CONTRACTS.md):

    tiles/<image_id>/<image_id>.dzi                 the XML descriptor (contract location)
    tiles/<image_id>.dzi                            a copy where `vips dzsave` / OpenSeadragon expect it
    tiles/<image_id>_files/<level>/<x>_<y>.jpeg     the pyramid; level 0 = 1x1 px, max_level = full res

DZI level numbering: ``max_level = ceil(log2(max(width, height)))`` and the image at ``level``
has size ``ceil(width / 2**(max_level - level)) x ceil(height / 2**(max_level - level))``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

DZI_NS = "http://schemas.microsoft.com/deepzoom/2008"
DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 0
DEFAULT_FORMAT = "jpeg"


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def max_level_for(width: int, height: int) -> int:
    """DZI max level: the smallest L with 2**L >= max(width, height)."""
    return math.ceil(math.log2(max(width, height, 1)))


def level_scale(level: int, max_level: int) -> int:
    """Full-resolution image pixels per pixel at `level` (= 2 ** (max_level - level))."""
    if level < 0 or level > max_level:
        raise ValueError(f"level {level} outside [0, {max_level}]")
    return 2 ** (max_level - level)


def level_dims(width: int, height: int, level: int, max_level: int) -> tuple[int, int]:
    """Pixel size of the image at `level`."""
    s = level_scale(level, max_level)
    return max(1, math.ceil(width / s)), max(1, math.ceil(height / s))


def level_grid(width: int, height: int, level: int, max_level: int,
               tile_size: int = DEFAULT_TILE_SIZE) -> tuple[int, int]:
    """(cols, rows) of tiles at `level`."""
    w, h = level_dims(width, height, level, max_level)
    return math.ceil(w / tile_size), math.ceil(h / tile_size)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
def dzi_path(tiles_dir: Path | str, image_id: str) -> Path:
    """Contract location of the descriptor: tiles/<id>/<id>.dzi."""
    return Path(tiles_dir) / image_id / f"{image_id}.dzi"


def files_dir(tiles_dir: Path | str, image_id: str) -> Path:
    return Path(tiles_dir) / f"{image_id}_files"


def tile_path(tiles_dir: Path | str, image_id: str, level: int, x: int, y: int,
              fmt: str = DEFAULT_FORMAT) -> Path:
    return files_dir(tiles_dir, image_id) / str(level) / f"{x}_{y}.{fmt}"


def find_dzi(tiles_dir: Path | str, image_id: str) -> Path | None:
    """Locate the .dzi for an image id, accepting both layouts."""
    tiles_dir = Path(tiles_dir)
    for cand in (dzi_path(tiles_dir, image_id), tiles_dir / f"{image_id}.dzi"):
        if cand.is_file():
            return cand
    return None


def list_image_ids(tiles_dir: Path | str) -> list[str]:
    """Image ids that have a descriptor under tiles_dir."""
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.is_dir():
        return []
    ids: set[str] = set()
    for p in tiles_dir.glob("*.dzi"):
        ids.add(p.stem)
    for p in tiles_dir.glob("*/*.dzi"):
        if p.parent.name == p.stem:
            ids.add(p.stem)
    return sorted(ids)


# --------------------------------------------------------------------------------------
# Descriptor read / write
# --------------------------------------------------------------------------------------
def parse_dzi(path: Path | str) -> dict:
    """Parse a .dzi XML -> {width, height, tile_size, overlap, format, max_level}."""
    root = ET.parse(str(path)).getroot()
    # Namespace-agnostic lookups (vips writes the namespace; some tools don't).
    size = None
    for el in root.iter():
        if el.tag.split("}")[-1] == "Size":
            size = el
            break
    if size is None:
        raise ValueError(f"{path}: no <Size> element")
    width = int(size.attrib["Width"])
    height = int(size.attrib["Height"])
    tile_size = int(root.attrib.get("TileSize", DEFAULT_TILE_SIZE))
    overlap = int(root.attrib.get("Overlap", DEFAULT_OVERLAP))
    fmt = root.attrib.get("Format", DEFAULT_FORMAT)
    return {
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "format": fmt,
        "max_level": max_level_for(width, height),
    }


def dzi_xml(width: int, height: int, tile_size: int = DEFAULT_TILE_SIZE,
            overlap: int = DEFAULT_OVERLAP, fmt: str = DEFAULT_FORMAT) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="{DZI_NS}" Format="{fmt}" Overlap="{overlap}" TileSize="{tile_size}">'
        f'<Size Width="{width}" Height="{height}"/></Image>\n'
    )


def write_dzi(tiles_dir: Path | str, image_id: str, width: int, height: int,
              tile_size: int = DEFAULT_TILE_SIZE, overlap: int = DEFAULT_OVERLAP,
              fmt: str = DEFAULT_FORMAT) -> Path:
    """Write the descriptor to the contract location and the vips-style sibling location."""
    tiles_dir = Path(tiles_dir)
    xml = dzi_xml(width, height, tile_size, overlap, fmt)
    primary = dzi_path(tiles_dir, image_id)
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.write_text(xml)
    (tiles_dir / f"{image_id}.dzi").write_text(xml)
    return primary


# --------------------------------------------------------------------------------------
# Iteration
# --------------------------------------------------------------------------------------
def iter_tiles(tiles_dir: Path | str, image_id: str, level: int, info: dict | None = None,
               only_existing: bool = True) -> Iterator[tuple[int, int, Path]]:
    """Yield (x, y, path) row-major for every tile at `level`."""
    tiles_dir = Path(tiles_dir)
    if info is None:
        p = find_dzi(tiles_dir, image_id)
        if p is None:
            raise FileNotFoundError(f"no .dzi for {image_id!r} under {tiles_dir}")
        info = parse_dzi(p)
    cols, rows = level_grid(info["width"], info["height"], level, info["max_level"],
                            info["tile_size"])
    for y in range(rows):
        for x in range(cols):
            p = tile_path(tiles_dir, image_id, level, x, y, info["format"])
            if only_existing and not p.is_file():
                continue
            yield x, y, p


# --------------------------------------------------------------------------------------
# Pyramid writers
# --------------------------------------------------------------------------------------
def pyvips_available() -> bool:
    try:
        import pyvips  # noqa: F401
    except (ImportError, OSError):  # OSError when the module exists but libvips does not
        return False
    return True


def tile_image_pillow(image, tiles_dir: Path | str, image_id: str,
                      tile_size: int = DEFAULT_TILE_SIZE, quality: int = 90,
                      progress=None) -> Path:
    """Pure-Pillow DZI pyramid writer.

    Starts from the full-res image at max_level and halves (LANCZOS) down to level 0,
    writing `<x>_<y>.jpeg` tiles for every level. Returns the .dzi path.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    max_level = max_level_for(width, height)
    tiles_dir = Path(tiles_dir)
    out_dir = files_dir(tiles_dir, image_id)

    cur = image
    for level in range(max_level, -1, -1):
        lw, lh = level_dims(width, height, level, max_level)
        if cur.size != (lw, lh):
            cur = cur.resize((lw, lh), Image.LANCZOS)
        level_dir = out_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        cols, rows = level_grid(width, height, level, max_level, tile_size)
        for y in range(rows):
            for x in range(cols):
                box = (x * tile_size, y * tile_size,
                       min((x + 1) * tile_size, lw), min((y + 1) * tile_size, lh))
                cur.crop(box).save(level_dir / f"{x}_{y}.jpeg", "JPEG", quality=quality)
        if progress is not None:
            progress(level, max_level, cols * rows)
    return write_dzi(tiles_dir, image_id, width, height, tile_size, 0, DEFAULT_FORMAT)


def tile_file_pyvips(src: Path | str, tiles_dir: Path | str, image_id: str,
                     tile_size: int = DEFAULT_TILE_SIZE, quality: int = 90) -> Path:
    """pyvips `dzsave` writer (handles .svs/.tiff/.ndpi via openslide/libvips)."""
    import pyvips  # may raise ImportError / OSError; caller checks pyvips_available()

    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    img = pyvips.Image.new_from_file(str(src), access="sequential")
    if img.bands == 4:
        img = img.flatten(background=[255, 255, 255])
    img = img.colourspace("srgb") if img.interpretation != "srgb" else img
    # dzsave(basepath) -> <basepath>.dzi + <basepath>_files/<level>/<x>_<y>.jpeg
    img.dzsave(str(tiles_dir / image_id), suffix=f".jpeg[Q={quality}]", tile_size=tile_size,
               overlap=0, depth="onepixel")
    # Mirror the descriptor to the contract location tiles/<id>/<id>.dzi.
    info = parse_dzi(tiles_dir / f"{image_id}.dzi")
    return write_dzi(tiles_dir, image_id, info["width"], info["height"], tile_size, 0,
                     DEFAULT_FORMAT)


def tile_file(src: Path | str, tiles_dir: Path | str, image_id: str | None = None,
              tile_size: int = DEFAULT_TILE_SIZE, prefer_vips: bool = True,
              progress=None) -> Path:
    """Tile an image file into a DZI pyramid; pyvips when available, else Pillow."""
    src = Path(src)
    image_id = image_id or src.stem
    if prefer_vips and pyvips_available():
        return tile_file_pyvips(src, tiles_dir, image_id, tile_size)
    if src.suffix.lower() in {".svs", ".ndpi", ".scn", ".mrxs", ".vms", ".bif"}:
        raise RuntimeError(
            f"{src.name}: whole-slide formats need pyvips + libvips "
            "(brew install vips && uv pip install pyvips)."
        )
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        im.load()
        return tile_image_pillow(im, tiles_dir, image_id, tile_size, progress=progress)
