"""Datasets: PCam (binary) and NCT-CRC-HE (9 classes) from Hugging Face parquet shards,
plus an offline synthetic fixture (CONTRACTS.md section 1).

On-disk layout (DATA_DIR = config.DATA_DIR):
    data/pcam/{train,valid,test}.parquet
    data/crc/{train,test}.parquet          # crc "valid" is a seeded 10% holdout carved out of train.parquet
    data/camelyon17/{train,test}.parquet   # slide-disjoint WILDS hospital-2 subsets (make_camelyon17_subset.py)

Each parquet has the HF schema: columns ``image`` (struct {bytes, path}) and ``label``
(bool for pcam, int class index for crc). Shards are concatenated into one file per split
with small row groups (ROW_GROUP_SIZE) so that random access by `limit` is cheap.

Selection semantics:
    * ``limit``    -> the first N rows of a seeded permutation of the split (so limit=500 is a
                      prefix of limit=4096 for the same seed).
    * ``balanced`` -> equal number of rows per class (limit // n_classes each; or the size of the
                      smallest class when limit is None), again following the seeded permutation.
    * ``load_split`` returns records in that shuffled selection order.
    * ``iter_split`` yields the SAME set but in file order (it streams row group by row group).
    * tile_id == f"{dataset}:{split}:{row}" where ``row`` is the physical row index in the split's
      parquet file (for crc train/valid: the row in data/crc/train.parquet).
"""
from __future__ import annotations

import io
import logging
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFilter

from pathfrontier.config import DATA_DIR, HF_TOKEN
from pathfrontier.schema import CLASSES, DatasetName, Split

log = logging.getLogger(__name__)

# dataset -> (HF repo id, {our split: HF split prefix in data/})
HF_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "pcam": ("1aurent/PatchCamelyon", {"train": "train", "valid": "valid", "test": "test"}),
    "crc": ("1aurent/NCT-CRC-HE", {"train": "NCT_CRC_HE_100K", "test": "CRC_VAL_HE_7K"}),
}
# crc has no official validation split: "valid" is a 10% holdout of train with this fixed seed
# (independent of the `seed` argument so the holdout never leaks into "train").
CRC_HOLDOUT_SEED = 20251029
CRC_HOLDOUT_FRAC = 0.10
ROW_GROUP_SIZE = 1024
TILE_SIZE = {"pcam": 96, "crc": 224, "camelyon17": 96}


@dataclass
class TileRecord:
    tile_id: str  # f"{dataset}:{split}:{index}"
    image: Image.Image  # RGB
    label: int  # index into schema.CLASSES[dataset]


# --------------------------------------------------------------------------------------
# Paths / download
# --------------------------------------------------------------------------------------
def split_path(dataset: str, split: str, data_dir: Path | None = None) -> Path:
    """Parquet file backing a split (crc valid/train both live in crc/train.parquet)."""
    data_dir = Path(data_dir or DATA_DIR)
    file_split = "train" if (dataset == "crc" and split == "valid") else split
    return data_dir / dataset / f"{file_split}.parquet"


def _check_args(dataset: str, split: str) -> None:
    if dataset not in CLASSES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(CLASSES)}")
    if split not in ("train", "valid", "test"):
        raise ValueError(f"unknown split {split!r}; expected train/valid/test")


def _concat_shards(shards: list[Path], out: Path) -> int:
    """Concatenate parquet shards into one file with small row groups. Returns row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    n = 0
    try:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=ROW_GROUP_SIZE, columns=["image", "label"]):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                n += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return n


def download(dataset: DatasetName, splits: Iterable[Split] = ("valid", "test"),
             force: bool = False, data_dir: Path | None = None) -> None:
    """Fetch only the HF parquet shards needed for ``splits`` and store
    data/<dataset>/<split>.parquet. Existing files are kept unless ``force``."""
    from huggingface_hub import snapshot_download

    data_dir = Path(data_dir or DATA_DIR)
    if dataset not in HF_SOURCES:
        raise RuntimeError(f"{dataset} has no HF parquet source; build it with "
                           "scripts/make_camelyon17_subset.py")
    repo, split_map = HF_SOURCES[dataset]
    for split in splits:
        _check_args(dataset, split)
        out = split_path(dataset, split, data_dir)
        if out.exists() and not force:
            log.info("%s exists, skipping", out)
            continue
        file_split = "train" if (dataset == "crc" and split == "valid") else split
        hf_split = split_map[file_split]
        shards_dir = data_dir / dataset / f"_shards_{file_split}"
        log.info("downloading %s split %s (%s) -> %s", repo, hf_split, file_split, out)
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=[f"data/{hf_split}-*.parquet"],
            local_dir=shards_dir,
            token=HF_TOKEN or None,
        )
        shards = sorted((shards_dir / "data").glob(f"{hf_split}-*.parquet"))
        if not shards:
            raise RuntimeError(f"no shards downloaded for {repo} {hf_split}")
        n = _concat_shards(shards, out)
        shutil.rmtree(shards_dir, ignore_errors=True)
        log.info("wrote %s (%d rows)", out, n)


def ensure_split(dataset: DatasetName, split: Split, data_dir: Path | None = None) -> Path:
    path = split_path(dataset, split, data_dir)
    if not path.exists():
        download(dataset, splits=(split,), data_dir=data_dir)
    return path


# --------------------------------------------------------------------------------------
# Row selection
# --------------------------------------------------------------------------------------
def _read_labels(path: Path) -> np.ndarray:
    col = pq.read_table(path, columns=["label"]).column("label")
    return np.asarray(col.to_numpy(zero_copy_only=False)).astype(np.int64)


def _split_rows(dataset: str, split: str, n_total: int) -> np.ndarray:
    """Physical rows of the parquet file that belong to ``split``."""
    if dataset == "crc" and split in ("train", "valid"):
        perm = np.random.default_rng(CRC_HOLDOUT_SEED).permutation(n_total)
        k = round(n_total * CRC_HOLDOUT_FRAC)
        rows = perm[:k] if split == "valid" else perm[k:]
        return np.sort(rows)
    return np.arange(n_total)


def select_rows(labels: np.ndarray, rows: np.ndarray, limit: int | None, seed: int,
                balanced: bool, n_classes: int) -> np.ndarray:
    """Return physical row indices in selection order (seeded shuffle, then prefix / balance)."""
    if limit is None and not balanced:
        return rows
    rng = np.random.default_rng(seed)
    order = rows[rng.permutation(len(rows))]
    if not balanced:
        return order[:limit] if limit is not None else order
    lab = labels[order]
    counts = np.bincount(lab, minlength=n_classes)
    present = counts > 0
    per_class = int(counts[present].min()) if limit is None else max(limit // int(present.sum()), 1)
    keep = np.zeros(len(order), dtype=bool)
    for c in np.flatnonzero(present):
        idx = np.flatnonzero(lab == c)[:per_class]
        keep[idx] = True
    return order[keep]


def _decode(img: dict | bytes) -> Image.Image:
    data = img["bytes"] if isinstance(img, dict) else img
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def _iter_rows(path: Path, rows: np.ndarray) -> Iterator[tuple[int, Image.Image, int]]:
    """Yield (physical_row, image, label) for the given rows, reading in file order."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    pf = pq.ParquetFile(path)
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    group_of = np.searchsorted(starts, rows, side="right") - 1
    for g in np.unique(group_of):
        sel = rows[group_of == g]
        table = pf.read_row_group(int(g), columns=["image", "label"])
        sub = table.take(pa.array(sel - starts[g]))
        images = sub.column("image").to_pylist()
        labels = sub.column("label").to_pylist()
        for r, img, lab in zip(sel.tolist(), images, labels):
            yield r, _decode(img), int(lab)


def _plan(dataset: str, split: str, limit: int | None, seed: int, balanced: bool,
          data_dir: Path | None) -> tuple[Path, np.ndarray]:
    _check_args(dataset, split)
    path = ensure_split(dataset, split, data_dir)
    labels = _read_labels(path)
    rows = _split_rows(dataset, split, len(labels))
    selected = select_rows(labels, rows, limit, seed, balanced, len(CLASSES[dataset]))
    return path, selected


def iter_split(dataset: DatasetName, split: Split, limit: int | None = None, seed: int = 0,
               balanced: bool = False, data_dir: Path | None = None) -> Iterator[TileRecord]:
    """Streaming variant of load_split: same selected set, yielded in file order."""
    path, selected = _plan(dataset, split, limit, seed, balanced, data_dir)
    for row, image, label in _iter_rows(path, selected):
        yield TileRecord(tile_id=f"{dataset}:{split}:{row}", image=image, label=label)


def load_split(dataset: DatasetName, split: Split, limit: int | None = None, seed: int = 0,
               balanced: bool = False, data_dir: Path | None = None) -> list[TileRecord]:
    """Load tiles as a list, in seeded-shuffle selection order (see module docstring)."""
    path, selected = _plan(dataset, split, limit, seed, balanced, data_dir)
    by_row = {row: (image, label) for row, image, label in _iter_rows(path, selected)}
    return [
        TileRecord(tile_id=f"{dataset}:{split}:{int(r)}", image=by_row[int(r)][0],
                   label=by_row[int(r)][1])
        for r in selected
    ]


def records_to_parquet(records: Iterable[TileRecord], path: Path, fmt: str = "PNG") -> int:
    """Write records in the HF parquet schema (image struct + label). Handy for tiny offline
    datasets in tests/demos. Returns the number of rows written."""
    images, labels = [], []
    for rec in records:
        buf = io.BytesIO()
        rec.image.save(buf, format=fmt)
        images.append({"bytes": buf.getvalue(), "path": None})
        labels.append(rec.label)
    table = pa.table({
        "image": pa.array(images, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "label": pa.array(labels, type=pa.int64()),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    return table.num_rows


# --------------------------------------------------------------------------------------
# Synthetic fixture
# --------------------------------------------------------------------------------------
_PURPLE_DARK = (70, 20, 110)
_PURPLE = (110, 60, 150)
_PURPLE_LIGHT = (160, 110, 175)
_BLUE_DARK = (40, 30, 120)


def _blobs(draw: ImageDraw.ImageDraw, rng: np.random.Generator, n: int, r_lo: float, r_hi: float,
           color: tuple[int, int, int], size: int, center: tuple[float, float] | None = None,
           ring_r: float = 0.0, elongate: float = 1.0) -> None:
    for i in range(n):
        if center is not None and ring_r > 0:
            ang = 2 * np.pi * i / max(n, 1) + rng.normal(0, 0.1)
            cx = center[0] + ring_r * np.cos(ang)
            cy = center[1] + ring_r * np.sin(ang)
        else:
            cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(r_lo, r_hi)
        rx, ry = r * elongate, r
        shade = tuple(int(np.clip(c + rng.normal(0, 12), 0, 255)) for c in color)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=shade)


def _lines(draw: ImageDraw.ImageDraw, rng: np.random.Generator, n: int,
           color: tuple[int, int, int], size: int, width: int = 1, horizontal: bool = False,
           wavy: bool = False) -> None:
    for i in range(n):
        if horizontal:
            y = (i + rng.uniform(0, 1)) * size / n
            pts = [(0, y), (size, y)]
        elif wavy:
            x0 = rng.uniform(-size, size)
            pts = [(x0 + t * size / 8 + 4 * np.sin(t), t * size / 8) for t in range(9)]
        else:
            pts = [tuple(rng.uniform(0, size, 2)) for _ in range(rng.integers(3, 6))]
        draw.line(pts, fill=color, width=width)


def _draw_tile(rng: np.random.Generator, dataset: str, label: int, size: int) -> Image.Image:
    cls = CLASSES[dataset][label]
    if dataset in ("pcam", "camelyon17"):
        cls = {"normal": "NORM_PCAM", "tumor": "TUM"}[cls]

    bg = {
        "NORM_PCAM": (238, 205, 218), "TUM": (200, 160, 200), "ADI": (246, 240, 246),
        "BACK": (250, 250, 250), "DEB": (230, 190, 200), "LYM": (215, 190, 225),
        "MUC": (205, 215, 232), "MUS": (235, 170, 190), "NORM": (240, 200, 215),
        "STR": (235, 180, 200),
    }[cls]
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    s = size / 96.0  # scale factor so textures look similar at any size

    if cls == "NORM_PCAM":  # sparse pink: a few pale nuclei, faint fibres
        _lines(d, rng, int(rng.integers(2, 5)), (225, 185, 205), size, width=max(1, int(2 * s)))
        _blobs(d, rng, int(rng.integers(3, 9)), 2 * s, 4 * s, _PURPLE_LIGHT, size)
    elif cls == "TUM":  # dense dark purple overlapping blobs
        _blobs(d, rng, int(rng.integers(45, 75)), 4 * s, 8 * s, _PURPLE_DARK, size)
    elif cls == "ADI":  # adipose: web of thin grey membranes on white
        _lines(d, rng, int(rng.integers(7, 12)), (200, 190, 205), size, width=max(1, int(s)))
    elif cls == "BACK":  # empty glass
        pass
    elif cls == "DEB":  # granular debris
        _blobs(d, rng, int(rng.integers(40, 60)), 1 * s, 2.5 * s, (180, 100, 130), size)
    elif cls == "LYM":  # many small dark blue dots
        _blobs(d, rng, int(rng.integers(70, 110)), 2 * s, 3.2 * s, _BLUE_DARK, size)
    elif cls == "MUC":  # smooth pale blue mucus with a few soft blobs
        _blobs(d, rng, int(rng.integers(1, 4)), 5 * s, 9 * s, (170, 185, 215), size)
    elif cls == "MUS":  # smooth muscle: horizontal stripes, elongated nuclei
        _lines(d, rng, 14, (215, 140, 170), size, width=max(1, int(2 * s)), horizontal=True)
        _blobs(d, rng, int(rng.integers(5, 11)), 1.5 * s, 2.5 * s, _PURPLE, size, elongate=3.0)
    elif cls == "NORM":  # colon mucosa: nuclei arranged in a gland ring
        c = tuple(rng.uniform(size * 0.3, size * 0.7, 2))
        _blobs(d, rng, int(rng.integers(16, 26)), 2.5 * s, 3.5 * s, _PURPLE, size, center=c,
               ring_r=rng.uniform(18 * s, 30 * s))
    elif cls == "STR":  # stroma: wavy fibres with spindle nuclei
        _lines(d, rng, int(rng.integers(10, 16)), (215, 150, 180), size, width=max(1, int(s)),
               wavy=True)
        _blobs(d, rng, int(rng.integers(5, 9)), 1.5 * s, 2.5 * s, _PURPLE, size, elongate=2.5)

    arr = np.asarray(img).astype(np.float32)
    noise_sigma = {"BACK": 2.0, "MUC": 3.0, "DEB": 18.0}.get(cls, 6.0)
    arr += rng.normal(0, noise_sigma, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if cls in ("MUC", "NORM_PCAM"):
        out = out.filter(ImageFilter.GaussianBlur(radius=0.6 * s))
    return out


def make_synthetic(dataset: DatasetName = "pcam", n: int = 64, seed: int = 0, size: int = 96,
                   split: Split = "test") -> list[TileRecord]:
    """Offline fixture: ``n`` RGB tiles (default 96x96) with visibly different textures per
    class, labels round-robin over CLASSES[dataset] (so classes are balanced). tile_id uses
    ``split`` (default "test") so downstream Prediction records validate."""
    _check_args(dataset, split)
    rng = np.random.default_rng(seed)
    n_classes = len(CLASSES[dataset])
    labels = np.arange(n) % n_classes
    rng.shuffle(labels)
    return [
        TileRecord(tile_id=f"{dataset}:{split}:{i}", image=_draw_tile(rng, dataset, int(lab), size),
                   label=int(lab))
        for i, lab in enumerate(labels)
    ]


__all__ = [
    "HF_SOURCES",
    "TILE_SIZE",
    "TileRecord",
    "download",
    "ensure_split",
    "iter_split",
    "load_split",
    "make_synthetic",
    "records_to_parquet",
    "select_rows",
    "split_path",
]
