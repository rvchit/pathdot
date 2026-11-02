"""Build the curated demo gallery: data/demo_gallery/manifest.json + image files.

    scan: 3 tiles (2 tumor + 1 normal) from the Camelyon17-WILDS test shards in
          data/camelyon17_wilds_test/**/*.tar (keys picked from data/camelyon17/test_manifest.csv,
          preferring tiles both the probe and the fine-tuned VLM got right on the frozen test set).
          Without a shard the tiles come from PCam test (manifest "source": "pcam").
    ask:  3 PathVQA test images with 3-6 reference questions (>= 1 closed, >= 1 open, short answers)
          copied from pathvqa_kit/data/images/pathvqa.

Idempotent: an existing manifest is kept unless --force. Served by GET /gallery (pathfrontier.serve).
After the manifest, every gallery image is also tiled into a DZI pyramid under tiles/<id>/ (skipped
when the .dzi exists unless --force) so the slide viewer lists scan_1..3 / ask_1..3 as slides.

    .venv/bin/python scripts/make_demo_gallery.py [--force] [--tiles-dir tiles]
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import typer
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config, dzi, gallery

TILE_PX = 224
N_TUMOR, N_NORMAL, N_ASK = 2, 1, 3
PROBE_PREDS = config.RESULTS_DIR / "camelyon17" / "preds_probe_phikon-v2_c17train_test.jsonl"
VLM_PREDS = config.RESULTS_DIR / "camelyon17" / "preds_qwen3p5-9b_ft_v2_test.jsonl"
PATHVQA_JSONL = config.ROOT / "pathvqa_kit" / "data" / "pathvqa_test.jsonl"
PATHVQA_SCRIPTS = config.ROOT / "pathvqa_kit" / "scripts"   # image_path is relative to this dir
MAX_OPEN_ANSWER_WORDS = 4
MAX_QUESTION_CHARS = 80

app = typer.Typer(add_completion=False)


# --------------------------------------------------------------------------------------
# Camelyon17 shards
# --------------------------------------------------------------------------------------
def index_shards(shard_dir: Path) -> dict[int, set[str]]:
    """shard index -> set of sample keys readable from that tar (partial downloads tolerated)."""
    out: dict[int, set[str]] = {}
    for tar_path in sorted(shard_dir.rglob("*.tar")):
        if not tar_path.stem.isdigit():
            continue
        keys: list[str] = []
        try:
            with tarfile.open(tar_path) as tar:
                for m in tar:
                    if m.name.endswith(".png"):
                        keys.append(m.name[:-4])
        except (tarfile.TarError, EOFError, OSError) as e:
            typer.echo(f"  shard {tar_path.name}: truncated/partial ({e}); using {len(keys)} keys")
            keys = keys[:-1]  # the last member may be cut off
        if keys:
            out[int(tar_path.stem)] = set(keys)
            typer.echo(f"  shard {tar_path.name}: {len(keys)} tiles")
    return out


def read_tile(shard_dir: Path, shard: int, key: str) -> Image.Image:
    for tar_path in shard_dir.rglob(f"{shard}.tar"):
        with tarfile.open(tar_path) as tar:
            member = tar.getmember(f"{key}.png")
            return Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB")
    raise FileNotFoundError(f"{shard}.tar under {shard_dir}")


def load_preds(path: Path) -> dict[str, float]:
    """tile_id -> p_tumor from a predictions JSONL (empty when the file is absent)."""
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["tile_id"]] = float(r["y_prob"][1])
    return out


def pick_camelyon(shard_dir: Path, manifest_dir: Path) -> tuple[list[dict], str]:
    shards = index_shards(shard_dir)
    if not shards:
        return [], "no shards"
    probe_p, vlm_p = load_preds(PROBE_PREDS), load_preds(VLM_PREDS)
    cutoff = gallery.load_cutoff()

    def rows(split: str) -> list[dict]:
        p = manifest_dir / f"{split}_manifest.csv"
        if not p.is_file():
            return []
        with open(p) as f:
            rs = [r for r in csv.DictReader(f)
                  if int(r["shard"]) in shards and r["key"] in shards[int(r["shard"])]]
        for r in rs:
            r["split"] = split
            r["row"] = int(r["tile_id"].rsplit(":", 1)[1])
        return rs

    def score(r: dict) -> tuple:
        """Lower is better: both models confidently right first, then manifest order."""
        y = int(r["label"])
        pp, vp = probe_p.get(r["tile_id"]), vlm_p.get(r["tile_id"])
        probe_ok = pp is not None and ((pp >= 0.9) if y else (pp <= 0.1))
        vlm_ok = vp is not None and ((vp >= max(cutoff, 0.5)) if y else (vp < cutoff / 2))
        return (0 if (probe_ok and vlm_ok) else 1 if probe_ok else 2, r["row"])

    chosen: list[dict] = []
    for label, n in ((1, N_TUMOR), (0, N_NORMAL)):
        pool = sorted([r for r in rows("test") if int(r["label"]) == label], key=score)
        if len(pool) < n:   # the downloaded shard lacks this class in the eval slides
            extra = sorted([r for r in rows("train") if int(r["label"]) == label],
                           key=lambda r: r["row"])
            typer.echo(f"  label {label}: only {len(pool)} test tiles in the shards; "
                       f"using {n - len(pool)} from the training pool")
            pool += extra
        picked, slides = [], set()
        for r in pool:          # prefer distinct slides for variety
            if r["slide"] in slides and len(pool) - pool.index(r) > n - len(picked):
                continue
            picked.append(r)
            slides.add(r["slide"])
            if len(picked) == n:
                break
        if len(picked) < n:
            picked = pool[:n]
        chosen += picked
    if sum(int(r["label"]) for r in chosen) < N_TUMOR or len(chosen) < N_TUMOR + N_NORMAL:
        return [], "not enough tiles in the downloaded shards"

    items = []
    for i, r in enumerate(chosen, 1):
        y = int(r["label"])
        img = read_tile(shard_dir, int(r["shard"]), r["key"])
        items.append({
            "id": f"scan_{i}", "section": "scan", "label": y,
            "label_name": "tumor" if y else "normal", "source": "camelyon17",
            "meta": {"tile_id": r["tile_id"], "slide": int(r["slide"]), "key": r["key"],
                     "shard": int(r["shard"]), "hospital": int(r["hospital"]), "split": r["split"],
                     "probe_p_tumor": probe_p.get(r["tile_id"]),
                     "vlm_ft_p_tumor": vlm_p.get(r["tile_id"])},
            "file": f"scan_{i}.png", "_image": img,
        })
    return items, "camelyon17"


def pick_pcam() -> list[dict]:
    from pathfrontier.data import load_split

    recs = load_split("pcam", "test", limit=200, seed=0)
    items = []
    for label, n in ((1, N_TUMOR), (0, N_NORMAL)):
        for r in [r for r in recs if int(r.label) == label][:n]:
            i = len(items) + 1
            items.append({
                "id": f"scan_{i}", "section": "scan", "label": label,
                "label_name": "tumor" if label else "normal", "source": "pcam",
                "meta": {"tile_id": r.tile_id, "slide": None, "key": r.tile_id.rsplit(":", 1)[1]},
                "file": f"scan_{i}.png", "_image": r.image,
            })
    return items


# --------------------------------------------------------------------------------------
# PathVQA
# --------------------------------------------------------------------------------------
def pick_pathvqa() -> list[dict]:
    if not PATHVQA_JSONL.is_file():
        typer.echo(f"  no PathVQA test file at {PATHVQA_JSONL}; ask section empty")
        return []
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(PATHVQA_JSONL) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_image[r["image_path"]].append(r)

    def usable(qs: list[dict]) -> list[dict] | None:
        seen, uniq = set(), []
        for q in qs:
            k = (q["question"].strip().lower(), q["qtype"])
            if k not in seen:
                seen.add(k)
                uniq.append(q)
        if not 3 <= len(uniq) <= 6:
            return None
        types = {q["qtype"] for q in uniq}
        if "closed" not in types or "open" not in types:
            return None
        for q in uniq:
            if len(q["question"]) > MAX_QUESTION_CHARS or "\n" in q["question"]:
                return None
            if q["qtype"] == "open" and len(q["answer"].split()) > MAX_OPEN_ANSWER_WORDS:
                return None
        answers = {q["answer"].strip().lower() for q in uniq if q["qtype"] == "closed"}
        if not {"yes", "no"} <= answers:   # one yes and one no makes a better demo
            return None
        return uniq

    items = []
    for image_path in sorted(by_image):
        src = (PATHVQA_SCRIPTS / image_path).resolve()
        qs = usable(by_image[image_path])
        if qs is None or not src.is_file():
            continue
        i = len(items) + 1
        items.append({
            "id": f"ask_{i}", "section": "ask", "file": f"ask_{i}.jpg",
            "questions": [{"question": q["question"], "answer": q["answer"], "qtype": q["qtype"]}
                          for q in qs],
            "meta": {"image_path": image_path, "image_sha": qs[0].get("image_sha"),
                     "ids": [q["id"] for q in qs]},
            "_src": src,
        })
        if len(items) == N_ASK:
            break
    return items


# --------------------------------------------------------------------------------------
def tile_gallery(manifest: dict, gallery_dir: Path, tiles_dir: Path, force: bool = False) -> None:
    """DZI-tile each gallery image into tiles/<id>/ (ids == gallery ids) for the slide viewer."""
    for it in manifest["scan"] + manifest["ask"]:
        image_id = it["id"]
        if dzi.find_dzi(tiles_dir, image_id) is not None and not force:
            typer.echo(f"  tiles/{image_id}: exists (skip; --force to rebuild)")
            continue
        src = gallery_dir / it["file"]
        if not src.is_file():
            typer.echo(f"  tiles/{image_id}: missing image {src}; skipped")
            continue
        with Image.open(src) as im:
            im.load()
            out = dzi.tile_image_pillow(im, tiles_dir, image_id)
        info = dzi.parse_dzi(out)
        typer.echo(f"  tiles/{image_id}: {info['width']}x{info['height']}, max_level={info['max_level']}")


def summary(manifest: dict) -> str:
    lines = [(f"gallery: {len(manifest['scan'])} scan + {len(manifest['ask'])} ask items "
              f"(scan source: {manifest.get('source')})")]
    for it in manifest["scan"]:
        m = it["meta"]
        lines.append(f"  {it['id']}: {it['label_name']} (label {it['label']}) "
                     f"{it['source']} {m.get('tile_id')} slide={m.get('slide')} key={m.get('key')}")
    for it in manifest["ask"]:
        lines.append(f"  {it['id']}: {it['meta']['image_path']}")
        for q in it["questions"]:
            lines.append(f"      [{q['qtype']}] {q['question']} -> {q['answer']}")
    return "\n".join(lines)


@app.command()
def main(
    out_dir: str = typer.Option("", help="default: data/demo_gallery (PATHFRONTIER_GALLERY_DIR)"),
    shard_dir: str = typer.Option("", help="default: data/camelyon17_wilds_test"),
    force: bool = typer.Option(False, "--force", help="rebuild even if the manifest exists"),
    tiles_dir: str = typer.Option("", help="DZI output root (default: tiles/, PATHFRONTIER_TILES_DIR)"),
    no_tiles: bool = typer.Option(False, "--no-tiles", help="skip tiling the gallery images for the viewer"),
) -> None:
    out = Path(out_dir) if out_dir else gallery.default_gallery_dir()
    shards = Path(shard_dir) if shard_dir else config.DATA_DIR / "camelyon17_wilds_test"
    tiles = Path(tiles_dir or os.getenv("PATHFRONTIER_TILES_DIR") or config.ROOT / "tiles")
    mpath = out / gallery.MANIFEST_NAME
    if mpath.is_file() and not force:
        typer.echo(f"{mpath} exists (use --force to rebuild)")
        manifest = gallery.load_manifest(out)
        typer.echo(summary(manifest))
        if not no_tiles:
            typer.echo(f"tiling gallery images into {tiles} ...")
            tile_gallery(manifest, out, tiles)
        return

    typer.echo(f"indexing shards in {shards} ...")
    scan_items, source = pick_camelyon(shards, config.DATA_DIR / "camelyon17")
    if not scan_items:
        typer.echo(f"  camelyon17 unavailable ({source}); falling back to PCam test tiles")
        scan_items, source = pick_pcam(), "pcam"
    ask_items = pick_pathvqa()

    out.mkdir(parents=True, exist_ok=True)
    for it in scan_items:
        img = it.pop("_image").resize((TILE_PX, TILE_PX), Image.NEAREST)
        img.save(out / it["file"], format="PNG")
    for it in ask_items:
        shutil.copyfile(it.pop("_src"), out / it["file"])
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "tile_px": TILE_PX,
        "scan": scan_items,
        "ask": ask_items,
    }
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    typer.echo(f"wrote {mpath}")
    typer.echo(summary(manifest))
    if not no_tiles:
        typer.echo(f"tiling gallery images into {tiles} ...")
        tile_gallery(manifest, out, tiles, force=force)


if __name__ == "__main__":
    app()
