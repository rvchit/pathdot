"""Download PathVQA from Hugging Face, save normalized images, write per-split records,
and compute perceptual hashes of test images (used to keep test images out of training).

    python scripts/prepare_pathvqa.py --out data
Outputs: data/images/pathvqa/<sha>.jpg, data/pathvqa_{train,validation,test}.jsonl, data/test_phash.json
"""
import argparse, hashlib, io, json
from pathlib import Path
import imagehash
from datasets import load_dataset
from common import save_jpeg, is_closed, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--hf-id", default="flaviagiammarino/path-vqa")
args = ap.parse_args()
out = Path(args.out)
img_dir = out / "images" / "pathvqa"

ds = load_dataset(args.hf_id)
print(ds)
test_hashes = {}
for split in ds:
    rows, seen = [], {}
    for i, ex in enumerate(ds[split]):
        img = ex["image"]
        buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
        sha = hashlib.sha1(buf.getvalue()).hexdigest()[:16]
        p = img_dir / f"{sha}.jpg"
        if sha not in seen:
            if not p.exists():
                save_jpeg(img, p)
            seen[sha] = str(imagehash.phash(img.convert("RGB")))
        if split == "test":
            test_hashes[sha] = seen[sha]
        rows.append({
            "id": f"pathvqa-{split}-{i}", "split": split, "source": "pathvqa",
            "image_path": str(p), "image_sha": sha, "phash": seen[sha],
            "question": ex["question"], "answer": str(ex["answer"]),
            "qtype": "closed" if is_closed(ex["answer"]) else "open",
        })
    write_jsonl(rows, out / f"pathvqa_{split}.jsonl")
    n_closed = sum(r["qtype"] == "closed" for r in rows)
    print(f"{split}: {len(rows)} questions, {len(seen)} unique images, {n_closed} yes/no")

json.dump(test_hashes, open(out / "test_phash.json", "w"))

# Leakage report: train/val questions whose image is (near-)identical to a test image.
test_list = [imagehash.hex_to_hash(h) for h in set(test_hashes.values())]
for split in ("train", "validation"):
    f = out / f"pathvqa_{split}.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(l) for l in open(f)]
    uniq = {r["phash"] for r in rows}
    leaked = {h for h in uniq if min((imagehash.hex_to_hash(h) - t) for t in test_list) <= 4}
    print(f"{split}: {len(leaked)}/{len(uniq)} images near-duplicate a test image "
          f"(removed later by build_fireworks_jsonl.py)")
