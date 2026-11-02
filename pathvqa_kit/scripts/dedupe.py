"""Remove new images that (near-)duplicate a PathVQA test image, or each other.

    python scripts/dedupe.py --images data/commons_images.jsonl --test-hashes data/test_phash.json \
        --out data/commons_images_clean.jsonl
"""
import argparse, json
import imagehash
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--test-hashes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-dist", type=int, default=8, help="phash Hamming distance counted as duplicate")
args = ap.parse_args()

test = [imagehash.hex_to_hash(h) for h in set(json.load(open(args.test_hashes)).values())]
kept_hashes, kept, drop_test, drop_dup = [], [], 0, 0
for line in open(args.images):
    m = json.loads(line)
    h = imagehash.phash(Image.open(m["image_path"]).convert("RGB"))
    if test and min(h - t for t in test) <= args.max_dist:
        drop_test += 1
        continue
    if any(h - k <= args.max_dist for k in kept_hashes):
        drop_dup += 1
        continue
    kept_hashes.append(h)
    m["phash"] = str(h)
    kept.append(m)

with open(args.out, "w") as f:
    for m in kept:
        f.write(json.dumps(m) + "\n")
print(f"kept {len(kept)} | dropped {drop_test} matching test images | {drop_dup} internal duplicates")
