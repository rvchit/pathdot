"""Build Fireworks SFT datasets (OpenAI chat JSONL, base64 images in the user turn).

Writes two training sets so you can tell whether the synthetic data helped:
  ft_pathvqa_only.jsonl      PathVQA train (minus images that match test images)
  ft_pathvqa_plus_synth.jsonl  same + synthetic Q&A
and ft_val.jsonl (a PathVQA validation sample) for --evaluation-dataset.

    python scripts/build_fireworks_jsonl.py --data data --synthetic data/synthetic_qa.jsonl --out fireworks
"""
import argparse, json, random
from pathlib import Path
import imagehash
from common import SYSTEM_PROMPT, user_content, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--synthetic", default=None)
ap.add_argument("--out", default="fireworks")
ap.add_argument("--max-train", type=int, default=0, help="subsample PathVQA train (0 = all)")
ap.add_argument("--val-size", type=int, default=500)
ap.add_argument("--leak-dist", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
data, out = Path(args.data), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

test_hashes = [imagehash.hex_to_hash(h) for h in set(json.load(open(data / "test_phash.json")).values())]
_cache = {}
def leaks(phash):
    if phash not in _cache:
        h = imagehash.hex_to_hash(phash)
        _cache[phash] = min(h - t for t in test_hashes) <= args.leak_dist
    return _cache[phash]

def example(r):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content(r["image_path"], r["question"])},
        {"role": "assistant", "content": r["answer"]},
    ]}

def write(rows, name):
    p = out / name
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(example(r)) + "\n")
    mb = p.stat().st_size / 1e6
    print(f"{name}: {len(rows)} examples, {mb:.0f} MB" + ("  (>500 MB: upload with firectl)" if mb > 500 else ""))

train = read_jsonl(data / "pathvqa_train.jsonl")
before = len(train)
train = [r for r in train if not leaks(r["phash"])]
print(f"PathVQA train: removed {before - len(train)} questions whose image matches a test image")
if args.max_train and len(train) > args.max_train:
    train = random.sample(train, args.max_train)

val = [r for r in read_jsonl(data / "pathvqa_validation.jsonl") if not leaks(r["phash"])]
val = random.sample(val, min(args.val_size, len(val)))

write(train, "ft_pathvqa_only.jsonl")
write(val, "ft_val.jsonl")
if args.synthetic:
    synth = read_jsonl(args.synthetic)
    combined = train + synth
    random.shuffle(combined)
    write(combined, "ft_pathvqa_plus_synth.jsonl")
    print(f"synthetic share: {len(synth)}/{len(combined)} = {len(synth)/len(combined):.0%}")
