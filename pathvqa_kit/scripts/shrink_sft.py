"""Make a smaller copy of an SFT JSONL (re-encode images, optional subsample) WITHOUT touching the
source. Target: under Fireworks' 150 MB direct-upload limit.
  python shrink_sft.py ../fireworks/ft_pathvqa_only.jsonl ../fireworks_small/ft_train_4000_384.jsonl --n 4000 --px 384 --q 80
"""
import argparse, base64, io, json, random, sys
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--n", type=int, default=0, help="seeded random subset size (0 = all)")
ap.add_argument("--px", type=int, default=384, help="max side in px")
ap.add_argument("--q", type=int, default=80, help="jpeg quality")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

def shrink(url: str) -> str:
    head, b64 = url.split(",", 1)
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    im.thumbnail((a.px, a.px))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=a.q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

with open(a.src) as f:
    lines = f.readlines()
if a.n and a.n < len(lines):
    random.Random(a.seed).shuffle(lines); lines = lines[: a.n]
out_bytes = 0
with open(a.dst, "w") as out:
    for i, line in enumerate(lines):
        ex = json.loads(line)
        for m in ex["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        if isinstance(part["image_url"], dict):
                            part["image_url"]["url"] = shrink(part["image_url"]["url"])
                        else:
                            part["image_url"] = shrink(part["image_url"])
        s = json.dumps(ex, separators=(",", ":")) + "\n"
        out.write(s); out_bytes += len(s)
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{len(lines)} {out_bytes/1e6:.1f} MB", file=sys.stderr)
print(f"wrote {a.dst}: {len(lines)} examples, {out_bytes/1e6:.1f} MB")
