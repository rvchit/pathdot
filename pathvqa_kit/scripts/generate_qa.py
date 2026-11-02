"""Write PathVQA-style Q&A for each new image with Claude, grounded in the image's caption.

Claude only states facts the caption supports, so labels inherit the caption author's
expertise instead of relying on Claude reading the slide.

    export ANTHROPIC_API_KEY=...
    python scripts/generate_qa.py --images data/commons_images_clean.jsonl --out data/synthetic_qa.jsonl
"""
import argparse, asyncio, json, os, re
from pathlib import Path
import anthropic
from common import data_uri, is_closed

PROMPT = """You are building a pathology visual question answering training set in the style of PathVQA.

Image caption (written by the image's author, treat as ground truth):
\"\"\"{caption}\"\"\"

Write {n} question-answer pairs about this image.
Rules:
- Every answer must be directly supported by the caption. Use the image only to phrase questions naturally; never add findings the caption does not state.
- About half should be yes/no questions. Include some whose correct answer is "no", but only when the caption makes that answer certain (e.g. asking about a different organ or stain than the caption states).
- The rest are short-answer questions ("What organ is shown?", "What stain was used?", "What is the diagnosis?", "What structures are seen?") with answers of 1-3 words.
- Do not mention the caption in the question. Do not ask about magnification numbers, authors, or file details.
- If the caption is not about a pathology/histology/microscopy image, return an empty list.

Return only JSON: [{{"question": "...", "answer": "..."}}, ...]"""


def parse(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        q, a = str(it.get("question", "")).strip(), str(it.get("answer", "")).strip()
        if q and a and len(a.split()) <= 5:
            out.append({"question": q, "answer": a.lower() if is_closed(a) else a})
    return out


async def one(client, sem, m, args):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.messages.create(
                    model=args.model, max_tokens=800,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": data_uri(m["image_path"]).split(",", 1)[1]}},
                        {"type": "text", "text": PROMPT.format(caption=m["caption"][:1500], n=args.per_image)},
                    ]}])
                return m, parse(r.content[0].text)
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
                err = e
        print("failed", m.get("title"), err)
        return m, []


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5"))
    ap.add_argument("--per-image", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    images = [json.loads(l) for l in open(args.images)]
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(one(client, sem, m, args) for m in images))

    n = 0
    with open(args.out, "w") as f:
        for m, qas in results:
            for j, qa in enumerate(qas):
                f.write(json.dumps({
                    "id": f"synthetic-{m['pageid']}-{j}", "split": "train", "source": m["source"],
                    "image_path": m["image_path"], "question": qa["question"], "answer": qa["answer"],
                    "qtype": "closed" if is_closed(qa["answer"]) else "open",
                    "license": m.get("license"), "attribution": m.get("artist"), "source_url": m.get("page_url"),
                }) + "\n")
                n += 1
    print(f"{n} Q&A from {sum(bool(q) for _, q in results)}/{len(images)} images -> {args.out}")


asyncio.run(main())
