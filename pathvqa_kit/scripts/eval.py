"""Run a model on the PathVQA test set and score it. Same prompt and images for every model.

Run predictions (OpenAI-compatible endpoint, e.g. Fireworks):
    export FIREWORKS_API_KEY=...
    python scripts/eval.py run --model accounts/fireworks/models/qwen2p5-vl-7b-instruct \
        --name base --data data --n 1000
    python scripts/eval.py run --model accounts/<you>/deployedModels/<id> --name ft_synth --data data --n 1000
    # frontier baseline via Anthropic:
    python scripts/eval.py run --provider anthropic --model claude-sonnet-4-5 --name frontier --data data --n 1000

Score and compare (bootstrap 95% CIs + McNemar paired test):
    python scripts/eval.py score results/base.jsonl results/ft_synth.jsonl results/frontier.jsonl

Metrics: yes/no accuracy (closed), exact match and token recall on open questions,
overall exact match, latency p50/p95, output tokens.
Test sampling is seeded, so --n 1000 picks the same questions for every model.
"""
import argparse, asyncio, json, os, random, sys, time
from pathlib import Path
import numpy as np
from common import SYSTEM_PROMPT, user_content, data_uri, norm_answer, read_jsonl


# ---------------------------------------------------------------- scoring
def closed_pred(p):
    t = norm_answer(p).split()
    return t[0] if t and t[0] in ("yes", "no") else ""

def correct(r):
    gt, pr = norm_answer(r["answer"]), norm_answer(r["prediction"])
    if r["qtype"] == "closed":
        return float(closed_pred(r["prediction"]) == gt)
    return float(pr == gt)

def recall(r):  # standard open-ended PathVQA metric: share of GT tokens present in prediction
    gt = norm_answer(r["answer"]).split()
    pr = set(norm_answer(r["prediction"]).split())
    return sum(t in pr for t in gt) / max(len(gt), 1)

def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def mcnemar(a, b):
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)  # exact binomial
    return p, n01, n10

def fmt(m, lo, hi):
    return f"{100*m:5.1f} ({100*lo:.1f}-{100*hi:.1f})"

def score(files):
    runs = {Path(f).stem: {r["id"]: r for r in read_jsonl(f)} for f in files}
    common_ids = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"scoring {len(common_ids)} questions answered by every model\n")
    print(f"{'model':22} {'yes/no acc':>20} {'open exact':>20} {'open recall':>20} {'overall exact':>20} {'p50 s':>6} {'p95 s':>6} {'out tok':>7}")
    per = {}
    for name, rs in runs.items():
        rows = [rs[i] for i in common_ids]
        c = [correct(r) for r in rows]
        per[name] = c
        closed = [correct(r) for r in rows if r["qtype"] == "closed"]
        open_ = [r for r in rows if r["qtype"] == "open"]
        lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        tok = [r.get("output_tokens") or 0 for r in rows]
        print(f"{name:22} {fmt(*boot_ci(closed)):>20} {fmt(*boot_ci([correct(r) for r in open_])):>20} "
              f"{fmt(*boot_ci([recall(r) for r in open_])):>20} {fmt(*boot_ci(c)):>20} "
              f"{np.percentile(lat,50) if lat else float('nan'):6.2f} {np.percentile(lat,95) if lat else float('nan'):6.2f} {np.mean(tok):7.1f}")
    names = list(per)
    if len(names) > 1:
        print("\npaired McNemar test on overall exact match (n01 = first right/second wrong):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p, n01, n10 = mcnemar(per[names[i]], per[names[j]])
                d = np.mean(per[names[j]]) - np.mean(per[names[i]])
                print(f"  {names[i]} vs {names[j]}: diff {100*d:+.1f} pts, n01={n01}, n10={n10}, p={p:.4f}")


# ---------------------------------------------------------------- running
SHOTS = []  # few-shot demonstrations (list of train rows), filled by --shots
EXTRA_BODY = {}  # extra OpenAI-compatible request fields, e.g. {"reasoning_effort": "none"} (--no-think)

def shot_messages():
    """In-context demonstrations as prior user/assistant turns (image + question -> answer)."""
    msgs = []
    for ex in SHOTS:
        msgs.append({"role": "user", "content": user_content(ex["image_path"], ex["question"])})
        msgs.append({"role": "assistant", "content": norm_answer(ex["answer"])})
    return msgs

def pick_shots(data_dir, k, seed=0):
    """k seeded train examples, half yes/no and half open, never from the test set."""
    train = read_jsonl(Path(data_dir) / "pathvqa_train.jsonl")
    rng = random.Random(seed)
    closed = [r for r in train if r["qtype"] == "closed"]
    open_ = [r for r in train if r["qtype"] == "open"]
    rng.shuffle(closed); rng.shuffle(open_)
    shots = closed[: k // 2] + open_[: k - k // 2]
    rng.shuffle(shots)
    return shots

async def ask_openai(client, model, r, max_tokens):
    t = time.time()
    resp = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens, extra_body=EXTRA_BODY or None,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *shot_messages(),
                  {"role": "user", "content": user_content(r["image_path"], r["question"])}])
    return resp.choices[0].message.content or "", time.time() - t, getattr(resp.usage, "completion_tokens", None)

async def ask_anthropic(client, model, r, max_tokens):
    t = time.time()
    resp = await client.messages.create(
        model=model, max_tokens=max_tokens, temperature=0, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": data_uri(r["image_path"]).split(",", 1)[1]}},
            {"type": "text", "text": r["question"]}]}])
    return resp.content[0].text, time.time() - t, resp.usage.output_tokens

async def run(args):
    test = read_jsonl(Path(args.data) / "pathvqa_test.jsonl")
    if args.n and args.n < len(test):
        random.Random(0).shuffle(test)
        test = sorted(test[:args.n], key=lambda r: r["id"])
    if args.no_think:
        # thinking models (e.g. Qwen3.5) otherwise spend the whole max_tokens budget on reasoning
        EXTRA_BODY["reasoning_effort"] = "none"
        print("reasoning disabled (reasoning_effort=none)")
    if args.shots:
        SHOTS.extend(pick_shots(args.data, args.shots))
        print(f"few-shot: {len(SHOTS)} train demonstrations prepended to every request")
    if args.provider == "anthropic":
        import anthropic
        client, ask = anthropic.AsyncAnthropic(), ask_anthropic
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env))
        ask = ask_openai
    out = Path(args.results) / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in test if r["id"] not in done]
    print(f"{args.name}: {len(todo)} to run ({len(done)} already done) -> {out}")
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    f = open(out, "a")

    async def one(r):
        async with sem:
            for attempt in range(5):
                try:
                    pred, lat, tok = await ask(client, args.model, r, args.max_tokens)
                    break
                except Exception as e:
                    if attempt == 4:
                        print("failed", r["id"], e); pred, lat, tok = "", None, None
                    await asyncio.sleep(2 ** attempt)
            rec = {k: r[k] for k in ("id", "question", "answer", "qtype")}
            rec.update(prediction=pred, latency_s=lat, output_tokens=tok, model=args.model, shots=args.shots, no_think=args.no_think)
            async with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()

    await asyncio.gather(*(one(r) for r in todo))
    f.close()
    score([out])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    r.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    r.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    r.add_argument("--data", default="data")
    r.add_argument("--results", default="results")
    r.add_argument("--n", type=int, default=0, help="seeded test subsample (0 = full test set)")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--shots", type=int, default=0, help="prepend N train examples as few-shot demos (openai provider)")
    r.add_argument("--no-think", action="store_true", help="send reasoning_effort=none (needed for thinking models like Qwen3.5)")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        score(a.files)
