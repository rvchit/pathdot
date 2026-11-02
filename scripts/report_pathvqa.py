"""Summarise PathVQA runs (pathvqa_kit/scripts/eval.py output) the same way report_camelyon17.py does:
accuracy with bootstrap CIs, cost (deployment GPU time), speed; plus a COMPARISON.md with paired
McNemar tests across runs on the same seeded 1,000-question test subset.

    .venv/bin/python scripts/report_pathvqa.py results/pathvqa/preds_*_test.jsonl --out-dir results/pathvqa
Each preds file may have a sibling run_info_<stem>.json with {"model","deployment_shape","accelerator",
"precision","gpu_minutes","gpu_rate_usd_per_hour","training":{...},"setup":"..."}.
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)
DEFAULT_OUT = typer.Option(Path("results/pathvqa"))


def norm(a: str) -> str:
    a = re.sub(r"[^\w\s]", " ", str(a).lower().strip())
    return re.sub(r"\s+", " ", a).strip()


def correct(r: dict) -> float:
    gt, pr = norm(r["answer"]), norm(r["prediction"])
    if r["qtype"] == "closed":
        t = pr.split()
        return float(bool(t) and t[0] in ("yes", "no") and t[0] == gt)
    return float(pr == gt)


def recall(r: dict) -> float:
    gt = norm(r["answer"]).split()
    pr = set(norm(r["prediction"]).split())
    return sum(t in pr for t in gt) / max(len(gt), 1)


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    n10 = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n), n01, n10


def pct(t):
    m, lo, hi = t
    return "n/a" if m is None else f"{100*m:.1f}% ({100*lo:.1f}-{100*hi:.1f})"


def summarise(path: Path, run_info: dict) -> dict:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    closed = [r for r in rows if r["qtype"] == "closed"]
    open_ = [r for r in rows if r["qtype"] == "open"]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    tok = [r.get("output_tokens") or 0 for r in rows]
    corr = [correct(r) for r in rows]
    gpu_min = run_info.get("gpu_minutes")
    rate = run_info.get("gpu_rate_usd_per_hour")
    return {
        "model": run_info.get("model") or rows[0].get("model"),
        "preds_file": str(path),
        "run_info": run_info,
        "overall": {
            "n": len(rows), "n_closed": len(closed), "n_open": len(open_),
            "closed_accuracy": boot([correct(r) for r in closed]),
            "open_exact": boot([correct(r) for r in open_]),
            "open_recall": boot([recall(r) for r in open_]),
            "overall_exact": boot(corr),
            "shots": rows[0].get("shots", 0), "no_think": rows[0].get("no_think", False),
        },
        "speed": {
            "latency_p50_ms": float(np.percentile(lat, 50) * 1000) if lat else None,
            "latency_p95_ms": float(np.percentile(lat, 95) * 1000) if lat else None,
            "output_tokens_mean": float(np.mean(tok)),
            "wall_s_est": float(sum(lat) / 16) if lat else None,
        },
        "cost": {
            "gpu_minutes": gpu_min, "gpu_rate_usd_per_hour": rate,
            "gpu_cost_usd": (gpu_min / 60 * rate) if gpu_min and rate else None,
            "cost_per_1k_questions_usd": (gpu_min / 60 * rate / len(rows) * 1000) if gpu_min and rate else None,
            "training": run_info.get("training"),
        },
        "_correct": corr, "_ids": [r["id"] for r in rows],
    }


def write_md(s: dict, out: Path) -> None:
    o, sp, c, ri = s["overall"], s["speed"], s["cost"], s["run_info"]
    lines = [f"# PathVQA: `{s['model']}`", "",
             (f"{o['n']} test questions (seeded subset of the 6,719-question PathVQA test set; {o['n_closed']} yes/no, "
              f"{o['n_open']} open). {ri.get('setup', '')}"), "", "## Accuracy", "", "| metric | value |", "|---|---|",
             f"| yes/no accuracy | {pct(o['closed_accuracy'])} |", f"| open exact match | {pct(o['open_exact'])} |",
             f"| open token recall (PathVQA metric) | {pct(o['open_recall'])} |", f"| overall exact match | {pct(o['overall_exact'])} |",
             f"| few-shot examples | {o['shots']} |", "", "## Cost", "", "| metric | value |", "|---|---|"]
    if c["gpu_minutes"]:
        lines += [f"| deployment GPU time for this eval | {c['gpu_minutes']:.1f} min x {ri.get('accelerator_count', 1)} {ri.get('accelerator', '')} |",
                  f"| GPU cost (@ ${c['gpu_rate_usd_per_hour']:.2f}/GPU-h) | ${c['gpu_cost_usd']:.2f} (${c['cost_per_1k_questions_usd']:.2f} per 1k questions) |"]
    if c.get("training"):
        t = c["training"]
        lines += [f"| training | {t.get('method', '')}, {t.get('examples', '')} examples, {t.get('minutes', '')} min, ~${t.get('cost_usd_est', '')} ({t.get('trained_tokens', '')} tokens) |"]
    lines += ["", "## Speed", "", "| metric | value |", "|---|---|",
              f"| latency p50 / p95 | {sp['latency_p50_ms']:.0f} / {sp['latency_p95_ms']:.0f} ms |",
              f"| output tokens per answer | {sp['output_tokens_mean']:.1f} |",
              f"| wall time (concurrency 16) | {sp['wall_s_est']/60:.1f} min |"]
    if ri.get("deployment_shape"):
        lines += [f"| deployment | {ri['deployment_shape']} ({ri.get('precision', '')}) |"]
    out.write_text("\n".join(lines) + "\n")


@app.command()
def main(preds: list[Path], out_dir: Path = DEFAULT_OUT):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for p in preds:
        stem = p.stem.replace("preds_", "").replace("_test", "")
        ri_path = out_dir / f"run_info_{stem}.json"
        ri = json.loads(ri_path.read_text()) if ri_path.exists() else {}
        s = summarise(p, ri)
        runs[stem] = s
        public = {k: v for k, v in s.items() if not k.startswith("_")}
        (out_dir / f"{stem}.json").write_text(json.dumps(public, indent=2))
        write_md(s, out_dir / f"{stem}.md")
        typer.echo(f"{stem}: overall {pct(s['overall']['overall_exact'])}  yes/no {pct(s['overall']['closed_accuracy'])}")
    ids = set.intersection(*(set(s["_ids"]) for s in runs.values()))
    lines = ["# PathVQA comparison", "", f"Same {len(ids)} seeded test questions for every run; paired exact McNemar test on overall exact match.", "",
             "| run | yes/no acc | open exact | open recall | overall exact | p50 ms | $/1k questions |", "|---|---|---|---|---|---|---|"]
    for k, s in runs.items():
        o, c = s["overall"], s["cost"]
        per_1k = c["cost_per_1k_questions_usd"]
        cost_txt = f"${per_1k:.2f}" if per_1k else "n/a"
        lines.append(f"| {k} | {pct(o['closed_accuracy'])} | {pct(o['open_exact'])} | {pct(o['open_recall'])} | {pct(o['overall_exact'])} | "
                     f"{s['speed']['latency_p50_ms']:.0f} | {cost_txt} |")
    lines += ["", "| A | B | B - A (pts) | n01 (A right, B wrong) | n10 | p |", "|---|---|---|---|---|---|"]
    for a, b in combinations(runs, 2):
        ia = {i: c for i, c in zip(runs[a]["_ids"], runs[a]["_correct"], strict=True)}
        ib = {i: c for i, c in zip(runs[b]["_ids"], runs[b]["_correct"], strict=True)}
        ca = [ia[i] for i in sorted(ids)]; cb = [ib[i] for i in sorted(ids)]
        p, n01, n10 = mcnemar(ca, cb)
        lines.append(f"| {a} | {b} | {100*(np.mean(cb)-np.mean(ca)):+.1f} | {n01} | {n10} | {p:.4g} |")
    (out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    typer.echo(f"wrote {out_dir}/COMPARISON.md")


if __name__ == "__main__":
    app()
