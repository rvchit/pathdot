"""Summarise a Camelyon17 run: accuracy, token usage / cost, and speed, overall and per slide.

    .venv/bin/python scripts/report_camelyon17.py results/camelyon17/preds_<model>_test.jsonl \
        --run-info results/camelyon17/run_info.json --out results/camelyon17/baseline

`--run-info` is an optional JSON with wall-clock and deployment facts recorded by the caller:
{"wall_s": ..., "gpu_minutes": ..., "gpu_rate_usd_per_hour": ..., "accelerator": "...",
 "accelerator_count": 1, "deploy_ready_s": ...}. GPU cost is what an on-demand deployment bills;
the per-token cost in the predictions is only the client's price-table proxy.
Writes <out>.json and <out>.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.schema import read_predictions

app = typer.Typer(add_completion=False)


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def binary_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    return {
        "n": len(y), "n_tumor": int((y == 1).sum()), "n_normal": int((y == 0).sum()),
        "accuracy": _div(tp + tn, len(y)),
        "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "auroc": float(roc_auc_score(y, prob)) if len(set(y.tolist())) == 2 else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


@app.command()
def main(
    preds_file: Path,
    manifest: str = typer.Option("", help="default: data/camelyon17/test_manifest.csv"),
    run_info: str = typer.Option("", help="optional run facts JSON (see module docstring)"),
    out: str = typer.Option("", help="output stem; default: results/camelyon17/baseline"),
) -> None:
    manifest_path = Path(manifest) if manifest else config.DATA_DIR / "camelyon17" / "test_manifest.csv"
    out = Path(out) if out else config.RESULTS_DIR / "camelyon17" / "baseline"
    preds = list(read_predictions(preds_file))
    with open(manifest_path) as f:
        slide_of = {r["tile_id"]: int(r["slide"]) for r in csv.DictReader(f)}
    info = json.loads(Path(run_info).read_text()) if run_info else {}

    y = np.array([p.y_true for p in preds])
    pred = np.array([p.y_pred for p in preds])
    prob = np.array([p.y_prob[1] for p in preds])
    overall = binary_metrics(y, pred, prob)
    overall["accuracy_95ci"] = wilson(int((y == pred).sum()), len(y))
    overall["parse_errors"] = sum(bool(p.extra.get("parse_error")) for p in preds)
    overall["api_errors"] = sum(bool(p.extra.get("api_error")) for p in preds)
    overall["predicted_tumor_rate"] = float(pred.mean())

    slides = np.array([slide_of[p.tile_id] for p in preds])
    per_slide = {int(s): binary_metrics(y[slides == s], pred[slides == s], prob[slides == s])
                 for s in sorted(set(slides.tolist()))}

    tin = np.array([p.extra.get("tokens_in", 0) for p in preds])
    tout = np.array([p.extra.get("tokens_out", 0) for p in preds])
    lat = np.array([p.latency_ms for p in preds])
    n = len(preds)
    tokens = {
        "input_total": int(tin.sum()), "output_total": int(tout.sum()),
        "input_per_image": float(tin.mean()), "output_per_image": float(tout.mean()),
        "total_per_image": float((tin + tout).mean()),
    }
    cost: dict = {"token_price_proxy_usd": float(sum(p.cost_usd for p in preds))}
    if "gpu_minutes" in info and "gpu_rate_usd_per_hour" in info:
        gpu_usd = info["gpu_minutes"] / 60 * info["gpu_rate_usd_per_hour"] * info.get(
            "accelerator_count", 1)
        cost.update({"gpu_usd_estimate": gpu_usd, "gpu_usd_per_1k_images": 1000 * gpu_usd / n})
    speed = {"latency_mean_ms": float(lat.mean()), "latency_p50_ms": float(np.percentile(lat, 50)),
             "latency_p95_ms": float(np.percentile(lat, 95)),
             "latency_max_ms": float(lat.max())}
    if info.get("wall_s"):
        speed.update({"wall_s": info["wall_s"], "images_per_s": n / info["wall_s"],
                      "output_tokens_per_s": float(tout.sum()) / info["wall_s"]})
    report = {"model": info.get("model", preds[0].model), "preds_file": str(preds_file), "run_info": info,
              "overall": overall, "per_slide": per_slide, "tokens": tokens, "cost": cost,
              "speed": speed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))

    o, c = overall, overall["confusion"]
    lo, hi = o["accuracy_95ci"]
    md = [
        f"# Camelyon17 baseline: `{report['model']}`", "",
        (f"{n} tiles (WILDS hospital 2, slides {', '.join(map(str, per_slide))}), "
        f"{o['n_tumor']} tumor / {o['n_normal']} normal. {info.get('setup', 'Zero-shot, prompt v1, temperature 0.')}"), "",
        "## Accuracy", "",
        "| metric | value |", "|---|---|",
        f"| accuracy | {pct(o['accuracy'])} (95% CI {pct(lo)}-{pct(hi)}) |",
        f"| balanced accuracy | {pct(o['balanced_accuracy'])} |",
        f"| sensitivity (tumors caught) | {pct(o['sensitivity'])} |",
        f"| specificity (normals cleared) | {pct(o['specificity'])} |",
        f"| precision | {pct(o['precision'])} |", f"| F1 | {pct(o['f1'])} |",
        f"| AUROC (stated confidence) | {o['auroc']:.3f} |" if o["auroc"] is not None
        else "| AUROC | - |",
        f"| predicted tumor rate | {pct(o['predicted_tumor_rate'])} |",
        f"| parse / API errors | {o['parse_errors']} / {o['api_errors']} |", "",
        f"Confusion: TP {c['tp']}, FN {c['fn']}, FP {c['fp']}, TN {c['tn']}", "",
        "### Per slide", "", "| slide | n | tumor | accuracy | sensitivity | specificity |",
        "|---|---|---|---|---|---|",
        *[f"| {s} | {m['n']} | {m['n_tumor']} | {pct(m['accuracy'])} | {pct(m['sensitivity'])} "
          f"| {pct(m['specificity'])} |" for s, m in per_slide.items()], "",
        "## Cost (tokens)", "", "| metric | value |", "|---|---|",
        f"| input tokens | {tokens['input_total']:,} ({tokens['input_per_image']:.0f}/image) |",
        f"| output tokens | {tokens['output_total']:,} ({tokens['output_per_image']:.0f}/image) |",
    ]
    if "gpu_usd_estimate" in cost:
        md += [(f"| GPU time | {info['gpu_minutes']:.1f} min x {info.get('accelerator_count', 1)} "
               f"{info.get('accelerator', 'GPU')} |"),
               (f"| GPU cost (estimate @ ${info['gpu_rate_usd_per_hour']:.2f}/GPU-h) | "
               f"${cost['gpu_usd_estimate']:.2f} (${cost['gpu_usd_per_1k_images']:.2f} per 1k images) |")]
    md += [f"| token-price proxy | ${cost['token_price_proxy_usd']:.2f} |", "",
           "## Speed", "", "| metric | value |", "|---|---|",
           (f"| latency p50 / p95 / max | {speed['latency_p50_ms']:.0f} / "
           f"{speed['latency_p95_ms']:.0f} / {speed['latency_max_ms']:.0f} ms |")]
    if "wall_s" in speed:
        md += [f"| wall time | {speed['wall_s'] / 60:.1f} min |",
               (f"| throughput | {speed['images_per_s']:.2f} images/s, "
               f"{speed['output_tokens_per_s']:.0f} output tokens/s |")]
    if info.get("deploy_ready_s"):
        md += [f"| deployment spin-up | {info['deploy_ready_s'] / 60:.1f} min |"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    typer.echo("\n".join(md))


if __name__ == "__main__":
    app()
