"""THE chart: cost per slide (log x) vs sensitivity, with the cascade sweep as a
curve (Pareto front highlighted) and the single models as labelled points; second
panel: pathologist minutes per slide vs sensitivity.

    uv run python scripts/plot_frontier.py --sweep results/sweep_vlm_ft.csv \
        --sweep results/sweep_pathologist.csv --points results/preds_*_test.jsonl \
        --out results/frontier.png --metric sensitivity_system
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated

import matplotlib
import numpy as np
import pandas as pd
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.cascade import pareto_mask, resolve_compute_metrics, scalar_metrics
from pathfrontier.schema import load_predictions

app = typer.Typer(add_completion=False)

# dataviz reference palette: one accent for the cascade, red only for the baseline,
# neutral grey for the other single points, text in ink tokens.
PALETTE = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984",
    "grid": "#e6e5e1", "accent": "#2a78d6", "accent_dark": "#184f95",
    "accent_light": "#9ec5f4", "red": "#e34948", "grey": "#6b6a66",
}
SWEEP_LABELS = {"vlm_ft": "cascade: probe + fine-tuned VLM", "pathologist": "cascade: probe + pathologist",
                "none": "cascade: probe only"}
MIN_COST = 1e-4   # floor so $0 points can sit on a log axis


def lookup_metric(metrics: dict, name: str) -> float:
    """Tolerant key lookup: 'sensitivity_at_spec_0.95' also matches 'sensitivity_at_spec0.95',
    'sensitivity_at_spec_0.9' or a nested {'sensitivity_at_spec': {'0.95': ...}}."""
    if name in metrics:
        return float(metrics[name])
    m = re.fullmatch(r"(.*?)_?(\d*\.\d+)", name)
    if m:
        base, val = m.group(1), float(m.group(2))
        for k, v in metrics.items():
            km = re.fullmatch(re.escape(base) + r"_?(\d*\.\d+)", k)
            if km and abs(float(km.group(1)) - val) < 1e-9:
                return float(v)
        nested = metrics.get(base)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if abs(float(k) - val) < 1e-9:
                    return float(v)
    if name == "sensitivity_system" and "sensitivity" in metrics:
        return float(metrics["sensitivity"])
    raise KeyError(f"metric {name!r} not in {sorted(metrics)}")


def sweep_name(path: Path, df: pd.DataFrame) -> str:
    fb = str(df["fallback"].iloc[0]) if "fallback" in df.columns else path.stem.replace("sweep_", "")
    return SWEEP_LABELS.get(fb, f"cascade: {fb}")


def load_sweeps(paths: list[Path], metric: str) -> list[dict]:
    out = []
    for p in paths:
        df = pd.read_csv(p)
        df["_y"] = [lookup_metric(r, metric) for r in df.to_dict("records")]
        df["_pareto"] = pareto_mask(df, "cost_per_slide_usd", "_y")
        out.append({"file": str(p), "name": sweep_name(p, df),
                    "fallback": str(df["fallback"].iloc[0]) if "fallback" in df.columns else None,
                    "df": df})
    return out


def load_points(paths: list[Path], metric: str, dataset: str, tiles_per_slide: int) -> list[dict]:
    compute_metrics = resolve_compute_metrics()
    out = []
    for p in paths:
        preds = load_predictions(p)
        if not preds:
            continue
        ds = preds[0].dataset or dataset
        m = scalar_metrics(compute_metrics(preds, ds, config.FIXED_SPECIFICITY_POINTS, tiles_per_slide))
        out.append({"file": str(p), "model": preds[0].model, "stage": preds[0].stage,
                    "cost_per_slide_usd": float(m["cost_per_slide_usd"]),
                    "pathologist_minutes_per_slide": float(m.get("pathologist_minutes_per_slide", 0.0)),
                    "y": lookup_metric(m, metric), "metrics": m})
    return out


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["ink2"], labelsize=9, length=0)


def _draw_curves(ax: plt.Axes, sweeps: list[dict], xcol: str, log_x: bool, annotate_pareto: bool) -> None:
    """Faint dots for every operating point; a line + labels only on the Pareto front for *this* x."""
    styles = [{"ls": "-"}, {"ls": ":"}, {"ls": "--"}]
    for i, s in enumerate(sweeps):
        df = s["df"]
        x = df[xcol].clip(lower=MIN_COST) if log_x else df[xcol]
        st = styles[i % len(styles)]
        ax.scatter(x, df["_y"], s=12, color=PALETTE["accent_light"], alpha=0.8, lw=0, zorder=3)
        front = df[pareto_mask(df, xcol, "_y")]
        xf = (front[xcol].clip(lower=MIN_COST) if log_x else front[xcol]).values
        fo = np.argsort(xf, kind="stable")
        ax.plot(xf[fo], front["_y"].values[fo], lw=2.0, color=PALETTE["accent_dark"], ls=st["ls"],
                zorder=4, label=f"{s['name']} (Pareto front)")
        ax.scatter(xf, front["_y"], s=34, color=PALETTE["accent_dark"], edgecolor=PALETTE["surface"],
                   lw=1.2, zorder=5)
        if annotate_pareto:
            for _, r in front.iloc[fo].iterrows():
                if r["low"] == r["high"]:        # probe-alone point: the probe marker labels it
                    continue
                xv = max(r[xcol], MIN_COST) if log_x else r[xcol]
                # the front rises to the upper-right, so lower-right is always clear of the line
                ax.annotate(f"{r['low']:g}/{r['high']:g}", (xv, r["_y"]), xytext=(5, -10),
                            textcoords="offset points", fontsize=6.5, color=PALETTE["ink2"], zorder=6)


def _draw_points(ax: plt.Axes, points: list[dict], xkey: str, log_x: bool) -> None:
    for pt in points:
        is_base = pt["stage"] == "baseline"
        col = PALETTE["red"] if is_base else PALETTE["grey"]
        xv = max(pt[xkey], MIN_COST) if log_x else pt[xkey]
        ax.scatter([xv], [pt["y"]], s=60, marker="D" if is_base else "o", color=col,
                   edgecolor=PALETTE["surface"], lw=1.2, zorder=6)
        ax.annotate(f"{pt['model']}\n({pt['stage']})", (xv, pt["y"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=8, color=PALETTE["ink"], zorder=7, va="top")


def plot(
    sweeps: list[dict], points: list[dict], out: Path, metric: str,
    title: str = "Cost vs sensitivity frontier — PCam test",
    target: float = config.TARGET_SENSITIVITY,
) -> Path:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1600 / 150, 900 / 150), dpi=150,
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor(PALETTE["surface"])
    ylabel = {"sensitivity_system": "Sensitivity (system: automated + human-read tiles)",
              }.get(metric, metric.replace("_", " "))

    _style(ax1)
    ax1.set_xscale("log")
    _draw_curves(ax1, sweeps, "cost_per_slide_usd", log_x=True, annotate_pareto=True)
    _draw_points(ax1, points, "cost_per_slide_usd", log_x=True)
    ax1.axhline(target, color=PALETTE["muted"], ls="--", lw=1.0, zorder=1)
    ax1.text(ax1.get_xlim()[1], target + 0.004, f"target sensitivity {target:.2f}", fontsize=8,
             color=PALETTE["ink2"], va="bottom", ha="right")
    ax1.set_xlabel(f"Cost per slide, USD (log scale; {config.TILES_PER_SLIDE:,} tiles)",
                   color=PALETTE["ink2"])
    ax1.set_ylabel(ylabel, color=PALETTE["ink2"])
    ax1.set_title("Dollars", loc="left", fontsize=11, color=PALETTE["ink"])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v:g}" if v >= 0.01 else f"${v:.4f}".rstrip("0")))

    _style(ax2)
    _draw_curves(ax2, sweeps, "pathologist_minutes_per_slide", log_x=False, annotate_pareto=False)
    _draw_points(ax2, points, "pathologist_minutes_per_slide", log_x=False)
    ax2.axhline(target, color=PALETTE["muted"], ls="--", lw=1.0, zorder=1)
    full_read = config.TILES_PER_SLIDE * config.PATHOLOGIST_SEC_PER_TILE / 60
    ax2.set_xlabel(f"Pathologist minutes per slide (full read: {full_read:,.0f} min)",
                   color=PALETTE["ink2"])
    ax2.set_title("Human time", loc="left", fontsize=11, color=PALETTE["ink"])
    ax2.set_xlim(left=-5)

    # Shared y-limits so both panels read against the same scale.
    ys = [y for s in sweeps for y in s["df"]["_y"].dropna()] + [p["y"] for p in points] + [target]
    lo = max(0.0, min(ys) - 0.03)
    for ax in (ax1, ax2):
        ax.set_ylim(lo, 1.005)

    handles, labels = ax1.get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", ls="", ms=4, color=PALETTE["accent_light"],
                           label="other cascade operating points"),
                plt.Line2D([], [], marker="D", ls="", color=PALETTE["red"], label="baseline VLM"),
                plt.Line2D([], [], marker="o", ls="", color=PALETTE["grey"], label="single model (probe / fine-tuned VLM)"),
                plt.Line2D([], [], ls="--", color=PALETTE["muted"], label=f"target sensitivity {target:.2f}")]
    labels += [h.get_label() for h in handles[len(labels):]]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.01, 0.93, "Cascade points are labelled low/high thresholds on p_tumor; "
             "pathologist-routed tiles count as correct (a human reads them).",
             fontsize=8.5, color=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.08, 0.99, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def write_points_json(sweeps: list[dict], points: list[dict], out: Path, metric: str) -> Path:
    keep = ["low", "high", "fallback", "fraction_routed", "n_routed", "cost_per_slide_usd",
            "pathologist_minutes_per_slide", "coverage", "sensitivity_system", "sensitivity",
            "specificity", "auroc", "latency_p50_ms", "cost_per_1k_usd"]
    payload = {
        "metric": metric, "target_sensitivity": config.TARGET_SENSITIVITY,
        "tiles_per_slide": config.TILES_PER_SLIDE,
        "sweeps": [{
            "name": s["name"], "file": s["file"], "fallback": s["fallback"],
            "points": [{**{k: r[k] for k in keep if k in r}, "y": r["_y"], "pareto": bool(r["_pareto"])}
                       for r in s["df"].to_dict("records")],
        } for s in sweeps],
        "points": [{k: v for k, v in p.items()} for p in points],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return out


@app.command()
def main(
    sweep: Annotated[list[Path], typer.Option(help="results/sweep_*.csv (repeatable)")] = (),
    points: Annotated[list[Path], typer.Option(help="results/preds_*.jsonl single-model points (repeatable)")] = (),
    out: Annotated[Path, typer.Option()] = config.RESULTS_DIR / "frontier.png",
    metric: Annotated[str, typer.Option(help="sensitivity_system | sensitivity_at_spec_0.95")] = "sensitivity_system",
    dataset: Annotated[str, typer.Option()] = "pcam",
    tiles_per_slide: Annotated[int, typer.Option()] = config.TILES_PER_SLIDE,
    title: Annotated[str, typer.Option()] = "Cost vs sensitivity frontier — PCam test",
) -> None:
    sweep, points = list(sweep or []), list(points or [])
    if not sweep and not points:
        raise typer.BadParameter("give at least one --sweep or --points")
    sweeps = load_sweeps(sweep, metric)
    pts = load_points(points, metric, dataset, tiles_per_slide)
    png = plot(sweeps, pts, out, metric, title=title)
    js = write_points_json(sweeps, pts, out.with_suffix(".json"), metric)
    typer.echo(f"wrote {png} and {js}")
    for p in pts:
        typer.echo(f"  {p['model']:30s} {p['stage']:9s} cost/slide ${p['cost_per_slide_usd']:.4f}  {metric}={p['y']:.3f}")
    for s in sweeps:
        front = s["df"][s["df"]["_pareto"]]
        typer.echo(f"  {s['name']}: {len(s['df'])} points, {len(front)} on the Pareto front")


if __name__ == "__main__":
    app()
