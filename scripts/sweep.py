"""Sweep cascade thresholds and write results/sweep_<fallback>.csv.

    uv run python scripts/sweep.py --probe-preds results/preds_probe_phikon-v2_test.jsonl \
        --vlm-preds results/preds_<ft-model>_test.jsonl --dataset pcam --fallback vlm_ft
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.cascade import pareto_front, sweep
from pathfrontier.schema import load_predictions

app = typer.Typer(add_completion=False)

PRINT_COLS = ["low", "high", "fraction_routed", "cost_per_slide_usd", "sensitivity_system",
              "specificity", "coverage", "pathologist_minutes_per_slide", "latency_p50_ms"]


def run_sweep(
    probe_preds: Path,
    vlm_preds: Path | None,
    dataset: str,
    fallback: str,
    tiles_per_slide: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    probe = load_predictions(probe_preds)
    vlm = {p.tile_id: p for p in load_predictions(vlm_preds)} if vlm_preds else None
    if fallback == "vlm_ft" and not vlm:
        typer.echo("warning: --fallback vlm_ft but no --vlm-preds; band tiles go to the pathologist",
                   err=True)
    df = sweep(probe, vlm, dataset, grid=None, fallback=fallback, tiles_per_slide=tiles_per_slide)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sweep_{fallback}.csv"
    df.to_csv(out, index=False)
    return df, out


@app.command()
def main(
    probe_preds: Annotated[Path, typer.Option(exists=True, help="probe preds_*.jsonl")],
    vlm_preds: Annotated[Path | None, typer.Option(exists=True, help="fine-tuned VLM preds_*.jsonl")] = None,
    dataset: Annotated[str, typer.Option()] = "pcam",
    fallback: Annotated[str, typer.Option(help="vlm_ft | pathologist | none")] = "vlm_ft",
    tiles_per_slide: Annotated[int, typer.Option()] = config.TILES_PER_SLIDE,
    out_dir: Annotated[Path, typer.Option(help="directory for sweep_<fallback>.csv")] = config.RESULTS_DIR,
) -> None:
    df, out = run_sweep(probe_preds, vlm_preds, dataset, fallback, tiles_per_slide, out_dir)
    typer.echo(f"wrote {out} ({len(df)} operating points)")
    front = pareto_front(df)
    cols = [c for c in PRINT_COLS if c in front.columns]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        typer.echo("Pareto front (min cost/slide, max sensitivity_system):")
        typer.echo(front[cols].to_string(index=False))


if __name__ == "__main__":
    app()
