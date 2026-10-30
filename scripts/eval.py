"""Evaluate prediction files and build the leaderboard.

    uv run python scripts/eval.py                      # all results/preds_*.jsonl
    uv run python scripts/eval.py results/preds_probe_phikon-v2_test.jsonl --dataset pcam

Writes results/metrics_<model>.json per file and results/leaderboard.md, prints the table.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier.config import RESULTS_DIR, TILES_PER_SLIDE
from pathfrontier.metrics import compute_metrics
from pathfrontier.schema import load_predictions, save_json

app = typer.Typer(add_completion=False)

COLUMNS = [
    ("model", "model"), ("stage", "stage"), ("n", "n"), ("AUROC", "auroc"),
    ("bal. acc", "balanced_accuracy"), ("sens@95%spec", "sens95"), ("sens@99%spec", "sens99"),
    ("p50 ms", "latency_p50_ms"), ("p95 ms", "latency_p95_ms"), ("$/1k tiles", "cost_per_1k_usd"),
    ("$/slide", "cost_per_slide_usd"), ("coverage", "coverage"),
]


def _fmt(key: str, v) -> str:
    if v is None:
        return "-"
    if key in ("model", "stage"):
        return str(v)
    if key == "n":
        return str(int(v))
    if key in ("latency_p50_ms", "latency_p95_ms"):
        return f"{v:.1f}"
    if key in ("cost_per_1k_usd", "cost_per_slide_usd"):
        return f"{v:.4f}" if v < 1 else f"{v:.2f}"
    return f"{v:.3f}"


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def leaderboard_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(k, r.get(k)) for _, k in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def evaluate_file(path: Path, dataset: str | None, tiles_per_slide: int,
                  results_dir: Path) -> dict:
    preds = load_predictions(path)
    if not preds:
        raise typer.BadParameter(f"{path} is empty")
    ds = dataset or preds[0].dataset
    m = compute_metrics(preds, ds, tiles_per_slide=tiles_per_slide)
    m["source"] = str(path)
    out = results_dir / f"metrics_{safe_name(m['model'])}.json"
    save_json(m, out)
    row = {k: m.get(k) for _, k in COLUMNS}
    row["sens95"] = m["sensitivity_at_spec"].get("0.95")
    row["sens99"] = m["sensitivity_at_spec"].get("0.99")
    row["_metrics_path"] = str(out)
    return row


@app.command()
def main(
    files: Annotated[list[Path] | None, typer.Argument(
        help="preds_*.jsonl files (default: results/preds_*.jsonl)")] = None,
    dataset: Annotated[str | None, typer.Option(
        "--dataset", "-d", help="override dataset (pcam|crc) if the file lacks it")] = None,
    tiles_per_slide: Annotated[int, typer.Option(help="tiles per slide for $/slide")] = TILES_PER_SLIDE,
    results_dir: Annotated[Path, typer.Option(
        help="where metrics_*.json and leaderboard.md go")] = RESULTS_DIR,
    leaderboard: Annotated[Path | None, typer.Option(
        help="leaderboard path (default results_dir/leaderboard.md)")] = None,
) -> None:
    files = list(files or sorted(results_dir.glob("preds_*.jsonl")))
    if not files:
        typer.echo(f"no prediction files found in {results_dir}", err=True)
        raise typer.Exit(1)
    rows = []
    for f in files:
        try:
            row = evaluate_file(f, dataset, tiles_per_slide, results_dir)
        except (ValueError, ValidationError, OSError, typer.BadParameter) as e:
            typer.echo(f"[eval] skipping {f}: {e}", err=True)
            continue
        rows.append(row)
        typer.echo(f"[eval] {f} -> {row['_metrics_path']}")
    rows.sort(key=lambda r: (r.get("auroc") is None, -(r.get("auroc") or 0.0)))
    table = leaderboard_markdown(rows)
    out = leaderboard or results_dir / "leaderboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Leaderboard\n\n" + table)
    typer.echo("\n" + table)
    typer.echo(f"[eval] wrote {out}")


if __name__ == "__main__":
    app()
