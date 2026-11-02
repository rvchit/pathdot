"""Run a zero-shot VLM baseline over a dataset split and write preds_<model>_<split>.jsonl.

    .venv/bin/python scripts/run_baseline.py --model accounts/fireworks/models/kimi-k3 \
        --dataset pcam --split test --limit 500
    .venv/bin/python scripts/run_baseline.py --dry-run --limit 8      # no API key needed
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.vlm_client import DEFAULT_BASELINE_MODEL, run_split

app = typer.Typer(add_completion=False)


def print_summary(out: Path, summary: dict, model: str, stage: str) -> None:
    typer.echo(f"stage={stage} model={model}")
    typer.echo(f"wrote {out}")
    typer.echo(
        f"n={summary['n']}  accuracy={summary['accuracy']:.4f}  "
        f"mean_latency={summary['mean_latency_ms']:.0f} ms  "
        f"total_cost=${summary['total_cost_usd']:.4f}  parse_errors={summary['parse_errors']}"
    )


@app.command()
def main(
    model: str = typer.Option(DEFAULT_BASELINE_MODEL, help="Model id on the provider"),
    dataset: str = typer.Option("pcam", help="pcam | crc | camelyon17"),
    split: str = typer.Option("test", help="train | valid | test"),
    limit: int | None = typer.Option(500, help="Number of tiles (seeded shuffle); 0 = all"),
    concurrency: int = typer.Option(8),
    prompt_version: str = typer.Option("v1"),
    provider: str = typer.Option("openai", help="openai (Fireworks/vLLM) | anthropic"),
    seed: int = typer.Option(0),
    dry_run: bool = typer.Option(False, help="Fake client with canned answers; no key needed"),
    results_dir: str = typer.Option("", help="default: results/"),
    no_thinking: bool = typer.Option(False, help="reasoning_effort=none for hybrid-reasoning models"),
    ids_file: str = typer.Option("", help="CSV with a tile_id column: classify only these tiles"),
    out_name: str = typer.Option("", help="name for the preds file (default: the model id)"),
) -> None:
    out, summary = run_split(
        model=model, dataset=dataset, split=split, limit=(limit or None), concurrency=concurrency,
        prompt_version=prompt_version, provider=provider, stage="baseline", dry_run=dry_run,
        seed=seed, results_dir=Path(results_dir) if results_dir else config.RESULTS_DIR,
        out_name=out_name or None, no_thinking=no_thinking,
        ids_file=Path(ids_file) if ids_file else None,
    )
    print_summary(out, summary, model, "baseline")


if __name__ == "__main__":
    app()
