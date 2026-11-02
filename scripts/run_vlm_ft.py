"""Run a fine-tuned VLM deployment over a split; stage="vlm_ft".

Model defaults to env FIREWORKS_FT_MODEL (the deployable id printed by launch_finetune.py).
The predictions file is named after the fine-tuned model id: results/preds_<ft id>_<split>.jsonl.

    FIREWORKS_FT_MODEL=accounts/<acct>/models/kimi-k3-pcam-v1 \
        .venv/bin/python scripts/run_vlm_ft.py --dataset pcam --split test --limit 500
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.vlm_client import run_split

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(os.getenv("FIREWORKS_FT_MODEL", ""),
                              help="Fine-tuned model/deployment id (env FIREWORKS_FT_MODEL)"),
    dataset: str = typer.Option("pcam"),
    split: str = typer.Option("test"),
    limit: int | None = typer.Option(500, help="Number of tiles; 0 = all"),
    concurrency: int = typer.Option(8),
    prompt_version: str = typer.Option("v1"),
    provider: str = typer.Option("openai"),
    seed: int = typer.Option(0),
    dry_run: bool = typer.Option(False),
    results_dir: str = typer.Option("", help="default: results/"),
    no_thinking: bool = typer.Option(False, help="reasoning_effort=none for hybrid-reasoning models"),
    ids_file: str = typer.Option("", help="CSV with a tile_id column: classify only these tiles"),
    out_name: str = typer.Option("", help="name for the preds file (default: the model id)"),
    tta: int = typer.Option(1, help="average over 1-4 rotations of each tile"),
) -> None:
    if not model:
        if dry_run:
            model = "accounts/example/models/kimi-k3-pcam-ft-dryrun"
        else:
            raise typer.BadParameter("pass --model or set FIREWORKS_FT_MODEL")
    out, summary = run_split(
        model=model, dataset=dataset, split=split, limit=(limit or None), concurrency=concurrency,
        prompt_version=prompt_version, provider=provider, stage="vlm_ft", dry_run=dry_run,
        seed=seed, results_dir=Path(results_dir) if results_dir else config.RESULTS_DIR,
        out_name=out_name or model, no_thinking=no_thinking,
        ids_file=Path(ids_file) if ids_file else None, tta=tta,
    )
    typer.echo(f"stage=vlm_ft model={model}")
    typer.echo(f"wrote {out}")
    typer.echo(
        f"n={summary['n']}  accuracy={summary['accuracy']:.4f}  "
        f"mean_latency={summary['mean_latency_ms']:.0f} ms  "
        f"total_cost=${summary['total_cost_usd']:.4f}  parse_errors={summary['parse_errors']}"
    )


if __name__ == "__main__":
    app()
