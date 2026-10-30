"""Launch a Fireworks vision LoRA SFT job (CONTRACTS.md section 5).

With FIREWORKS_API_KEY + FIREWORKS_ACCOUNT_ID set: validates the JSONL, uploads the dataset(s),
creates the job, optionally waits and deploys, and prints the model id for FIREWORKS_FT_MODEL.
Without credentials (or with --dry-run): validates the file and prints the full firectl recipe.

    .venv/bin/python scripts/launch_finetune.py --dataset-file data/sft_pcam_train.jsonl --wait --deploy
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.finetune import (
    DEFAULT_BASE_MODEL,
    VISION_SFT_BASE_MODELS,
    FireworksFT,
    estimate_training_cost,
    manual_recipe,
    validate_sft_jsonl,
)

app = typer.Typer(add_completion=False)


DEFAULT_DATASET_FILE = config.DATA_DIR / "sft_pcam_train.jsonl"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")[:63]


@app.command()
def main(
    base_model: str = typer.Option(DEFAULT_BASE_MODEL, help="Vision-capable, managed-SFT-tunable base"),
    dataset_file: Annotated[Path, typer.Option(help="Training JSONL from prepare_sft.py")] = DEFAULT_DATASET_FILE,
    eval_file: Annotated[Path | None, typer.Option(help="Optional eval JSONL (else Fireworks carves one out)")] = None,
    epochs: int = typer.Option(2),
    lora_rank: int = typer.Option(16, help="1..128 (firectl default 8); 0 would mean full-parameter"),
    learning_rate: float | None = typer.Option(None, help="Leave unset: Fireworks picks per model"),
    name: str | None = typer.Option(None, help="Output model id (default pathfrontier-<ds>-<base>-<ts>)"),
    dataset_id: str | None = typer.Option(None, help="Fireworks dataset id (default from file name)"),
    wait: bool = typer.Option(False, "--wait/--no-wait", help="Block until the job finishes"),
    deploy: bool = typer.Option(False, "--deploy/--no-deploy", help="After --wait, create the deployment"),
    dry_run: bool = typer.Option(False, help="Validate + print the manual recipe even with a key"),
    list_models: bool = typer.Option(False, help="Print the known vision-SFT base models and exit"),
) -> None:
    if list_models:
        for mid, info in VISION_SFT_BASE_MODELS.items():
            typer.echo(f"{mid}\n   {info['params_b']}B  ${info['train_usd_per_m']:.2f}/M train tokens  "
                       f"shape {info['lora_shape']}\n   {info['notes']}")
        raise typer.Exit()

    if base_model not in VISION_SFT_BASE_MODELS:
        typer.echo(f"warning: {base_model} is not in the researched vision-SFT list; make sure "
                   "`firectl model get -a fireworks <id>` says Tunable: true and the model page "
                   "says 'Supporting image input: Supported'.")

    rep = validate_sft_jsonl(dataset_file)
    typer.echo(rep.summary())
    if not rep.ok:
        typer.echo("dataset file is invalid; fix it first (scripts/prepare_sft.py)")
        raise typer.Exit(code=1)
    if eval_file:
        erep = validate_sft_jsonl(eval_file)
        typer.echo(erep.summary())
        if not erep.ok:
            raise typer.Exit(code=1)
    cost = estimate_training_cost(rep.total_tokens, epochs, base_model)
    typer.echo(f"estimated training cost: ${cost:.2f} ({rep.total_tokens:,} tokens x {epochs} epochs)")

    ds_tag = dataset_file.stem.replace("sft_", "").replace("_train", "")
    name = name or f"pathfrontier-{_slug(ds_tag)}-{_slug(base_model.rsplit('/', 1)[-1])}-{int(time.time())}"
    dataset_id = dataset_id or _slug(dataset_file.stem)

    ft = FireworksFT()
    if dry_run or not ft.available:
        if not ft.available:
            typer.echo("\nNo FIREWORKS_API_KEY / FIREWORKS_ACCOUNT_ID -> manual recipe:\n")
        typer.echo(manual_recipe(dataset_file, base_model, name, lora_rank, epochs, learning_rate,
                                 eval_file, dataset_id))
        raise typer.Exit()

    typer.echo(f"\n[1/4] uploading dataset {dataset_id} ...   ({ft.cli_create_dataset(dataset_file, dataset_id)})")
    ds = ft.create_dataset(dataset_file, dataset_id)
    typer.echo(f"      {ds.get('name')} state={ds.get('state')}")
    eval_id = None
    if eval_file:
        eval_id = f"{dataset_id}-eval"
        eds = ft.create_dataset(eval_file, eval_id)
        typer.echo(f"      {eds.get('name')} state={eds.get('state')}")

    typer.echo("[2/4] creating SFT job ...")
    job = ft.create_job(base_model, dataset_id, lora_rank=lora_rank, epochs=epochs, lr=learning_rate,
                        name=name, eval_dataset_id=eval_id)
    typer.echo(f"      job {job['jobId']} state={job.get('state')} output={job.get('outputModel')}")
    typer.echo(f"      manual: {job.cli_hint}")
    typer.echo(f"      poll:   {ft.cli_get_job(job['jobId'])}")

    if not wait:
        typer.echo("\nnot waiting. Later:\n"
                   f"  .venv/bin/python scripts/deploy_finetuned.py --model {job['outputModel']} --wait")
        raise typer.Exit()

    typer.echo("[3/4] waiting for the job (poll every 30 s) ...")
    done = ft.wait(job["jobId"], on_update=lambda j: typer.echo(
        f"      state={j.get('state')} progress={(j.get('jobProgress') or {}).get('percent', '?')}%"))
    typer.echo(f"      finished: {done.get('state')}  estimatedCost={done.get('estimatedCost')}")
    model_id = done.get("outputModel") or job["outputModel"]

    if not deploy:
        typer.echo(f"\nfine-tuned model: {model_id}\n"
                   f"deploy: .venv/bin/python scripts/deploy_finetuned.py --model {model_id} --wait")
        raise typer.Exit()

    typer.echo("[4/4] creating on-demand deployment (live-merged LoRA) ...")
    dep = ft.deploy(model_id, wait=True)
    typer.echo(f"      {dep['name']} state={dep.get('state')}")
    typer.echo(f"\nexport FIREWORKS_FT_MODEL={dep['inference_model']}\n"
               ".venv/bin/python scripts/run_vlm_ft.py --dataset pcam --split test --limit 500\n"
               f"when done: firectl deployment delete {dep['name'].rsplit('/', 1)[-1]}")


if __name__ == "__main__":
    app()
