"""Deploy (or tear down) a fine-tuned Fireworks LoRA model on an on-demand deployment.

Fireworks serves trained LoRA models ONLY on dedicated deployments (live-merged into the base
model); there is no serverless LoRA serving. The inference id is accounts/<acct>/models/<name>.

    .venv/bin/python scripts/deploy_finetuned.py --model accounts/<acct>/models/<name> --wait
    .venv/bin/python scripts/deploy_finetuned.py --delete <deployment-id>
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier.finetune import FireworksFT

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option("", help="Fine-tuned model id (accounts/<acct>/models/<name> or <name>)"),
    deployment_id: str | None = typer.Option(None, help="Default: derived from the model name"),
    shape: str = typer.Option("default", help="Deployment shape ('default' lets Fireworks pick)"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    min_replicas: int = typer.Option(0, help="0 enables scale-to-zero (>= 5 min window)"),
    scale_to_zero_window: str = typer.Option("600s"),
    delete: str | None = typer.Option(None, help="Delete this deployment id instead of creating one"),
) -> None:
    ft = FireworksFT()
    if delete:
        if not ft.available:
            typer.echo(f"no credentials; run manually:\n  firectl deployment delete {delete}")
            raise typer.Exit()
        ft.delete_deployment(delete)
        typer.echo(f"deleted deployment {delete}")
        raise typer.Exit()
    if not model:
        raise typer.BadParameter("--model is required (or --delete <deployment-id>)")
    hint = ft.cli_deploy(model if model.startswith("accounts/") else f"accounts/<ACCOUNT_ID>/models/{model}",
                         deployment_id, shape)
    if not ft.available:
        typer.echo("No FIREWORKS_API_KEY / FIREWORKS_ACCOUNT_ID. Manual:\n  " + hint)
        raise typer.Exit()
    typer.echo(f"creating deployment ...  (manual: {hint})")
    dep = ft.deploy(model, deployment_id=deployment_id, shape=shape, wait=wait,
                    min_replicas=min_replicas, scale_to_zero_window=scale_to_zero_window)
    typer.echo(f"{dep['name']} state={dep.get('state')}")
    typer.echo(f"\nexport FIREWORKS_FT_MODEL={dep['inference_model']}")
    typer.echo(f"tear down: firectl deployment delete {dep['name'].rsplit('/', 1)[-1]}")


if __name__ == "__main__":
    app()
