"""Run the pathfrontier inference service (docs/CONTRACTS.md section 8).

    uv run python scripts/serve.py --fake                 # no model download, demo scorer
    uv run python scripts/serve.py --encoder phikon-v2 --probe results/probe_phikon-v2_pcam.joblib
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(add_completion=False)


@app.command()
def main(
    port: Annotated[int, typer.Option(help="port to listen on")] = 8008,
    host: Annotated[str, typer.Option(help="bind address")] = "0.0.0.0",
    fake: Annotated[bool, typer.Option("--fake", help="fake encoder + random probe (no downloads)")] = False,
    encoder: Annotated[str | None, typer.Option(help="encoder name (default phikon-v2)")] = None,
    probe: Annotated[Path | None, typer.Option(help="probe joblib (default results/probe_<enc>_pcam.joblib)")] = None,
    tiles_dir: Annotated[Path | None, typer.Option(help="DZI pyramids dir (default tiles/)")] = None,
    vlm_model: Annotated[str | None, typer.Option(help="Fireworks model id for the uncertain band")] = None,
    dataset: Annotated[str | None, typer.Option(help="label space: pcam (default) or crc")] = None,
    low: Annotated[float | None, typer.Option(help="auto-clear threshold (default 0.05)")] = None,
    high: Annotated[float | None, typer.Option(help="auto-flag threshold (default 0.95)")] = None,
    reload: Annotated[bool, typer.Option(help="uvicorn auto-reload (dev)")] = False,
):
    """Thin wrapper: translates flags into PATHFRONTIER_* env vars and starts uvicorn."""
    if fake:
        os.environ["PATHFRONTIER_FAKE"] = "1"
    for key, val in (
        ("PATHFRONTIER_ENCODER", encoder),
        ("PATHFRONTIER_PROBE", probe),
        ("PATHFRONTIER_TILES_DIR", tiles_dir),
        ("PATHFRONTIER_VLM_MODEL", vlm_model),
        ("PATHFRONTIER_DATASET", dataset),
        ("PATHFRONTIER_LOW", low),
        ("PATHFRONTIER_HIGH", high),
    ):
        if val is not None:
            os.environ[key] = str(val)
    from pathfrontier.serve import main as run

    typer.echo(f"pathfrontier serve on http://{host}:{port} "
               f"(fake={fake}, encoder={os.getenv('PATHFRONTIER_ENCODER', 'phikon-v2')})")
    run(host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
