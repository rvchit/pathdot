"""Benchmark every encoder that loads on this machine and print a throughput/cost table.

    uv run python scripts/bench_encoders.py --n 256
    uv run python scripts/bench_encoders.py --only phikon-v2,resnet50

Gated encoders that fail to download (no HF_TOKEN / no access) are skipped with the reason.
The $/1k column applies GPU_USD_PER_HOUR (H100, $2.5/h) to the throughput measured HERE, so on a Mac
it is an upper bound; use --h100-speedup to scale the measured tiles/s before costing.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.encoders import ENCODERS, GatedModelError, benchmark, get_encoder

GPU_USD_PER_HOUR = 2.5
app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(256, help="tiles per timed run"),
    batch_size: int = typer.Option(64),
    only: str | None = typer.Option(None, help="comma-separated encoder names"),
    include_fake: bool = typer.Option(False),
    gpu_usd_per_hour: float = typer.Option(GPU_USD_PER_HOUR),
    h100_speedup: float = typer.Option(1.0, help="multiply measured tiles/s by this for the $ column"),
    device: str | None = typer.Option(None),
    verbose: bool = typer.Option(False, help="print full tracebacks for failed encoders"),
) -> None:
    names = [s.strip() for s in only.split(",")] if only else [
        k for k in ENCODERS if include_fake or k != "fake"]
    rows: list[tuple] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        typer.echo(f"[{name}] loading ...", err=True)
        try:
            enc = get_encoder(name, device=device)
            tps = benchmark(enc, n=n, batch_size=batch_size)
        except GatedModelError as e:
            skipped.append((name, str(e).splitlines()[0]))
            continue
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"))
            if verbose:
                traceback.print_exc()
            continue
        cost_1k = 1000.0 / (tps * h100_speedup) * gpu_usd_per_hour / 3600.0
        rows.append((name, enc.dim, enc.device, tps, cost_1k))
        typer.echo(f"[{name}] {tps:.1f} tiles/s", err=True)
        del enc
        _free(device or config.device())

    hdr = f"{'encoder':<13}{'dim':>6}{'device':>8}{'tiles/s':>10}{'$/1k tiles':>13}"
    typer.echo(hdr)
    typer.echo("-" * len(hdr))
    for name, dim, dev, tps, cost in rows:
        typer.echo(f"{name:<13}{dim:>6}{dev:>8}{tps:>10.1f}{cost:>13.5f}")
    typer.echo(f"($/1k at ${gpu_usd_per_hour}/h GPU, measured tiles/s x{h100_speedup}; "
               f"n={n}, batch={batch_size})")
    for name, why in skipped:
        typer.echo(f"skipped {name}: {why}")


def _free(device: str) -> None:
    import gc

    gc.collect()
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001, S110 - sync/cache are best-effort
        pass


if __name__ == "__main__":
    app()
