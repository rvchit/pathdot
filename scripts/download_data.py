"""Download HF parquet shards into data/<dataset>/<split>.parquet.

    uv run python scripts/download_data.py --dataset pcam --splits valid,test
    uv run python scripts/download_data.py --dataset crc --splits test
    uv run python scripts/download_data.py --dataset all          # valid+test for both
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier.data import download, split_path

app = typer.Typer(add_completion=False)


def _parse_splits(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        out.extend(s.strip() for s in v.split(",") if s.strip())
    return out


@app.command()
def main(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="pcam | crc | all")] = "pcam",
    splits: Annotated[list[str] | None, typer.Option(
        "--splits", "-s", help="comma-separated or repeated: train, valid, test")] = None,
    force: Annotated[bool, typer.Option(help="re-download even if the parquet exists")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = ["pcam", "crc"] if dataset == "all" else [dataset]
    split_list = _parse_splits(splits or ["valid,test"])
    for ds in datasets:
        download(ds, splits=split_list, force=force)
        for s in split_list:
            p = split_path(ds, s)
            size_mb = p.stat().st_size / 1e6 if p.exists() else 0
            typer.echo(f"{ds}/{s}: {p} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    app()
