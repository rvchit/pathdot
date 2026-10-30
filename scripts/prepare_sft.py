"""Prepare the Fireworks vision-SFT JSONL (CONTRACTS.md section 5).

Writes data/sft_<dataset>_train.jsonl (+ a small disjoint eval file) from a balanced, seeded
sample of real tiles, or from make_synthetic() with --synthetic (offline smoke test).

    .venv/bin/python scripts/prepare_sft.py --dataset pcam --n 2000
    .venv/bin/python scripts/prepare_sft.py --dataset pcam --n 20 --synthetic
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config, schema
from pathfrontier.finetune import (
    DEFAULT_BASE_MODEL,
    estimate_training_cost,
    validate_sft_jsonl,
    write_sft_jsonl,
)

app = typer.Typer(add_completion=False)


def _default_split(dataset: str) -> str:
    # pcam: the official valid split (train is 262k rows; valid is 32k, plenty for 2k samples and
    # keeps test untouched). crc: "train" (NCT_CRC_HE_100K); test (CRC_VAL_HE_7K) stays held out.
    return "valid" if dataset == "pcam" else "train"


@app.command()
def main(
    dataset: str = typer.Option("pcam", help="pcam | crc"),
    split: str | None = typer.Option(None, help="Source split (default: valid for pcam, train for crc)"),
    n: int = typer.Option(2000, help="Training examples"),
    eval_n: int = typer.Option(200, help="Held-out eval examples (disjoint from train)"),
    balanced: bool = typer.Option(True, "--balanced/--no-balanced", help="Equal class counts"),
    image_size: int = typer.Option(448, help="Square JPEG side sent to the model"),
    out: Annotated[Path | None, typer.Option(help="Train JSONL path (default data/sft_<dataset>_train.jsonl)")] = None,
    seed: int = typer.Option(0),
    synthetic: bool = typer.Option(False, help="Use make_synthetic() instead of real data (offline)"),
    base_model: str = typer.Option(DEFAULT_BASE_MODEL, help="Only used for the cost estimate"),
    epochs: int = typer.Option(2, help="Only used for the cost estimate"),
) -> None:
    if dataset not in schema.CLASSES:
        raise typer.BadParameter(f"dataset must be one of {list(schema.CLASSES)}")
    split = split or _default_split(dataset)
    total = n + eval_n

    from pathfrontier import data  # heavy import (pandas/pyarrow); keep it off --help

    if synthetic:
        typer.echo(f"synthetic: generating {total} {dataset} tiles (seed={seed})")
        records = data.make_synthetic(dataset, n=total, seed=seed, split=split)  # type: ignore[arg-type]
    else:
        typer.echo(f"loading {total} {dataset}/{split} tiles (balanced={balanced}, seed={seed})")
        records = data.load_split(dataset, split, limit=total, seed=seed, balanced=balanced)  # type: ignore[arg-type]

    # Stratified split so BOTH files keep the class balance of the sample.
    rng = random.Random(seed)
    by_label: dict[int, list] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    train, evals = [], []
    for lab in sorted(by_label):
        group = by_label[lab]
        rng.shuffle(group)
        n_eval = min(len(group), round(eval_n * len(group) / max(1, len(records)))) if eval_n else 0
        evals += group[:n_eval]
        train += group[n_eval:]
    rng.shuffle(train)
    rng.shuffle(evals)
    train, evals = train[:n], evals[:eval_n]
    if len(train) < n:
        typer.echo(f"warning: only {len(train)} training tiles available (asked for {n})")

    out = out or (config.DATA_DIR / f"sft_{dataset}_train.jsonl")
    eval_out = out.with_name(out.name.replace("_train", "_eval")) if "_train" in out.name \
        else out.with_name(out.stem + "_eval.jsonl")

    classes = schema.CLASSES[dataset]
    for name, recs, path in (("train", train, out), ("eval", evals, eval_out)):
        if not recs:
            typer.echo(f"{name}: no records, skipping")
            continue
        n_written = write_sft_jsonl(recs, dataset, path, image_size=image_size, seed=seed,
                                    progress=lambda it, nm=name: tqdm(it, desc=f"write {nm}", unit="ex"))
        counts = Counter(classes[r.label] for r in recs)
        typer.echo(f"{name}: wrote {n_written} examples -> {path}")
        typer.echo(f"  class counts: {dict(sorted(counts.items()))}")
        rep = validate_sft_jsonl(path)
        typer.echo(rep.summary())
        if name == "train":
            cost = estimate_training_cost(rep.total_tokens, epochs, base_model)
            typer.echo(f"  est. LoRA SFT cost for {epochs} epoch(s) on {base_model}: ${cost:.2f}")
        if not rep.ok:
            raise typer.Exit(code=1)
    typer.echo("\nnext: .venv/bin/python scripts/launch_finetune.py "
               f"--dataset-file {out} --eval-file {eval_out}")


if __name__ == "__main__":
    app()
