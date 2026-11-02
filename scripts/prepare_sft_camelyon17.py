"""Build the Fireworks vision-SFT files for Camelyon17 from the slide-disjoint training pool.

The pool (data/camelyon17/train.parquet, slides 20, 21, 27-29) is split by SLIDE so the dev set
behaves like the real test set (slides the model never trained on). The test set (slides 22-26)
is never read here.

    train: every pool tile from --train-slides           -> data/sft_camelyon17_train.jsonl
    eval:  --eval-n tiles from --dev-slides (loss curve)  -> data/sft_camelyon17_eval.jsonl
    dev:   the full --dev-slides tile list, for choosing the decision cutoff after training
           (written as data/camelyon17/dev_manifest.csv; scored with run_vlm_ft.py --split train
           filtered to these ids)

    .venv/bin/python scripts/prepare_sft_camelyon17.py
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import typer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config, data
from pathfrontier.finetune import estimate_training_cost, validate_sft_jsonl, write_sft_jsonl
from pathfrontier.vlm_client import V2_IMAGE_PX

app = typer.Typer(add_completion=False)
BASE_MODEL = "accounts/fireworks/models/qwen3-vl-8b-instruct"


@app.command()
def main(
    train_slides: str = typer.Option("20,21,27,28"),
    dev_slides: str = typer.Option("29"),
    eval_n: int = typer.Option(300, help="Dev tiles uploaded as the Fireworks eval set"),
    prompt_version: str = typer.Option("v2"),
    image_size: int = typer.Option(V2_IMAGE_PX),
    epochs: int = typer.Option(2, help="Only used for the cost estimate"),
    seed: int = typer.Option(0),
) -> None:
    train_ids = {int(s) for s in train_slides.split(",")}
    dev_ids = {int(s) for s in dev_slides.split(",")}
    if train_ids & dev_ids:
        raise typer.BadParameter("train and dev slides overlap")
    manifest = config.DATA_DIR / "camelyon17" / "train_manifest.csv"
    with open(manifest) as f:
        slide_of = {r["tile_id"]: int(r["slide"]) for r in csv.DictReader(f)}

    records = data.load_split("camelyon17", "train", limit=None, seed=seed)
    train = [r for r in records if slide_of[r.tile_id] in train_ids]
    dev = [r for r in records if slide_of[r.tile_id] in dev_ids]
    rng = random.Random(seed)
    rng.shuffle(train)
    evals = rng.sample(dev, min(eval_n, len(dev)))

    out_dir = config.DATA_DIR
    dev_manifest = out_dir / "camelyon17" / "dev_manifest.csv"
    with open(dev_manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "slide", "label"])
        for r in dev:
            w.writerow([r.tile_id, slide_of[r.tile_id], r.label])
    typer.echo(f"dev: {len(dev)} tiles from slides {sorted(dev_ids)} -> {dev_manifest}")

    for name, recs in (("train", train), ("eval", evals)):
        path = out_dir / f"sft_camelyon17_{name}.jsonl"
        n = write_sft_jsonl(recs, "camelyon17", path, image_size=image_size, seed=seed,
                            prompt_version=prompt_version,
                            progress=lambda it, nm=name: tqdm(it, desc=f"write {nm}", unit="ex"))
        counts = Counter("tumor" if r.label else "normal" for r in recs)
        typer.echo(f"{name}: {n} examples -> {path} ({path.stat().st_size / 1e6:.0f} MB) "
                   f"{dict(sorted(counts.items()))}")
        rep = validate_sft_jsonl(path)
        typer.echo(rep.summary())
        if name == "train":
            cost = estimate_training_cost(rep.total_tokens, epochs, BASE_MODEL)
            typer.echo(f"  est. LoRA SFT cost for {epochs} epoch(s): ${cost:.2f}")
        if not rep.ok:
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
