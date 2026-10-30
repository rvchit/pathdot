"""Generate a realistic SYNTHETIC set of prediction files so the sweep and the
frontier chart can be demoed before any real model run exists.

    uv run python scripts/make_demo_results.py --out-dir results/demo

Writes (model name -> file):
    baseline_demo  -> results/demo/preds_baseline_demo_test.jsonl   (stage=baseline, AUROC~0.75)
    probe_demo     -> results/demo/preds_probe_demo_test.jsonl      (stage=probe,    AUROC~0.98)
    vlm_ft_demo    -> results/demo/preds_vlm_ft_demo_test.jsonl     (stage=vlm_ft,   AUROC~0.93)

Scores follow a binormal model (score ~ N(+-mu/2, 1), AUROC = Phi(mu/sqrt 2)) with a
shared per-tile "difficulty" so the tiles the probe finds hard are also harder for the
VLMs. Probe probabilities are the exact calibrated posterior; VLM probabilities are
quantised the way an LLM's self-reported confidence looks. Fully seeded.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathfrontier import config
from pathfrontier.schema import POSITIVE_INDEX, Prediction, preds_path, write_predictions

app = typer.Typer(add_completion=False)

REASONS = {
    1: [
        "Sheets of crowded cells with enlarged hyperchromatic nuclei and loss of nodal architecture.",
        "Cohesive epithelial nests with pleomorphic nuclei replacing lymphoid tissue.",
        "Atypical cells with prominent nucleoli and high nuclear-to-cytoplasmic ratio.",
    ],
    0: [
        "Uniform small lymphocytes with preserved follicular architecture; no atypia.",
        "Benign lymphoid tissue with sinus histiocytes and intact capsule.",
        "Adipose and fibrous stroma without epithelial cells.",
    ],
}


@dataclass(frozen=True)
class DemoModel:
    name: str
    stage: str
    auroc: float
    latency_ms: float
    cost_per_1k: float
    quantise: float | None   # VLM "confidence" granularity; None = calibrated continuous
    with_reason: bool


DEMO_MODELS = (
    DemoModel("baseline_demo", "baseline", 0.75, 1200.0, 3.0, 0.1, True),
    DemoModel("probe_demo", "probe", 0.98, 0.7, 0.0005, None, False),
    DemoModel("vlm_ft_demo", "vlm_ft", 0.93, 900.0, 0.8, 0.05, True),
)


def _mu_for_auroc(auroc: float) -> float:
    return float(np.sqrt(2.0) * NormalDist().inv_cdf(auroc))


def _posterior(z: np.ndarray, mu: float) -> np.ndarray:
    """Calibrated P(y=1 | z) for z ~ N(+-mu/2, 1) with a 50/50 prior."""
    return 1.0 / (1.0 + np.exp(-mu * z))


def generate(
    out_dir: Path,
    n: int = 4096,
    seed: int = 0,
    dataset: str = "pcam",
    split: str = "test",
    shared_difficulty: float = 0.5,
) -> dict[str, Path]:
    """Write the three demo prediction files; returns {model_name: path}."""
    if dataset != "pcam":
        raise typer.BadParameter("demo generator is binary (pcam) only")
    rng = np.random.default_rng(seed)
    pos = POSITIVE_INDEX[dataset]
    y = np.zeros(n, dtype=int)
    y[rng.permutation(n)[: n // 2]] = pos          # exactly 50% positives
    sign = np.where(y == pos, 1.0, -1.0)
    tile_ids = [f"{dataset}:{split}:{i}" for i in range(n)]
    e_shared = rng.standard_normal(n)              # hard tiles are hard for everyone

    written: dict[str, Path] = {}
    for m in DEMO_MODELS:
        mu = _mu_for_auroc(m.auroc)
        e_own = rng.standard_normal(n)
        noise = shared_difficulty * e_shared + np.sqrt(1 - shared_difficulty**2) * e_own
        z = sign * mu / 2 + noise
        p1 = _posterior(z, mu)
        if m.quantise is not None:                 # LLM-style confidence buckets
            p1 = np.clip(np.round(p1 / m.quantise) * m.quantise, 0.02, 0.98)
        latency = m.latency_ms * rng.lognormal(0.0, 0.25, n)
        cost = (m.cost_per_1k / 1000.0) * rng.lognormal(0.0, 0.10, n)
        preds = []
        for i in range(n):
            y_prob = [float(1 - p1[i]), float(p1[i])]
            y_pred = int(p1[i] >= 0.5)
            reason = None
            if m.with_reason:
                reason = REASONS[y_pred][int(rng.integers(len(REASONS[y_pred])))]
            preds.append(Prediction(
                tile_id=tile_ids[i], dataset=dataset, split=split, y_true=int(y[i]),
                y_prob=y_prob, y_pred=y_pred, model=m.name, stage=m.stage,
                latency_ms=float(latency[i]), cost_usd=float(cost[i]), reason=reason,
                extra={"synthetic": True, "target_auroc": m.auroc},
            ))
        path = preds_path(m.name, split, out_dir)
        write_predictions(preds, path)
        written[m.name] = path
    return written


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="where to write preds_*.jsonl")] = config.RESULTS_DIR / "demo",
    n: Annotated[int, typer.Option(help="number of tiles")] = 4096,
    seed: Annotated[int, typer.Option()] = 0,
    dataset: Annotated[str, typer.Option()] = "pcam",
) -> None:
    written = generate(out_dir, n=n, seed=seed, dataset=dataset)
    for name, path in written.items():
        typer.echo(f"{name:14s} -> {path}")


if __name__ == "__main__":
    app()
