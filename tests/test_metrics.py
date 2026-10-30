from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pathfrontier.config import PATHOLOGIST_SEC_PER_TILE
from pathfrontier.metrics import compute_metrics, roc_points, sensitivity_at_specificity
from pathfrontier.schema import CLASSES, Prediction, write_predictions

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = {
    "n", "auroc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "f1",
    "sensitivity_at_spec", "threshold_at_spec", "latency_p50_ms", "latency_p95_ms",
    "latency_mean_ms", "cost_per_1k_usd", "cost_per_slide_usd", "coverage",
    "pathologist_minutes_per_slide", "confusion", "per_class", "sensitivity_automated",
    "sensitivity_system",
}


def test_contract_keys(synthetic_preds):
    m = compute_metrics(synthetic_preds, "pcam")
    assert REQUIRED_KEYS <= set(m)
    assert m["n"] == 64
    assert set(m["sensitivity_at_spec"]) == {"0.90", "0.95", "0.99"}
    assert m["coverage"] == 1.0 and m["pathologist_minutes_per_slide"] == 0.0
    assert m["sensitivity"] == m["sensitivity_automated"] == m["sensitivity_system"]
    assert 0.8 < m["auroc"] <= 1.0
    assert len(m["confusion"]) == 2 and len(m["confusion"][0]) == 2
    assert sum(map(sum, m["confusion"])) == 64
    assert set(m["per_class"]) == set(CLASSES["pcam"])
    json.dumps(m)  # must be JSON serialisable


def test_perfect_predictor(make_binary_pred):
    preds = [make_binary_pred(1, 0.9 + 0.01 * i, i) for i in range(10)]
    preds += [make_binary_pred(0, 0.05 + 0.01 * i, 10 + i) for i in range(10)]
    m = compute_metrics(preds, "pcam")
    assert m["auroc"] == 1.0
    assert m["accuracy"] == m["balanced_accuracy"] == 1.0
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0 and m["f1"] == 1.0
    assert all(v == 1.0 for v in m["sensitivity_at_spec"].values())
    # any threshold between the two blocks works; it must separate them
    for t in m["threshold_at_spec"].values():
        assert 0.14 < t <= 0.9


def test_sensitivity_at_spec_interpolation(make_binary_pred):
    # scores: positives 0.9 (5 of them) and 0.3 (5); negatives 0.6 (1) and 0.1 (9)
    preds = [make_binary_pred(1, 0.9, i) for i in range(5)]
    preds += [make_binary_pred(1, 0.3, 5 + i) for i in range(5)]
    preds += [make_binary_pred(0, 0.6, 10)]
    preds += [make_binary_pred(0, 0.1, 11 + i) for i in range(9)]
    fpr, tpr, thr = roc_points(preds, "pcam")
    # ROC points: (0, .5) at thr .9 ; (.1, .5) at .6 ; (.1, 1.0) at .3 ; (1, 1) at .1
    s99, t99 = sensitivity_at_specificity(fpr, tpr, thr, 0.99)
    assert s99 == pytest.approx(0.5, abs=0.06)  # interpolated between fpr 0 (.5) and .1 (1.0)
    assert t99 == pytest.approx(0.9)
    s90, t90 = sensitivity_at_specificity(fpr, tpr, thr, 0.90)
    assert s90 == pytest.approx(1.0)
    assert t90 == pytest.approx(0.3)
    m = compute_metrics(preds, "pcam")
    assert m["sensitivity_at_spec"]["0.90"] == pytest.approx(1.0)
    assert m["threshold_at_spec"]["0.90"] == pytest.approx(0.3)


def test_random_predictor_near_chance(synthetic_records, make_preds):
    preds = make_preds(synthetic_records, "pcam", quality=0.0, seed=5)
    m = compute_metrics(preds, "pcam")
    assert 0.3 < m["auroc"] < 0.7


def test_pathologist_routing(make_binary_pred):
    # 4 positives: 2 auto-correct, 1 auto-missed, 1 routed; 4 negatives: 3 auto-correct, 1 routed
    preds = [
        make_binary_pred(1, 0.9, 0), make_binary_pred(1, 0.8, 1), make_binary_pred(1, 0.2, 2),
        make_binary_pred(1, 0.5, 3, stage="cascade", routed_to="pathologist", cost_usd=0.0),
        make_binary_pred(0, 0.1, 4), make_binary_pred(0, 0.2, 5), make_binary_pred(0, 0.3, 6),
        make_binary_pred(0, 0.5, 7, stage="pathologist", routed_to="pathologist"),
    ]
    m = compute_metrics(preds, "pcam", tiles_per_slide=1000)
    assert m["n"] == 8 and m["n_routed"] == 2 and m["n_automated"] == 6
    assert m["coverage"] == pytest.approx(0.75)
    assert m["sensitivity_automated"] == pytest.approx(2 / 3)
    assert m["sensitivity_system"] == pytest.approx(3 / 4)
    assert m["sensitivity"] == m["sensitivity_system"]
    assert m["specificity_automated"] == 1.0 and m["specificity_system"] == 1.0
    assert m["accuracy"] == pytest.approx(5 / 6)  # routed excluded from automated accuracy
    assert sum(map(sum, m["confusion"])) == 6
    assert m["pathologist_minutes_per_slide"] == pytest.approx(
        0.25 * 1000 * PATHOLOGIST_SEC_PER_TILE / 60)
    # AUROC on automated tiles only: positives {.9,.8,.2} vs negatives {.1,.2,.3}
    # -> 3 + 3 + (1 win + 0.5 tie) = 7.5 of 9 pairs
    assert m["auroc"] == pytest.approx(7.5 / 9)


def test_all_routed_gives_none_automated(make_binary_pred):
    preds = [make_binary_pred(i % 2, 0.5, i, stage="pathologist", routed_to="pathologist")
             for i in range(6)]
    m = compute_metrics(preds, "pcam")
    assert m["coverage"] == 0.0
    assert m["accuracy"] is None and m["auroc"] is None
    assert m["sensitivity_system"] == 1.0 and m["specificity_system"] == 1.0
    assert m["sensitivity_at_spec"]["0.95"] is None


def test_latency_and_cost(make_binary_pred):
    preds = [make_binary_pred(i % 2, 0.9 if i % 2 else 0.1, i, latency_ms=float(i + 1),
                              cost_usd=0.001) for i in range(100)]
    m = compute_metrics(preds, "pcam", tiles_per_slide=10_000)
    assert m["latency_p50_ms"] == pytest.approx(np.percentile(np.arange(1, 101), 50))
    assert m["latency_p95_ms"] == pytest.approx(np.percentile(np.arange(1, 101), 95))
    assert m["latency_mean_ms"] == pytest.approx(50.5)
    assert m["cost_per_1k_usd"] == pytest.approx(1.0)
    assert m["cost_per_slide_usd"] == pytest.approx(10.0)


def test_crc_multiclass(synthetic_records_crc, make_preds):
    preds = make_preds(synthetic_records_crc, "crc", quality=0.6, seed=2, model="probe/x")
    m = compute_metrics(preds, "crc")
    assert len(m["confusion"]) == 9
    assert m["positive_class"] == "TUM"
    assert 0.5 < m["auroc"] <= 1.0  # TUM-vs-rest
    assert 0.5 < m["auroc_macro"] <= 1.0
    assert set(m["per_class"]) == set(CLASSES["crc"])
    fpr, tpr, thr = roc_points(preds, "crc")
    assert fpr[0] == 0.0 and tpr[-1] == 1.0 and thr.max() <= 1.0


def test_bad_prob_length():
    p = Prediction(tile_id="pcam:test:0", dataset="pcam", split="test", y_true=1,
                   y_prob=[0.2, 0.3, 0.5], y_pred=2, model="m")
    with pytest.raises(ValueError):
        compute_metrics([p], "pcam")
    with pytest.raises(ValueError):
        compute_metrics([], "pcam")


def test_eval_cli(tmp_path, synthetic_preds, make_preds, synthetic_records):
    results = tmp_path / "results"
    write_predictions(synthetic_preds, results / "preds_probe_synthetic_test.jsonl")
    base = make_preds(synthetic_records, "pcam", quality=0.3, seed=9,
                      model="fireworks/fake-vlm", stage="baseline", latency_ms=800,
                      cost_per_tile=0.002)
    write_predictions(base, results / "preds_fireworks_fake-vlm_test.jsonl")
    cmd = [sys.executable, str(ROOT / "scripts" / "eval.py"), "--results-dir", str(results)]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert (results / "metrics_probe_synthetic.json").exists()
    assert (results / "metrics_fireworks_fake-vlm.json").exists()
    lb = (results / "leaderboard.md").read_text()
    assert "| model |" in lb and "probe/synthetic" in lb and "fireworks/fake-vlm" in lb
    assert "$/slide" in lb and "coverage" in lb
    m = json.loads((results / "metrics_fireworks_fake-vlm.json").read_text())
    assert m["cost_per_1k_usd"] == pytest.approx(2.0)
    assert m["stage"] == "baseline"
