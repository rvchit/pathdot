"""Offline tests for pathfrontier.probe (synthetic gaussian clusters, no downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from pathfrontier.probe import LinearProbe
from pathfrontier.schema import CLASSES, Prediction, load_predictions, write_predictions


def clusters(n_classes: int, n: int, dim: int = 32, sep: float = 4.0, seed: int = 0):
    centers = np.random.default_rng(1234).standard_normal((n_classes, dim)) * sep  # fixed centres
    rng = np.random.default_rng(seed)
    y = np.arange(n) % n_classes
    X = centers[y] + rng.standard_normal((n, dim))
    return X.astype(np.float32), y.astype(np.int64)


@pytest.mark.parametrize("dataset", ["pcam", "crc"])
def test_probe_learns_clusters(dataset):
    k = len(CLASSES[dataset])
    Xtr, ytr = clusters(k, 1200, seed=1)
    Xte, yte = clusters(k, 400, seed=2)
    probe = LinearProbe("fake", dataset, C=1.0, calibrate=True).fit(Xtr, ytr)
    assert probe.is_calibrated and probe.n_calib == 240 and probe.n_train == 960
    p = probe.predict_proba(Xte)
    assert p.shape == (400, k)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert (p.argmax(1) == yte).mean() > 0.97


def test_small_n_skips_calibration():
    X, y = clusters(2, 100)
    probe = LinearProbe("fake", "pcam", calibrate=True).fit(X, y)
    assert not probe.is_calibrated and probe.n_train == 100
    p = probe.predict_proba(X)
    assert p.shape == (100, 2) and (p.argmax(1) == y).mean() > 0.97


def test_no_calibrate_flag():
    X, y = clusters(2, 500)
    probe = LinearProbe("fake", "pcam", calibrate=False).fit(X, y)
    assert not probe.is_calibrated and probe.n_train == 500


def test_calibration_improves_or_keeps_probabilities_sane():
    X, y = clusters(2, 1000, sep=1.5)
    probe = LinearProbe("fake", "pcam", calibrate=True).fit(X, y)
    p = probe.predict_proba(X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # isotonic output for the positive class must be monotone in the raw score
    raw = probe._raw_proba(X)[:, 1]
    order = np.argsort(raw)
    assert np.all(np.diff(p[order, 1]) >= -1e-9)


def test_save_load_roundtrip(tmp_path):
    X, y = clusters(2, 600)
    probe = LinearProbe("phikon-v2", "pcam").fit(X, y)
    path = tmp_path / "probe.joblib"
    probe.save(path)
    loaded = LinearProbe.load(path)
    assert loaded.encoder_name == "phikon-v2" and loaded.dataset == "pcam"
    assert loaded.is_calibrated == probe.is_calibrated
    np.testing.assert_allclose(loaded.predict_proba(X), probe.predict_proba(X))


def test_predict_records_emits_valid_predictions(tmp_path):
    X, y = clusters(2, 300)
    probe = LinearProbe("phikon-v2", "pcam").fit(X, y)
    ids = [f"pcam:test:{i}" for i in range(len(y))]
    preds = probe.predict_records(X, y, ids, "test", latency_ms_per_tile=0.7, cost_per_tile=5e-7)
    assert len(preds) == 300
    p0 = preds[0]
    assert isinstance(p0, Prediction)
    assert p0.stage == "probe" and p0.model == "probe/phikon-v2"
    assert p0.tile_id == "pcam:test:0" and p0.dataset == "pcam" and p0.split == "test"
    assert len(p0.y_prob) == 2 and abs(sum(p0.y_prob) - 1) < 1e-6
    assert p0.y_pred == int(np.argmax(p0.y_prob))
    assert p0.latency_ms == 0.7 and p0.cost_usd == 5e-7
    assert p0.confidence == max(p0.y_prob)
    # round-trips through the JSONL contract
    out = tmp_path / "preds.jsonl"
    assert write_predictions(preds, out) == 300
    back = load_predictions(out)
    assert back[0] == p0


def test_unseen_class_gets_zero_probability():
    # only 3 of 9 CRC classes present in training: probe must still emit 9 columns
    X, y = clusters(3, 300)
    probe = LinearProbe("fake", "crc").fit(X, y)
    p = probe.predict_proba(X)
    assert p.shape == (300, 9)
    assert np.allclose(p[:, 3:], 0.0)


def test_bad_inputs():
    with pytest.raises(ValueError):
        LinearProbe("fake", "nope")
    X, y = clusters(2, 50)
    with pytest.raises(ValueError):
        LinearProbe("fake", "pcam").fit(X, y + 5)
