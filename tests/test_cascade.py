"""Tests for pathfrontier.cascade + scripts/sweep.py, plot_frontier.py, make_demo_results.py.

Runs offline with synthetic fixtures. Uses the minimal fallback metrics in
cascade.py unless pathfrontier.metrics exists (one test exercises the real one).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathfrontier import cascade, config
from pathfrontier.cascade import CascadeConfig, fallback_compute_metrics, pareto_front, route, sweep
from pathfrontier.schema import Prediction

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _probe(tile: str, p1: float, y: int, cost=5e-7, lat=0.7) -> Prediction:
    return Prediction(tile_id=tile, dataset="pcam", split="test", y_true=y, y_prob=[1 - p1, p1],
                      y_pred=int(p1 >= 0.5), model="probe/phikon-v2", stage="probe",
                      latency_ms=lat, cost_usd=cost)


def _vlm(tile: str, p1: float, y: int, cost=8e-4, lat=900.0) -> Prediction:
    return Prediction(tile_id=tile, dataset="pcam", split="test", y_true=y, y_prob=[1 - p1, p1],
                      y_pred=int(p1 >= 0.5), model="ft/qwen", stage="vlm_ft",
                      latency_ms=lat, cost_usd=cost, reason="nuclei look atypical")


@pytest.fixture
def small():
    probe = [
        _probe("pcam:test:0", 0.01, 0),   # below low -> auto negative
        _probe("pcam:test:1", 0.99, 1),   # above high -> auto positive
        _probe("pcam:test:2", 0.40, 1),   # band
        _probe("pcam:test:3", 0.60, 0),   # band, no VLM pred available
        _probe("pcam:test:4", 0.97, 0),   # above high but actually negative (probe FP)
    ]
    vlm = {"pcam:test:2": _vlm("pcam:test:2", 0.9, 1)}
    return probe, vlm


@pytest.fixture(autouse=True)
def use_fallback_metrics(monkeypatch):
    """Deterministic: the sweep tests use the local fallback regardless of metrics.py."""
    monkeypatch.setattr(cascade, "resolve_compute_metrics", lambda: fallback_compute_metrics)


# ----------------------------------------------------------------------------- route
def test_route_band_vlm_ft(small):
    probe, vlm = small
    out = route(probe, vlm, CascadeConfig(low=0.05, high=0.95, fallback="vlm_ft"))
    by = {r.tile_id: r for r in out}
    assert len(out) == len(probe)
    assert all(r.stage == "cascade" for r in out)
    assert all(r.model == "cascade/probe/phikon-v2+ft/qwen" for r in out)
    assert all(r.y_true == p.y_true and r.tile_id == p.tile_id for r, p in zip(out, probe))

    assert by["pcam:test:0"].routed_to == "probe" and by["pcam:test:0"].y_pred == 0
    assert by["pcam:test:1"].routed_to == "probe" and by["pcam:test:1"].y_pred == 1
    assert by["pcam:test:4"].y_pred == 1  # probe decides, even when wrong

    r2 = by["pcam:test:2"]
    assert r2.routed_to == "vlm_ft"
    assert r2.y_pred == 1 and r2.y_prob == pytest.approx([0.1, 0.9])
    assert r2.reason == "nuclei look atypical"
    assert r2.cost_usd == pytest.approx(5e-7 + 8e-4)
    assert r2.latency_ms == pytest.approx(0.7 + 900.0)
    assert r2.confidence == pytest.approx(0.9)
    assert r2.extra["p_tumor_probe"] == pytest.approx(0.40)

    r3 = by["pcam:test:3"]  # VLM missing -> pathologist path
    assert r3.routed_to == "pathologist" and r3.y_pred == -1
    assert r3.cost_usd == pytest.approx(5e-7) and r3.latency_ms == pytest.approx(0.7)
    assert r3.extra["pathologist_seconds"] == config.PATHOLOGIST_SEC_PER_TILE


def test_route_pathologist_fallback(small):
    probe, vlm = small
    out = route(probe, vlm, CascadeConfig(low=0.05, high=0.95, fallback="pathologist"))
    by = {r.tile_id: r for r in out}
    assert all(r.model == "cascade/probe/phikon-v2+pathologist" for r in out)
    for t in ("pcam:test:2", "pcam:test:3"):
        assert by[t].routed_to == "pathologist"
        assert by[t].y_pred == -1
        assert by[t].y_pred != by[t].y_true           # never leaks the label
        assert by[t].cost_usd == pytest.approx(5e-7)  # probe cost only
        assert by[t].extra["pathologist_seconds"] == config.PATHOLOGIST_SEC_PER_TILE
    assert by["pcam:test:0"].routed_to == "probe"


def test_route_none_fallback(small):
    probe, vlm = small
    out = route(probe, vlm, CascadeConfig(low=0.05, high=0.95, fallback="none"))
    by = {r.tile_id: r for r in out}
    assert by["pcam:test:2"].routed_to == "probe" and by["pcam:test:2"].y_pred == 0
    assert by["pcam:test:3"].routed_to == "probe" and by["pcam:test:3"].y_pred == 1
    assert all(r.cost_usd == pytest.approx(5e-7) for r in out)


def test_route_config_overrides(small):
    probe, vlm = small
    cfg = CascadeConfig(low=0.05, high=0.95, probe_cost_per_tile=1e-6, probe_latency_ms=2.0,
                        vlm_cost_per_tile=0.01, vlm_latency_ms=100.0)
    by = {r.tile_id: r for r in route(probe, vlm, cfg)}
    assert by["pcam:test:0"].cost_usd == pytest.approx(1e-6)
    assert by["pcam:test:2"].cost_usd == pytest.approx(1e-6 + 0.01)
    assert by["pcam:test:2"].latency_ms == pytest.approx(102.0)


def test_config_validation():
    with pytest.raises(ValueError):
        CascadeConfig(low=0.9, high=0.1)
    with pytest.raises(ValueError):
        CascadeConfig(fallback="human")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- metrics fallback
def test_fallback_metrics_pathologist_semantics(small):
    probe, vlm = small
    out = route(probe, vlm, CascadeConfig(low=0.05, high=0.95, fallback="pathologist"))
    m = fallback_compute_metrics(out, "pcam", tiles_per_slide=1000)
    # automated: tiles 0 (TN), 1 (TP), 4 (FP). routed: 2 (pos), 3 (neg)
    assert m["coverage"] == pytest.approx(3 / 5)
    assert m["sensitivity_automated"] == pytest.approx(1.0)
    assert m["sensitivity_system"] == pytest.approx(1.0)  # TP + the routed positive
    assert m["specificity"] == pytest.approx(0.5)
    assert m["pathologist_minutes_per_slide"] == pytest.approx(2 / 5 * 1000 * config.PATHOLOGIST_SEC_PER_TILE / 60)
    assert m["cost_per_slide_usd"] == pytest.approx(5e-7 * 1000)
    for k in ("auroc", "latency_p50_ms", "sensitivity_at_spec_0.95", "cost_per_1k_usd"):
        assert k in m


# ----------------------------------------------------------------------------- sweep
def _synthetic(n=400, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = np.where(y == 1, 2.0, -2.0) + rng.standard_normal(n) * 1.3
    p_probe = 1 / (1 + np.exp(-z))
    z2 = np.where(y == 1, 1.5, -1.5) + rng.standard_normal(n)
    p_vlm = 1 / (1 + np.exp(-z2))
    probe = [_probe(f"pcam:test:{i}", float(p_probe[i]), int(y[i])) for i in range(n)]
    vlm = {f"pcam:test:{i}": _vlm(f"pcam:test:{i}", float(p_vlm[i]), int(y[i])) for i in range(n)}
    return probe, vlm


def test_sweep_shape_and_monotone():
    probe, vlm = _synthetic()
    df = sweep(probe, vlm, "pcam", fallback="vlm_ft", tiles_per_slide=1000)
    assert len(df) == len(cascade.default_grid())
    for col in ("low", "high", "fraction_routed", "n_routed", "cost_per_slide_usd",
                "sensitivity_system", "auroc", "coverage", "latency_p50_ms"):
        assert col in df.columns
    assert ((df["fraction_routed"] >= 0) & (df["fraction_routed"] <= 1)).all()
    # probe-alone point routes nothing
    alone = df[(df.low == 0.5) & (df.high == 0.5)].iloc[0]
    assert alone["n_routed"] == 0 and alone["coverage"] == 1.0
    # raising `low` (fixed high) narrows the band: fewer tiles routed, lower cost
    for _high, g in df[df.high != 0.5].groupby("high"):
        g = g.sort_values("low")
        assert (np.diff(g["fraction_routed"].values) <= 1e-12).all()
        assert (np.diff(g["cost_per_slide_usd"].values) <= 1e-12).all()
    # raising `high` (fixed low) widens the band: more tiles routed, higher cost
    for _low, g in df[df.low != 0.5].groupby("low"):
        g = g.sort_values("high")
        assert (np.diff(g["fraction_routed"].values) >= -1e-12).all()
        assert (np.diff(g["cost_per_slide_usd"].values) >= -1e-12).all()
    # with a VLM fallback nobody goes to the pathologist
    assert (df["coverage"] == 1.0).all()
    assert (df["pathologist_minutes_per_slide"] == 0).all()


def test_sweep_pathologist_fallback_trades_minutes_for_sensitivity():
    probe, _ = _synthetic()
    df = sweep(probe, None, "pcam", fallback="pathologist", tiles_per_slide=1000)
    assert (df["n_routed_pathologist"] == df["n_routed"]).all()
    assert (df["pathologist_minutes_per_slide"] >= 0).all()
    widest = df.sort_values("fraction_routed").iloc[-1]
    alone = df[(df.low == 0.5) & (df.high == 0.5)].iloc[0]
    assert widest["sensitivity_system"] >= alone["sensitivity_system"]
    assert widest["pathologist_minutes_per_slide"] > 0


# ----------------------------------------------------------------------------- pareto
def test_pareto_front():
    df = pd.DataFrame({
        "cost_per_slide_usd": [1.0, 2.0, 3.0, 2.0, 0.5, 4.0],
        "sensitivity_system": [0.90, 0.95, 0.94, 0.80, 0.85, 0.99],
    })
    front = pareto_front(df)
    assert list(front.index) == [4, 0, 1, 5]
    assert (np.diff(front["cost_per_slide_usd"].values) > 0).all()
    assert (np.diff(front["sensitivity_system"].values) > 0).all()
    mask = cascade.pareto_mask(df)
    assert mask.sum() == 4 and bool(mask[2]) is False
    # custom columns + NaNs are tolerated
    df2 = df.rename(columns={"cost_per_slide_usd": "x", "sensitivity_system": "y"})
    df2.loc[5, "y"] = np.nan
    assert list(pareto_front(df2, x="x", y="y").index) == [4, 0, 1]


# ----------------------------------------------------------------------------- end to end
def test_end_to_end_demo_sweep_plot(tmp_path: Path):
    demo = _load_script("make_demo_results")
    sw = _load_script("sweep")
    pf = _load_script("plot_frontier")

    files = demo.generate(tmp_path, n=512, seed=3)
    assert set(files) == {"baseline_demo", "probe_demo", "vlm_ft_demo"}
    # consistent tile ids across files, 50% positives
    from pathfrontier.schema import load_predictions
    ids = {k: [p.tile_id for p in load_predictions(v)] for k, v in files.items()}
    assert ids["baseline_demo"] == ids["probe_demo"] == ids["vlm_ft_demo"]
    probe = load_predictions(files["probe_demo"])
    assert sum(p.y_true for p in probe) == 256
    base_m = fallback_compute_metrics(load_predictions(files["baseline_demo"]), "pcam")
    probe_m = fallback_compute_metrics(probe, "pcam")
    assert probe_m["auroc"] > base_m["auroc"]
    assert probe_m["cost_per_slide_usd"] < base_m["cost_per_slide_usd"]

    _df1, csv1 = sw.run_sweep(files["probe_demo"], files["vlm_ft_demo"], "pcam", "vlm_ft", 10_000, tmp_path)
    _df2, csv2 = sw.run_sweep(files["probe_demo"], None, "pcam", "pathologist", 10_000, tmp_path)
    assert csv1.name == "sweep_vlm_ft.csv" and csv2.name == "sweep_pathologist.csv"
    assert csv1.exists() and csv2.exists()

    out = tmp_path / "frontier.png"
    sweeps = pf.load_sweeps([csv1, csv2], "sensitivity_system")
    pts = pf.load_points(list(files.values()), "sensitivity_system", "pcam", 10_000)
    pf.plot(sweeps, pts, out, "sensitivity_system")
    js = pf.write_points_json(sweeps, pts, out.with_suffix(".json"), "sensitivity_system")
    assert out.exists() and out.stat().st_size > 10_000
    payload = json.loads(js.read_text())
    assert len(payload["points"]) == 3 and len(payload["sweeps"]) == 2
    assert any(p["pareto"] for p in payload["sweeps"][0]["points"])

    # alternate metric works through the tolerant lookup
    sweeps2 = pf.load_sweeps([csv1], "sensitivity_at_spec_0.95")
    assert sweeps2[0]["df"]["_y"].notna().all()


def test_lookup_metric_tolerant():
    pf = _load_script("plot_frontier")
    assert pf.lookup_metric({"sensitivity_at_spec_0.95": 0.9}, "sensitivity_at_spec_0.95") == 0.9
    assert pf.lookup_metric({"sensitivity_at_spec0.95": 0.8}, "sensitivity_at_spec_0.95") == 0.8
    assert pf.lookup_metric({"sensitivity_at_spec": {"0.95": 0.7}}, "sensitivity_at_spec_0.95") == 0.7
    with pytest.raises(KeyError):
        pf.lookup_metric({"auroc": 1.0}, "sensitivity_at_spec_0.95")


@pytest.mark.skipif(importlib.util.find_spec("pathfrontier.metrics") is None,
                    reason="pathfrontier.metrics not written yet")
def test_sweep_with_real_metrics(monkeypatch):
    from pathfrontier.metrics import compute_metrics
    monkeypatch.setattr(cascade, "resolve_compute_metrics", lambda: compute_metrics)
    probe, vlm = _synthetic(n=200)
    df = sweep(probe, vlm, "pcam", grid=[(0.05, 0.95), (0.5, 0.5)], fallback="pathologist")
    assert {"cost_per_slide_usd", "sensitivity_system", "coverage"} <= set(df.columns)
