"""Two-stage cascade (CONTRACTS section 7).

Stage 1 is a cheap linear probe on a pathology foundation encoder. Tiles whose
p_tumor falls below ``low`` are auto-cleared, above ``high`` auto-flagged; the
uncertain band is handed to a fallback: a fine-tuned VLM, a pathologist, or
nothing (probe argmax).

``sweep`` runs ``route`` over a grid of (low, high) thresholds and reports the
metrics for each operating point; ``pareto_front`` picks the non-dominated rows.

Metrics come from ``pathfrontier.metrics.compute_metrics`` when that module is
available; a minimal local fallback with the same key names covers the subset
this module (and the frontier chart) needs, so the sweep works standalone.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pathfrontier import config
from pathfrontier.schema import POSITIVE_INDEX, Prediction

Fallback = Literal["vlm_ft", "pathologist", "none"]

DEFAULT_LOWS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
DEFAULT_HIGHS = (0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
PROBE_ALONE = (0.5, 0.5)  # degenerate band: stage 1 decides everything


def default_grid() -> list[tuple[float, float]]:
    """Default sweep grid: lows x highs plus the degenerate probe-alone point."""
    return [*itertools.product(DEFAULT_LOWS, DEFAULT_HIGHS), PROBE_ALONE]


@dataclass
class CascadeConfig:
    low: float = 0.05
    high: float = 0.95
    fallback: Fallback = "vlm_ft"
    # Per-tile cost/latency overrides. ``None`` means "use the value recorded on
    # each Prediction" (the honest measured number); set these to force a model.
    vlm_cost_per_tile: float | None = None
    vlm_latency_ms: float | None = None
    probe_cost_per_tile: float | None = None
    probe_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"need 0 <= low <= high <= 1, got low={self.low} high={self.high}")
        if self.fallback not in ("vlm_ft", "pathologist", "none"):
            raise ValueError(f"unknown fallback {self.fallback!r}")


def _negative_decision(y_prob: list[float], pos: int) -> int:
    """Argmax over the non-positive classes (0 for binary PCam)."""
    probs = np.asarray(y_prob, dtype=float)
    if probs.size <= 2:
        return 1 - pos
    probs = probs.copy()
    probs[pos] = -np.inf
    return int(np.argmax(probs))


def _pick(override: float | None, measured: float) -> float:
    return measured if override is None else override


def route(
    probe_preds: list[Prediction],
    vlm_preds: dict[str, Prediction] | None,
    cfg: CascadeConfig,
) -> list[Prediction]:
    """Apply the cascade to probe predictions. See module docstring for semantics.

    Routed-to-pathologist tiles get ``y_pred = -1`` (no automated decision) and
    ``extra["pathologist_seconds"]``; metrics treat them as correct for
    ``sensitivity_system`` and exclude them from automated metrics.
    """
    vlm_preds = vlm_preds or {}
    out: list[Prediction] = []
    vlm_model = next((v.model for v in vlm_preds.values()), None)
    for p in probe_preds:
        pos = POSITIVE_INDEX[p.dataset]
        p_tumor = float(p.y_prob[pos])
        probe_cost = _pick(cfg.probe_cost_per_tile, p.cost_usd)
        probe_latency = _pick(cfg.probe_latency_ms, p.latency_ms)
        base = {"tile_id": p.tile_id, "dataset": p.dataset, "split": p.split,
                "y_true": p.y_true, "stage": "cascade"}
        extra = {**p.extra, "p_tumor_probe": p_tumor, "low": cfg.low, "high": cfg.high,
                 "probe_y_pred": p.y_pred, "probe_model": p.model}
        fallback_name = "pathologist" if cfg.fallback == "pathologist" else (vlm_model or cfg.fallback)
        model = f"cascade/{p.model}+{fallback_name}"

        if p_tumor < cfg.low or p_tumor > cfg.high:
            y_pred = pos if p_tumor > cfg.high else _negative_decision(p.y_prob, pos)
            out.append(Prediction(
                **base, y_prob=list(p.y_prob), y_pred=y_pred, model=model,
                latency_ms=probe_latency, cost_usd=probe_cost,
                confidence=max(p.y_prob), reason=None, routed_to="probe", extra=extra,
            ))
            continue

        # Uncertain band.
        vlm = vlm_preds.get(p.tile_id) if cfg.fallback == "vlm_ft" else None
        if cfg.fallback == "none":
            out.append(Prediction(
                **base, y_prob=list(p.y_prob), y_pred=int(np.argmax(p.y_prob)), model=model,
                latency_ms=probe_latency, cost_usd=probe_cost,
                confidence=max(p.y_prob), reason=None, routed_to="probe", extra=extra,
            ))
        elif vlm is not None:
            out.append(Prediction(
                **base, y_prob=list(vlm.y_prob), y_pred=vlm.y_pred, model=model,
                latency_ms=probe_latency + _pick(cfg.vlm_latency_ms, vlm.latency_ms),
                cost_usd=probe_cost + _pick(cfg.vlm_cost_per_tile, vlm.cost_usd),
                confidence=max(vlm.y_prob), reason=vlm.reason, routed_to="vlm_ft",
                extra={**extra, "vlm_model": vlm.model},
            ))
        else:  # pathologist fallback, or VLM asked for but no prediction for this tile
            out.append(Prediction(
                **base, y_prob=list(p.y_prob), y_pred=-1, model=model,
                latency_ms=probe_latency, cost_usd=probe_cost,
                confidence=max(p.y_prob), reason=None, routed_to="pathologist",
                extra={**extra, "pathologist_seconds": config.PATHOLOGIST_SEC_PER_TILE},
            ))
    return out


# --------------------------------------------------------------------------- metrics
def fallback_compute_metrics(
    preds: list[Prediction],
    dataset: str,
    specificity_points=config.FIXED_SPECIFICITY_POINTS,
    tiles_per_slide: int = config.TILES_PER_SLIDE,
) -> dict:
    """Minimal stand-in for ``pathfrontier.metrics.compute_metrics``.

    Same key names for the subset the cascade/frontier needs. Tiles routed to a
    pathologist (``routed_to == "pathologist"`` or ``y_pred == -1``) are excluded
    from automated metrics and counted as correct for ``sensitivity_system``.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    pos = POSITIVE_INDEX[dataset]
    n = len(preds)
    y_true = np.array([p.y_true == pos for p in preds], dtype=bool)
    routed = np.array([p.routed_to == "pathologist" or p.y_pred == -1 for p in preds], dtype=bool)
    auto = ~routed
    y_pred = np.array([p.y_pred == pos for p in preds], dtype=bool)
    score = np.array([p.y_prob[pos] if len(p.y_prob) > pos else 0.0 for p in preds], dtype=float)
    cost = np.array([p.cost_usd for p in preds], dtype=float)
    lat = np.array([p.latency_ms for p in preds], dtype=float)

    def _rate(hit: np.ndarray, mask: np.ndarray) -> float:
        return float(hit[mask].mean()) if mask.any() else float("nan")

    pos_auto, neg_auto = y_true & auto, ~y_true & auto
    sens_auto = _rate(y_pred, pos_auto)
    spec = _rate(~y_pred, neg_auto)
    n_pos = int(y_true.sum())
    sens_system = float((y_pred[pos_auto].sum() + (y_true & routed).sum()) / n_pos) if n_pos else float("nan")
    acc = _rate(y_pred == y_true, auto)

    m: dict = {
        "n": n, "n_automated": int(auto.sum()), "n_routed_pathologist": int(routed.sum()),
        "accuracy": acc, "sensitivity": sens_auto, "sensitivity_automated": sens_auto,
        "sensitivity_system": sens_system, "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens_auto, spec])),
    }
    if auto.sum() > 1 and 0 < y_true[auto].sum() < auto.sum():
        m["auroc"] = float(roc_auc_score(y_true[auto], score[auto]))
        fpr, tpr, thr = roc_curve(y_true[auto], score[auto])
        for s in specificity_points:
            ok = (1 - fpr) >= s
            m[f"sensitivity_at_spec_{s:.2f}"] = float(tpr[ok].max()) if ok.any() else 0.0
            m[f"threshold_at_spec_{s:.2f}"] = float(thr[ok][np.argmax(tpr[ok])]) if ok.any() else 1.0
    else:
        m["auroc"] = float("nan")
        for s in specificity_points:
            m[f"sensitivity_at_spec_{s:.2f}"] = float("nan")
            m[f"threshold_at_spec_{s:.2f}"] = float("nan")
    m.update({
        "latency_p50_ms": float(np.percentile(lat, 50)) if n else float("nan"),
        "latency_p95_ms": float(np.percentile(lat, 95)) if n else float("nan"),
        "latency_mean_ms": float(lat.mean()) if n else float("nan"),
        "cost_per_1k_usd": float(cost.mean() * 1000) if n else float("nan"),
        "cost_per_slide_usd": float(cost.mean() * tiles_per_slide) if n else float("nan"),
        "coverage": float(auto.mean()) if n else float("nan"),
        "pathologist_minutes_per_slide": float(routed.mean() * tiles_per_slide
                                               * config.PATHOLOGIST_SEC_PER_TILE / 60) if n else float("nan"),
    })
    return m


def resolve_compute_metrics() -> Callable[..., dict]:
    """The real ``compute_metrics`` if ``pathfrontier.metrics`` exists, else the fallback."""
    try:
        from pathfrontier.metrics import compute_metrics
    except ImportError:
        return fallback_compute_metrics
    return compute_metrics


_MISSING = object()


def scalar_metrics(metrics: dict) -> dict:
    """Keep scalar entries; flatten one level of numeric-keyed dicts.

    ``{"sensitivity_at_spec": {"0.95": x}}`` becomes ``sensitivity_at_spec_0.95 = x`` so
    the nested form of the real ``compute_metrics`` lines up with the flat column names
    the sweep CSV and the frontier chart use. Confusion matrices / per-class dicts drop.
    """
    def _scalar(v):
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v if isinstance(v, (bool, int, float, str)) or v is None else _MISSING

    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                try:
                    float(kk)
                except (TypeError, ValueError):
                    continue
                if (sv := _scalar(vv)) is not _MISSING:
                    out[f"{k}_{kk}"] = sv
        elif (sv := _scalar(v)) is not _MISSING:
            out[k] = sv
    return out


# --------------------------------------------------------------------------- sweep
def sweep(
    probe_preds: list[Prediction],
    vlm_preds: dict[str, Prediction] | None,
    dataset: str,
    grid: list[tuple[float, float]] | None = None,
    fallback: Fallback = "vlm_ft",
    tiles_per_slide: int = config.TILES_PER_SLIDE,
    specificity_points=config.FIXED_SPECIFICITY_POINTS,
    cfg_kwargs: dict | None = None,
) -> pd.DataFrame:
    """One row per (low, high): routing stats + every scalar metric from compute_metrics."""
    grid = grid if grid is not None else default_grid()
    compute_metrics = resolve_compute_metrics()
    rows = []
    for low, high in grid:
        cfg = CascadeConfig(low=low, high=high, fallback=fallback, **(cfg_kwargs or {}))
        routed = route(probe_preds, vlm_preds, cfg)
        n_routed = sum(r.routed_to != "probe" for r in routed)
        n_path = sum(r.routed_to == "pathologist" for r in routed)
        m = scalar_metrics(compute_metrics(routed, dataset, specificity_points, tiles_per_slide))
        # Routing columns go last so they win over any same-named keys from compute_metrics:
        # here n_routed / fraction_routed mean "left the probe" (VLM or pathologist).
        rows.append({
            **m, "low": low, "high": high, "fallback": fallback,
            "fraction_routed": n_routed / len(routed) if routed else float("nan"),
            "n_routed": n_routed, "n_routed_pathologist": n_path,
        })
    df = pd.DataFrame(rows)
    lead = ["low", "high", "fallback", "fraction_routed", "n_routed", "n_routed_pathologist"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    for col in ("cost_per_slide_usd", "sensitivity_system"):
        if col not in df.columns:
            raise KeyError(f"compute_metrics did not return {col!r}")
    return df


def pareto_front(df: pd.DataFrame, x: str = "cost_per_slide_usd", y: str = "sensitivity_system") -> pd.DataFrame:
    """Rows not dominated on (minimise x, maximise y), sorted by x. Index is preserved."""
    d = df.dropna(subset=[x, y]).sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, row in d.iterrows():
        if row[y] > best:
            keep.append(idx)
            best = row[y]
    return df.loc[keep]


def pareto_mask(df: pd.DataFrame, x: str = "cost_per_slide_usd", y: str = "sensitivity_system") -> pd.Series:
    """Boolean Series aligned with ``df``: True on the Pareto front."""
    return pd.Series(df.index.isin(pareto_front(df, x, y).index), index=df.index)


__all__ = [
    "CascadeConfig",
    "default_grid",
    "fallback_compute_metrics",
    "pareto_front",
    "pareto_mask",
    "resolve_compute_metrics",
    "route",
    "scalar_metrics",
    "sweep",
]
