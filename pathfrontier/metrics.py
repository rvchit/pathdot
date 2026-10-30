"""Evaluation metrics over `Prediction` records (CONTRACTS.md section 6).

Conventions
-----------
* The clinical question is always positive-vs-rest with positive = POSITIVE_INDEX[dataset]
  (pcam: tumor, crc: TUM). ``auroc`` and ``sensitivity_at_spec`` use the positive-class
  probability ``y_prob[POSITIVE_INDEX]``. For 9-class crc we additionally report
  ``auroc_macro`` (one-vs-rest macro over all classes).
* Pathologist-routed tiles (stage == "pathologist" or routed_to == "pathologist") are read by a
  human, so they count as CORRECT for the *system* sensitivity/specificity, but they are excluded
  from every *automated* metric (accuracy, AUROC, confusion, ...). ``coverage`` is the fraction of
  tiles the system decided automatically. With no routing, system == automated.
* ``sensitivity`` / ``specificity`` are the system-level numbers (what the frontier plots);
  ``*_automated`` and ``*_system`` are always present as explicit keys.
* Undefined metrics (e.g. AUROC with a single class present) are reported as ``None``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from pathfrontier.config import PATHOLOGIST_SEC_PER_TILE, TILES_PER_SLIDE
from pathfrontier.schema import CLASSES, POSITIVE_INDEX, Prediction


def is_routed(p: Prediction) -> bool:
    return p.stage == "pathologist" or p.routed_to == "pathologist"


def _arrays(preds: Sequence[Prediction], dataset: str):
    n_classes = len(CLASSES[dataset])
    y_true = np.array([p.y_true for p in preds], dtype=np.int64)
    y_pred = np.array([p.y_pred for p in preds], dtype=np.int64)
    y_prob = np.zeros((len(preds), n_classes), dtype=np.float64)
    for i, p in enumerate(preds):
        if len(p.y_prob) != n_classes:
            raise ValueError(
                f"{p.tile_id}: y_prob has {len(p.y_prob)} entries, expected {n_classes}")
        y_prob[i] = p.y_prob
    routed = np.array([is_routed(p) for p in preds], dtype=bool)
    return y_true, y_pred, y_prob, routed


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
    except ValueError:
        return None
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def roc_points(preds: Sequence[Prediction], dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve (fpr, tpr, thresholds) for positive-vs-rest on the automated tiles.
    Thresholds are clipped to [0, 1] (sklearn reports +inf for the first point)."""
    y_true, _, y_prob, routed = _arrays(preds, dataset)
    pos = POSITIVE_INDEX[dataset]
    keep = ~routed
    y_bin = (y_true[keep] == pos).astype(int)
    score = y_prob[keep, pos]
    if len(y_bin) == 0 or y_bin.min() == y_bin.max():
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    fpr, tpr, thr = roc_curve(y_bin, score)
    return fpr, tpr, np.clip(thr, 0.0, 1.0)


def sensitivity_at_specificity(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray,
                               spec: float) -> tuple[float, float]:
    """Interpolate the ROC curve at FPR = 1 - spec. Returns (sensitivity, threshold) where the
    threshold is the most permissive score cut-off that still achieves >= spec."""
    target_fpr = 1.0 - spec
    # collapse vertical segments: for each distinct fpr keep the max tpr, so np.interp is well
    # defined on a strictly increasing x grid.
    ufpr, inv = np.unique(fpr, return_inverse=True)
    utpr = np.zeros_like(ufpr)
    np.maximum.at(utpr, inv, tpr)
    sens = float(np.interp(target_fpr, ufpr, utpr))
    ok = np.flatnonzero(fpr <= target_fpr + 1e-12)
    threshold = float(thr[ok[-1]]) if len(ok) else 1.0
    return sens, threshold


def compute_metrics(preds: Sequence[Prediction], dataset: str,
                    specificity_points: Sequence[float] = (0.90, 0.95, 0.99),
                    tiles_per_slide: int = TILES_PER_SLIDE) -> dict:
    """All headline metrics for one model's predictions on one dataset. See module docstring."""
    preds = list(preds)
    if not preds:
        raise ValueError("compute_metrics: no predictions")
    classes = CLASSES[dataset]
    n_classes = len(classes)
    pos = POSITIVE_INDEX[dataset]
    y_true, y_pred, y_prob, routed = _arrays(preds, dataset)
    n = len(preds)
    auto = ~routed
    n_auto = int(auto.sum())
    n_routed = n - n_auto

    yt, yp, pr = y_true[auto], y_pred[auto], y_prob[auto]
    is_pos_true = y_true == pos
    is_pos_pred = y_pred == pos

    # --- automated classification metrics -------------------------------------------------
    labels = list(range(n_classes))
    if n_auto:
        accuracy = float((yt == yp).mean())
        balanced_accuracy = _safe(balanced_accuracy_score, yt, yp)
        f1_avg = "binary" if n_classes == 2 else "macro"
        f1 = _safe(f1_score, yt, yp, average=f1_avg, pos_label=pos, zero_division=0) \
            if n_classes == 2 else _safe(f1_score, yt, yp, average="macro", zero_division=0)
        confusion = confusion_matrix(yt, yp, labels=labels).tolist()
        prec, rec, f1s, sup = precision_recall_fscore_support(
            yt, yp, labels=labels, zero_division=0)
        per_class = {
            name: {"n": int(sup[i]), "precision": float(prec[i]), "recall": float(rec[i]),
                   "f1": float(f1s[i])}
            for i, name in enumerate(classes)
        }
    else:
        accuracy = balanced_accuracy = f1 = None
        confusion = [[0] * n_classes for _ in range(n_classes)]
        per_class = {name: {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0} for name in classes}

    # --- positive-vs-rest: automated vs system ---------------------------------------------
    tp_auto = int((is_pos_true & is_pos_pred & auto).sum())
    tn_auto = int((~is_pos_true & ~is_pos_pred & auto).sum())
    pos_auto = int((is_pos_true & auto).sum())
    neg_auto = int((~is_pos_true & auto).sum())
    pos_routed = int((is_pos_true & routed).sum())
    neg_routed = int((~is_pos_true & routed).sum())
    sens_auto = _ratio(tp_auto, pos_auto)
    spec_auto = _ratio(tn_auto, neg_auto)
    sens_sys = _ratio(tp_auto + pos_routed, pos_auto + pos_routed)
    spec_sys = _ratio(tn_auto + neg_routed, neg_auto + neg_routed)
    f1_pos = _ratio(2 * tp_auto, 2 * tp_auto + (pos_auto - tp_auto) + (neg_auto - tn_auto))

    # --- ranking metrics on the positive-class probability (automated tiles) -------------
    y_bin = (yt == pos).astype(int)
    auroc = None
    auroc_macro = None
    if n_auto and y_bin.min() != y_bin.max():
        auroc = _safe(roc_auc_score, y_bin, pr[:, pos])
    if n_classes > 2 and n_auto and len(np.unique(yt)) == n_classes:
        auroc_macro = _safe(roc_auc_score, yt, pr, multi_class="ovr", average="macro",
                            labels=labels)
    fpr, tpr, thr = roc_points(preds, dataset)
    sens_at, thr_at = {}, {}
    for s in specificity_points:
        key = f"{s:.2f}"
        if auroc is None:
            sens_at[key], thr_at[key] = None, None
        else:
            se, th = sensitivity_at_specificity(fpr, tpr, thr, s)
            sens_at[key], thr_at[key] = se, th

    # --- latency / cost / routing ---------------------------------------------------------
    lat = np.array([p.latency_ms for p in preds], dtype=np.float64)
    total_cost = float(sum(p.cost_usd for p in preds))
    cost_per_1k = total_cost / n * 1000.0
    cost_per_slide = cost_per_1k * tiles_per_slide / 1000.0
    coverage = n_auto / n
    routed_frac = n_routed / n
    pathologist_minutes = routed_frac * tiles_per_slide * PATHOLOGIST_SEC_PER_TILE / 60.0

    first = preds[0]
    return {
        "model": first.model,
        "stage": first.stage,
        "dataset": dataset,
        "split": first.split,
        "n": n,
        "n_automated": n_auto,
        "n_routed": n_routed,
        "n_positive": int(is_pos_true.sum()),
        "auroc": auroc,
        "auroc_tumor_vs_rest": auroc,
        "auroc_macro": auroc_macro if n_classes > 2 else auroc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "f1_positive": f1_pos,
        "sensitivity": sens_sys,
        "specificity": spec_sys,
        "sensitivity_automated": sens_auto,
        "specificity_automated": spec_auto,
        "sensitivity_system": sens_sys,
        "specificity_system": spec_sys,
        "sensitivity_at_spec": sens_at,
        "threshold_at_spec": thr_at,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_mean_ms": float(lat.mean()),
        "latency_total_ms": float(lat.sum()),
        "cost_total_usd": total_cost,
        "cost_per_1k_usd": cost_per_1k,
        "cost_per_slide_usd": cost_per_slide,
        "tiles_per_slide": int(tiles_per_slide),
        "coverage": coverage,
        "fraction_routed": routed_frac,
        "pathologist_minutes_per_slide": pathologist_minutes,
        "confusion": confusion,
        "per_class": per_class,
        "classes": list(classes),
        "positive_class": classes[pos],
    }


__all__ = ["compute_metrics", "is_routed", "roc_points", "sensitivity_at_specificity"]
