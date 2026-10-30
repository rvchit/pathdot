"""Linear probe on cached embeddings (CONTRACTS.md section 3).

StandardScaler + LogisticRegression (lbfgs, max_iter=2000). When `calibrate=True` and there are at
least `MIN_CALIBRATION_N` samples, 20% of the training data is held out (stratified) and a per-class
isotonic map is fitted on it (one-vs-rest, renormalised, which is what sklearn's
CalibratedClassifierCV(method="isotonic") does); otherwise calibration is skipped and the model is
fitted on everything.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathfrontier.schema import CLASSES, Prediction

MIN_CALIBRATION_N = 200
CALIBRATION_FRACTION = 0.2


class LinearProbe:
    def __init__(self, encoder_name: str, dataset: str, C: float = 1.0, calibrate: bool = True,
                 class_weight: str | dict | None = "balanced", seed: int = 0) -> None:
        if dataset not in CLASSES:
            raise ValueError(f"unknown dataset '{dataset}'; choose from {list(CLASSES)}")
        self.encoder_name = encoder_name
        self.dataset = dataset
        self.C = C
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.seed = seed
        self.classes = list(CLASSES[dataset])
        self.n_classes = len(self.classes)
        self.scaler: StandardScaler | None = None
        self.clf: LogisticRegression | None = None
        self.calibrators: dict[int, IsotonicRegression] | None = None  # class index -> isotonic map
        self.n_train = 0
        self.n_calib = 0

    @property
    def model_name(self) -> str:
        return f"probe/{self.encoder_name}"

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrators)

    # -- training -------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearProbe:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"bad shapes X{X.shape} y{y.shape}")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"labels must be in [0, {self.n_classes})")

        X_fit, y_fit, X_cal, y_cal = X, y, None, None
        can_calibrate = self.calibrate and len(y) >= MIN_CALIBRATION_N and len(np.unique(y)) > 1
        if can_calibrate:
            counts = np.bincount(y, minlength=self.n_classes)
            strat = y if counts[counts > 0].min() >= 2 else None
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X, y, test_size=CALIBRATION_FRACTION, random_state=self.seed, stratify=strat)

        self.scaler = StandardScaler().fit(X_fit)
        self.clf = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000,
                                      class_weight=self.class_weight)
        self.clf.fit(self.scaler.transform(X_fit), y_fit)
        self.n_train = len(y_fit)
        self.calibrators = None
        self.n_calib = 0

        if can_calibrate:
            raw = self._raw_proba(X_cal)
            cals: dict[int, IsotonicRegression] = {}
            targets = [self.n_classes - 1] if self.n_classes == 2 else range(self.n_classes)
            for k in targets:
                yk = (y_cal == k).astype(np.float64)
                if yk.sum() == 0 or yk.sum() == len(yk):
                    continue  # degenerate class on the holdout: leave this class uncalibrated
                cals[k] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                    raw[:, k], yk)
            self.calibrators = cals or None
            self.n_calib = len(y_cal)
        return self

    # -- inference ------------------------------------------------------------------------------
    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        """LogisticRegression probabilities expanded to all dataset classes (zeros for unseen)."""
        assert self.clf is not None and self.scaler is not None, "probe is not fitted"
        p = self.clf.predict_proba(self.scaler.transform(np.asarray(X, dtype=np.float32)))
        out = np.zeros((len(p), self.n_classes), dtype=np.float64)
        out[:, self.clf.classes_] = p
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_proba(X)
        if not self.calibrators:
            return raw
        p = raw.copy()
        for k, iso in self.calibrators.items():
            p[:, k] = iso.predict(raw[:, k])
        if self.n_classes == 2:
            k = self.n_classes - 1
            p[:, 0] = 1.0 - p[:, k]
        else:
            s = p.sum(axis=1, keepdims=True)
            p = np.where(s > 0, p / np.maximum(s, 1e-12), raw)
        return np.clip(p, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_records(self, X: np.ndarray, y: np.ndarray, ids: list[str], split: str,
                        latency_ms_per_tile: float, cost_per_tile: float) -> list[Prediction]:
        probs = self.predict_proba(X)
        y = np.asarray(y, dtype=np.int64)
        if not (len(probs) == len(y) == len(ids)):
            raise ValueError("X, y and ids must have the same length")
        preds = []
        for i in range(len(ids)):
            row = probs[i].astype(float).tolist()
            preds.append(Prediction(
                tile_id=str(ids[i]), dataset=self.dataset, split=split, y_true=int(y[i]),
                y_prob=row, y_pred=int(np.argmax(row)), model=self.model_name, stage="probe",
                latency_ms=float(latency_ms_per_tile), cost_usd=float(cost_per_tile),
                extra={"calibrated": self.is_calibrated, "C": self.C},
            ))
        return preds

    # -- persistence ----------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: Path) -> LinearProbe:
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj
