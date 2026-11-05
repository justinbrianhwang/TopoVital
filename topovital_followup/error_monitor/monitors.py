"""Error monitor fitting and operating thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from topovital_analysis.factorial import _arrays, factorial_weights
from topovital_analysis.level2 import fit_weights


@dataclass
class Monitor:
    model: str
    c: float | None
    estimator: object
    subjects: frozenset[str]
    fallback: str | None = None
    wall_time_s: float = 0.0
    convergence_warnings: int = 0

    def score(self, matrix):
        if self.model == "M0":
            return -np.asarray(matrix, dtype=float).ravel()
        if self.fallback:
            return np.full(len(matrix), float(self.estimator))
        return self.estimator.predict_proba(matrix)[:, 1]


def fit_monitor(model, c, matrix, frame, windows, indices, subjects, cfg):
    started = time.perf_counter()
    indices = frame.window_index.to_numpy(dtype=int)
    subjects = frozenset(subjects)
    if model == "M0":
        return Monitor(model, None, None, subjects)
    y = frame.e.to_numpy(dtype=int)
    weights = factorial_weights(*_arrays(windows, indices))
    if len(np.unique(y)) < 2:
        return Monitor(model, float(c), float(np.average(y, weights=weights)), subjects,
                       "constant_monitor_fallback: single-class training labels", time.perf_counter()-started)
    clf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                        StandardScaler(), LogisticRegression(C=float(c), solver="lbfgs", max_iter=5000))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(matrix, y, logisticregression__sample_weight=fit_weights(weights))
    return Monitor(model, float(c), clf, subjects, None, time.perf_counter()-started,
                   sum(issubclass(w.category, ConvergenceWarning) for w in seen))


def operating_threshold(scores_by_subject, target=0.8):
    observed = np.unique(np.concatenate(list(scores_by_subject.values())))
    for score in observed:
        coverage = np.mean([np.mean(values <= score) for values in scores_by_subject.values()])
        if coverage >= target:
            return float(score)
    return float(observed[-1])
