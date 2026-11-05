"""Nested base fits and out-of-fold error labels."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from topovital.cohort import BlockedError
from topovital_analysis.factorial import _arrays, factorial_weights, threshold_weighted_pvc, weight_checks
from topovital_analysis.factorial_metrics import stratified_auroc
from topovital_analysis.level2 import _fit_score, choose_candidate
from topovital_analysis.level3 import KEYS, _classical_blocks


def subject_rows(windows, subjects):
    """Preserve the global Level 3 window order."""
    return [i for i, window in enumerate(windows) if window["subject_id"] in subjects]


def forbid_evaluated_subject(training_subjects, evaluation_subject):
    if evaluation_subject in training_subjects:
        raise BlockedError("evaluated subject occurs in base training subjects")


def forbid_final_model(path):
    if "final" + "_models" in Path(path).parts:
        raise BlockedError("full-Development final models are forbidden")


def fit_unit(windows, classical, subjects, c, *, cache_key=None):
    subjects = frozenset(subjects)
    train = subject_rows(windows, subjects)
    if not train:
        raise BlockedError("empty fit unit")
    weights = factorial_weights(*_arrays(windows, train))
    checks = weight_checks(*_arrays(windows, train), weights)
    x, _, columns, medians = _classical_blocks(classical, train, train, "fit_unit", [])
    y = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
    fits = []
    _, estimator = _fit_score(x, y, weights, x[:1], float(c), fits, label="fit_unit")
    return FitUnit(subjects, float(c), estimator, medians, tuple(columns), checks,
                   cache_key, fits[0])


@dataclass
class FitUnit:
    subjects: frozenset[str]
    c: float
    estimator: object
    medians: dict
    columns: tuple[str, ...]
    weight_checks: dict
    cache_key: str | None
    fit_record: dict

    @property
    def id(self):
        return "unit:" + ",".join(sorted(self.subjects)) + ":" + format(self.c, ".17g")

    def matrix(self, classical, indices):
        x = classical.iloc[indices][list(self.columns)].to_numpy(dtype=float).copy()
        for j, name in enumerate(self.columns):
            x[~np.isfinite(x[:, j]), j] = self.medians[name]
        return x

    def scores(self, classical, indices):
        x = self.matrix(classical, indices)
        return self.estimator.predict_proba(x)[:, 1], self.estimator.decision_function(x)


@dataclass
class BaseFit:
    subjects: frozenset[str]
    c: float
    theta: float
    cv_score: float
    unit: FitUnit
    unit_ids: tuple[str, ...]
    cv_scores: dict

    @property
    def id(self):
        return "base:" + ",".join(sorted(self.subjects))

    @property
    def provenance(self):
        return {name: self.subjects for name in
                ("training", "preprocessing", "tuning", "threshold")}

    def predict(self, windows, classical, indices):
        if not 0 < self.theta < 1:
            raise BlockedError("base threshold outside (0,1)")
        for i in indices:
            forbid_evaluated_subject(self.subjects, windows[i]["subject_id"])
        p, z = self.unit.scores(classical, indices)
        d = z - logit(self.theta)
        y_hat = (p >= self.theta).astype(int)
        if not np.array_equal(d >= 0, y_hat == 1):
            raise BlockedError("base decision and probability threshold disagree")
        return pd.DataFrame([{**{key: windows[i][key] for key in KEYS},
                              "y": int(windows[i]["condition"] in ("C1", "C3")),
                              "z": float(z[n]), "p": float(p[n]),
                              "d": float(d[n]), "abs_d": float(abs(d[n])),
                              "y_hat": int(y_hat[n]), "e": int(y_hat[n] !=
                              (windows[i]["condition"] in ("C1", "C3"))),
                              "generating_basefit_id": self.id,
                              "window_index": i,
                              "training_subjects": ",".join(sorted(self.subjects))}
                             for n, i in enumerate(indices)])


class BaseFitStore:
    def __init__(self, windows, classical, c_grid, threshold_target, feature_sha,
                 config_sha, code_sha):
        self.windows, self.classical = windows, classical
        self.c_grid = tuple(float(c) for c in c_grid)
        self.threshold_target = float(threshold_target)
        self.hashes = (feature_sha, config_sha, code_sha)
        self.units = {}
        self.basefits = {}
        self.cache_hits = 0

    def unit(self, subjects, c, *, bypass=False):
        subjects = frozenset(subjects)
        key = (tuple(sorted(subjects)), float(c), *self.hashes)
        if key in self.units and not bypass:
            self.cache_hits += 1
            return self.units[key]
        digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
        fitted = fit_unit(self.windows, self.classical, subjects, c, cache_key=digest)
        if not bypass:
            self.units[key] = fitted
        return fitted

    def basefit(self, subjects, *, bypass=False):
        subjects = frozenset(subjects)
        if subjects in self.basefits and not bypass:
            return self.basefits[subjects]
        frames = {c: [] for c in self.c_grid}
        scores = {c: [] for c in self.c_grid}
        ids = []
        for held in sorted(subjects):
            indices = subject_rows(self.windows, {held})
            for c in self.c_grid:
                unit = self.unit(subjects - {held}, c, bypass=bypass)
                ids.append(unit.id)
                p, _ = unit.scores(self.classical, indices)
                frame = pd.DataFrame([{**{k: self.windows[i][k] for k in KEYS},
                                       "score": float(v)} for i, v in zip(indices, p)])
                frames[c].append(frame)
                scores[c].append(stratified_auroc(frame))
        mean = {c: float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                for c, values in scores.items()}
        chosen = choose_candidate({(None, None, c): value for c, value in mean.items()},
                                  ["smaller_resolution", "larger_sigma", "smaller_C"])[2]
        pooled = pd.concat(frames[chosen], ignore_index=True)
        theta = float(threshold_weighted_pvc(pooled, self.threshold_target))
        if not 0 < theta < 1:
            raise BlockedError("base threshold outside (0,1)")
        final = self.unit(subjects, chosen, bypass=bypass)
        result = BaseFit(subjects, chosen, theta, mean[chosen], final,
                         tuple(dict.fromkeys([*ids, final.id])), mean)
        if not bypass:
            self.basefits[subjects] = result
        return result

    def error_table(self, subjects):
        subjects = frozenset(subjects)
        parts = []
        for held in sorted(subjects):
            base = self.basefit(subjects - {held})
            if held in set().union(*base.provenance.values()):
                raise BlockedError("error row subject leaked into generating BaseFit")
            indices = subject_rows(self.windows, {held})
            part = base.predict(self.windows, self.classical, indices)
            part["table_id"] = "error:" + ",".join(sorted(subjects))
            parts.append(part)
        return pd.concat(parts, ignore_index=True).sort_values("window_index").reset_index(drop=True)
