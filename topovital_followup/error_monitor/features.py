"""Fold-local monitor matrices and feature guard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital_analysis.level3 import KEYS
from topovital_analysis.persistence_images import estimate_regions, out_of_range, pi_features

SCORE = ("d", "abs_d", "p", "y_hat")


def check_features(names, forbidden):
    banned = set(forbidden) | set(KEYS) | {"y", "e", "z", "training_subjects", "generating_basefit_id"}
    bad = [name for name in names if name in banned or name.startswith("counterfactual_")]
    if bad:
        raise BlockedError(f"forbidden monitor features: {bad}")


def pi_matrix(windows, subjects, indices, cfg):
    source = [i for i, w in enumerate(windows) if w["subject_id"] in subjects and w["condition"] in ("C0", "C1")]
    reps = cfg["representations"]
    regions = estimate_regions(windows, source, reps, cfg["range_quantiles"])
    counts = {rep: sum(out_of_range(windows[i][half][rep], regions[rep])[0]
                       for i in indices for half in ("pre", "post")) for rep in reps}
    matrix = np.asarray([pi_features(windows[i], regions, int(cfg["resolution"]),
                                    float(cfg["sigma_pixels"]), reps) for i in indices])
    names = [f"pi__{j}" for j in range(matrix.shape[1])]
    return matrix, names, regions, counts


def monitor_matrix(model, frame, classical, windows, indices, cfg, region_subjects=None):
    indices = frame.window_index.to_numpy(dtype=int)
    if any(not (frame.iloc[n][key] == windows[i][key] or
                (pd.isna(frame.iloc[n][key]) and pd.isna(windows[i][key])))
           for n, i in enumerate(indices) for key in KEYS):
        raise BlockedError("monitor frame and window keys are misaligned")
    parts, names = [], []
    if model in ("M1", "M2", "M4"):
        score = list(SCORE)
        cols = [c for c in classical if c not in KEYS]
        if model == "M1":
            cols = [c for c in cols if c.startswith("sqi__") or c.startswith("amplitude__")]
        names.extend(score + cols)
        parts.extend([frame[score].to_numpy(dtype=float), classical.iloc[indices][cols].to_numpy(dtype=float)])
    regions, counts = None, {}
    if model in ("M3", "M4"):
        pi, pi_names, regions, counts = pi_matrix(windows, region_subjects, indices, cfg)
        names.extend(pi_names)
        parts.append(pi)
    check_features(names, cfg.get("forbidden_inputs", []))
    return np.column_stack(parts), names, regions, counts
