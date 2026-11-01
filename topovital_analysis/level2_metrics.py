"""Out-of-fold metrics, subject cluster intervals, and exploratory MMD."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from .stats import signed_auroc


def metric_value(frame, metric):
    if metric == "signed_auroc":
        return signed_auroc(frame.loc[frame.condition.eq("C1"), "score"],
                            frame.loc[frame.condition.eq("C2"), "score"])
    condition = {"c1_sensitivity": "C1", "c2_fpr": "C2", "c3_sensitivity": "C3", "c0_fpr": "C0"}[metric]
    rows = frame.loc[frame.condition.eq(condition)]
    return float(np.mean(rows.score.to_numpy() >= rows.threshold.to_numpy())) if len(rows) else np.nan


def metric_tables(predictions, cfg):
    """One row per model, cell or protocol aggregate, and metric."""
    window_rows, subject_rows, per_subject, ci_rows = [], [], [], []
    reporting = cfg["level2"]["reporting"]
    boot = reporting["bootstrap"]
    cells = [(n, s) for n in cfg["scope"]["regime1_cells"]["noise_types"]
             for s in cfg["scope"]["regime1_cells"]["snr_db"]]
    protocol = {(n, s) for n, s in cells if s in (12, 6, 0)}
    for model, model_frame in predictions.groupby("model", sort=True):
        for noise, snr in [("overall", "all"), *cells, ("protocol_9_cell", "12_6_0")]:
            selected_cells = cells if noise == "overall" else (list(protocol) if noise == "protocol_9_cell" else [(noise, snr)])
            clean = model_frame.loc[model_frame.condition.isin(("C0", "C1"))].drop_duplicates(
                ["event_id", "condition", "subject_id"])
            injected = model_frame.loc[model_frame.condition.isin(("C2", "C3")) &
                                   model_frame.apply(lambda row: (row.noise_type, row.snr_target) in selected_cells, axis=1)]
            frame = pd.concat((clean, injected), ignore_index=True)
            for metric in ("signed_auroc", "c1_sensitivity", "c2_fpr", "c3_sensitivity", "c0_fpr"):
                if noise == "protocol_9_cell":
                    # Equal weight for the nine cells within each subject, then across subjects.
                    by_subject = []
                    for subject, sg in frame.groupby("subject_id", sort=True):
                        values = []
                        for cell in sorted(protocol):
                            subset = pd.concat((sg.loc[sg.condition.isin(("C0", "C1"))],
                                                sg.loc[sg.condition.isin(("C2", "C3")) &
                                                       sg.noise_type.eq(cell[0]) & sg.snr_target.eq(cell[1])]))
                            values.append(metric_value(subset, metric))
                        by_subject.append((subject, float(np.mean(values)) if np.isfinite(values).all() else np.nan))
                    subject_values = dict(by_subject)
                    window_value = float(np.mean(list(subject_values.values()))) if subject_values and np.isfinite(list(subject_values.values())).all() else np.nan
                else:
                    window_value = metric_value(frame, metric)
                    subject_values = {subject: metric_value(sg, metric) for subject, sg in frame.groupby("subject_id", sort=True)}
                values = np.array(list(subject_values.values()), dtype=float)
                valid = values[np.isfinite(values)]
                subject_value = float(np.mean(valid)) if len(valid) else np.nan
                key = {"model": model, "noise_type": noise, "snr_target": snr, "metric": metric}
                for subject, value in subject_values.items():
                    per_subject.append({**key, "subject_id": subject, "value": value,
                                        "status": "ok" if np.isfinite(value) else "not_evaluable",
                                        "reason": "" if np.isfinite(value) else "missing_class_or_score"})
                window_rows.append({**key, "value": window_value, "n_windows": len(frame),
                                    "status": "ok" if np.isfinite(window_value) else "not_evaluable",
                                    "reason": "" if np.isfinite(window_value) else "missing_class_or_score"})
                subject_rows.append({**key, "value": subject_value, "n_subjects": len(valid),
                                     "status": "ok" if len(valid) else "not_evaluable",
                                     "reason": "" if len(valid) else "no_evaluable_subjects"})
                rng = np.random.default_rng(int(boot["seed"]))
                draws = []
                for _ in range(int(boot["resamples"])):
                    draws.append(float(np.mean(rng.choice(values, size=len(values), replace=True))) if len(values) else np.nan)
                draws = np.asarray(draws)
                finite = draws[np.isfinite(draws)]
                undefined = len(draws) - len(finite)
                ok = len(draws) > 0 and undefined / len(draws) <= .05 and len(finite) > 0
                ci_rows.append({**key, "ci_low": float(np.quantile(finite, .025)) if ok else np.nan,
                                "ci_high": float(np.quantile(finite, .975)) if ok else np.nan,
                                "undefined_resamples": undefined, "status": "ok" if ok else "not_evaluable",
                                "reason": "" if ok else "bootstrap_undefined_over_5_percent"})
    return tuple(map(pd.DataFrame, (window_rows, subject_rows, per_subject, ci_rows)))


def unbiased_mmd2(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    pooled = np.vstack((x, y))
    distances = pdist(pooled, metric="sqeuclidean")
    positive = distances[distances > 0]
    if not len(positive):
        return 0.0, 0.0
    bandwidth2 = float(np.median(positive))
    kxx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * bandwidth2))
    kyy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * bandwidth2))
    kxy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * bandwidth2))
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) +
                 (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()), bandwidth2
