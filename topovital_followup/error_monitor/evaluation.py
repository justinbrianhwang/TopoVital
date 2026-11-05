"""Count and factorial weighted selective prediction metrics."""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from topovital_analysis.factorial import _arrays, factorial_weights


def accepted_indices(frame, scores, coverage, seed):
    keys = frame[["event_id", "condition", "noise_type", "snr_target"]].fillna("").astype(str)
    hashes = [hashlib.sha256(f"{seed}|tie|".encode() + "|".join(row).encode()).hexdigest()
              for row in keys.itertuples(index=False, name=None)]
    order = np.lexsort((hashes, np.asarray(scores, dtype=float)))
    return order[:max(1, int(np.floor(coverage * len(frame))))]


def detection(frame, scores):
    y = frame.e.to_numpy(dtype=int)
    result = {"prevalence": float(y.mean())}
    if len(np.unique(y)) < 2:
        result.update(auroc=np.nan, auprc=np.nan, reason="one_class")
    else:
        result.update(auroc=float(roc_auc_score(y, scores)),
                      auprc=float(average_precision_score(y, scores)), reason="")
    return result


def risk_rows(frame, scores, windows, indices, coverages, seed):
    indices = frame.window_index.to_numpy(dtype=int)
    weights = factorial_weights(*_arrays(windows, indices))
    rows = []
    for coverage in coverages:
        accepted = accepted_indices(frame, scores, coverage, seed)
        y = frame.e.to_numpy(dtype=int)
        rows.append({"coverage_target": coverage, "coverage_realized": len(accepted) / len(y),
                     "risk": float(y[accepted].mean()),
                     "standardized_risk": float(np.average(y[accepted], weights=weights[accepted])),
                     "standardized_coverage": float(weights[accepted].sum() / weights.sum()),
                     "accepted": len(accepted)})
    return rows


def safety(frame, accepted):
    accepted = np.asarray(accepted, dtype=bool)
    y, pred = frame.y.to_numpy(dtype=int), frame.y_hat.to_numpy(dtype=int)
    result = {f"{c}_coverage": float(accepted[frame.condition.eq(c)].mean()) for c in ("C0", "C1", "C2", "C3")}
    def ratio(num, den):
        return float(num.sum() / den.sum()) if den.sum() else np.nan
    result.update(accepted_pvc_sensitivity=ratio(accepted & (y == 1) & (pred == 1), accepted & (y == 1)),
                  accepted_false_negative_share_of_all_pvc=ratio(accepted & (y == 1) & (pred == 0), y == 1),
                  accepted_false_positive_rate=ratio(accepted & (y == 0) & (pred == 1), accepted & (y == 0)),
                  abstention_rate_of_FP=ratio(~accepted & (y == 0) & (pred == 1), (y == 0) & (pred == 1)),
                  abstention_rate_of_FN=ratio(~accepted & (y == 1) & (pred == 0), (y == 1) & (pred == 0)))
    return result
