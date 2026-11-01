"""Descriptive separation and subject-aware uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata


def signed_auroc(positive, negative) -> float:
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    ranks = rankdata(np.concatenate((a, b)), method="average")
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def direction_free(auroc: float) -> float:
    return float(2 * abs(auroc - 0.5)) if np.isfinite(auroc) else np.nan


def subject_scores(frame: pd.DataFrame, positive: str, negative: str):
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        a = group.loc[group.condition.eq(positive), "score"].to_numpy(dtype=float)
        b = group.loc[group.condition.eq(negative), "score"].to_numpy(dtype=float)
        auc = signed_auroc(a, b)
        rows.append({"subject_id": subject, "n_c1": len(a), "n_c2": len(b),
                     "signed_auroc": auc, "direction_free_sep": direction_free(auc),
                     "ks_d": float(ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan,
                     "status": "ok" if np.isfinite(auc) else "not_evaluable",
                     "reason": "" if np.isfinite(auc) else "missing_class_or_nonfinite_score"})
    return pd.DataFrame(rows)


def loso_threshold(training: pd.DataFrame, positive: str, target_sensitivity: float = 0.90) -> float:
    positives = training.loc[training.condition.eq(positive)]
    if positives.empty or positives.score.isna().any():
        return np.nan
    groups = [group.score.to_numpy(dtype=float) for _, group in positives.groupby("subject_id")]
    if not groups or not all(len(group) for group in groups):
        return np.nan
    for threshold in np.sort(positives.score.unique())[::-1]:
        fraction = np.mean([np.mean(group >= threshold) for group in groups])
        if fraction >= target_sensitivity:
            return float(threshold)
    return np.nan


def loso_rates(frame: pd.DataFrame, positive: str, negative: str, target_sensitivity: float = 0.90):
    rows = []
    for subject in sorted(frame.subject_id.unique()):
        held = frame.loc[frame.subject_id.eq(subject)]
        train = frame.loc[~frame.subject_id.eq(subject)]
        threshold = loso_threshold(train, positive, target_sensitivity)
        a = held.loc[held.condition.eq(positive), "score"].to_numpy(dtype=float)
        b = held.loc[held.condition.eq(negative), "score"].to_numpy(dtype=float)
        valid = len(a) > 0 and len(b) > 0 and np.isfinite(threshold) and np.isfinite(a).all() and np.isfinite(b).all()
        rows.append({"subject_id": subject, "threshold": threshold,
                     "loso_c1_sensitivity": float(np.mean(a >= threshold)) if valid else np.nan,
                     "loso_c2_fpr": float(np.mean(b >= threshold)) if valid else np.nan,
                     "n_c1": len(a), "n_c2": len(b),
                     "status": "ok" if valid else "not_evaluable",
                     "reason": "" if valid else "missing_class_or_threshold"})
    return pd.DataFrame(rows)


def bootstrap_auroc(frame: pd.DataFrame, positive: str, negative: str,
                    resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = {s: group for s, group in frame.groupby("subject_id", sort=True)}
    names = sorted(groups)
    window, subject = [], []
    for _ in range(resamples):
        draws = rng.choice(names, size=len(names), replace=True)
        a, b, per_subject = [], [], []
        for name in draws:
            group = groups[name]
            aa = group.loc[group.condition.eq(positive), "score"].to_numpy(dtype=float)
            bb = group.loc[group.condition.eq(negative), "score"].to_numpy(dtype=float)
            a.extend(aa)
            b.extend(bb)
            per_subject.append(signed_auroc(aa, bb))
        window.append(signed_auroc(a, b))
        subject.append(float(np.mean(per_subject)) if np.isfinite(per_subject).all() else np.nan)
    result = {}
    for name, values in (("window", window), ("subject", subject)):
        values = np.asarray(values)
        good = values[np.isfinite(values)]
        undefined = int(len(values) - len(good))
        result[f"{name}_bootstrap_undefined"] = undefined
        result[f"{name}_ci_low"] = float(np.percentile(good, 2.5)) if len(good) and undefined / len(values) <= .05 else np.nan
        result[f"{name}_ci_high"] = float(np.percentile(good, 97.5)) if len(good) and undefined / len(values) <= .05 else np.nan
        result[f"{name}_bootstrap_status"] = "not_evaluable" if not len(values) or undefined / len(values) > .05 else "ok"
    return result


def separation(frame: pd.DataFrame, positive: str, negative: str,
               target_sensitivity: float, resamples: int, seed: int):
    subjects = subject_scores(frame, positive, negative)
    folds = loso_rates(frame, positive, negative, target_sensitivity)
    detail = subjects.merge(folds, on="subject_id", suffixes=("", "_fold"))
    a = frame.loc[frame.condition.eq(positive), "score"].to_numpy(dtype=float)
    b = frame.loc[frame.condition.eq(negative), "score"].to_numpy(dtype=float)
    auc = signed_auroc(a, b)
    valid_subject = len(subjects) > 0 and subjects.status.eq("ok").all()
    valid_folds = len(folds) > 0 and folds.status.eq("ok").all()
    result = {"signed_auroc_window": auc,
              "signed_auroc_subject": float(subjects.signed_auroc.mean()) if valid_subject else np.nan,
              "direction_free_sep_window": direction_free(auc),
              "direction_free_sep_subject": float(subjects.direction_free_sep.mean()) if valid_subject else np.nan,
              "ks_d_window": float(ks_2samp(a, b).statistic) if len(a) and len(b) and np.isfinite(a).all() and np.isfinite(b).all() else np.nan,
              "ks_d_subject": float(subjects.ks_d.mean()) if valid_subject else np.nan,
              "loso_c1_sensitivity_window": float(np.average(folds.loso_c1_sensitivity, weights=folds.n_c1)) if valid_folds else np.nan,
              "loso_c2_fpr_window": float(np.average(folds.loso_c2_fpr, weights=folds.n_c2)) if valid_folds else np.nan,
              "loso_c1_sensitivity_subject": float(folds.loso_c1_sensitivity.mean()) if valid_folds else np.nan,
              "loso_c2_fpr_subject": float(folds.loso_c2_fpr.mean()) if valid_folds else np.nan,
              "n_c1_windows": len(a), "n_c2_windows": len(b), "n_subjects": len(subjects)}
    result.update(bootstrap_auroc(frame, positive, negative, resamples, seed))
    reasons = []
    if not np.isfinite(auc):
        reasons.append("window_auroc_undefined")
    if not valid_subject:
        reasons.append("subject_missing_class_or_nonfinite_score")
    if not valid_folds:
        reasons.append("loso_fold_undefined")
    for level in ("window", "subject"):
        if result[f"{level}_bootstrap_status"] != "ok":
            reasons.append(f"{level}_bootstrap_undefined_over_5_percent")
    result.update(status="not_evaluable" if reasons else "ok", reason=";".join(reasons))
    return result, detail
