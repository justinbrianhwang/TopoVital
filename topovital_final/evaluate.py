"""Frozen Test metrics and paired model comparisons."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.util import atomic_write_csv
from topovital_analysis.factorial_metrics import metric_tables
from topovital_analysis.level3 import model_comparison


WINDOW_KEYS = ["event_id", "subject_id", "condition", "noise_type", "snr_target"]
PAIR_RULES = (("paired_change_normal", "C0", "C2"),
              ("paired_change_pvc", "C1", "C3"))


def _orphan_pairs(predictions: pd.DataFrame, base: str, changed: str) -> list:
    identity = ["event_id", "subject_id", "model"]
    bases = set(map(tuple, predictions.loc[predictions.condition.eq(base), identity].itertuples(
        index=False, name=None)))
    altered = predictions.loc[predictions.condition.eq(changed), identity]
    return [index for index, row in altered.iterrows() if tuple(row) not in bases]


def _replace_metric_rows(original: pd.DataFrame, replacement: pd.DataFrame,
                         metric: str) -> pd.DataFrame:
    mask = original.metric.eq(metric)
    other = replacement.loc[replacement.metric.eq(metric)]
    if len(other) != int(mask.sum()):
        from topovital.cohort import BlockedError
        raise BlockedError(f"Paired metric schema changed: {metric}")
    result = original.copy()
    result.loc[mask, other.columns] = other.to_numpy()
    return result


def _metric_tables_with_pairs(predictions: pd.DataFrame, cfg: dict) -> tuple:
    tables = metric_tables(predictions, cfg)
    for metric, base, changed in PAIR_RULES:
        orphans = _orphan_pairs(predictions, base, changed)
        if orphans:
            reduced = predictions.drop(index=orphans)
            replacement = metric_tables(reduced, cfg)
            tables = tuple(_replace_metric_rows(old, new, metric)
                           for old, new in zip(tables, replacement))
    return tables


def common_evaluable(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Keep precisely the windows scored by every frozen model."""
    models = sorted(predictions.model.unique())
    if len(models) != 5:
        from topovital.cohort import BlockedError
        raise BlockedError("Common evaluable set requires five models")
    grouped = predictions.groupby(WINDOW_KEYS, dropna=False, sort=False)
    common_indices = []
    summary_rows = []
    excluded_by_subject = {}
    common_count = 0
    for _, group in grouped:
        if len(group) != 5 or set(group.model) != set(models):
            from topovital.cohort import BlockedError
            raise BlockedError("Prediction model rows differ for a window")
        finite = np.isfinite(group.score.to_numpy(dtype=float))
        ok = bool(finite.all())
        common_count += int(ok)
        if ok:
            common_indices.extend(group.index.tolist())
        else:
            subject = str(group.subject_id.iloc[0])
            excluded_by_subject[subject] = excluded_by_subject.get(subject, 0) + 1
            reasons = sorted(set(str(reason) for reason in group.loc[~finite, "reason"]
                                 if pd.notna(reason) and str(reason)))
            summary_rows.append({**{k: group[k].iloc[0] for k in WINDOW_KEYS},
                                 "row_type": "excluded_window", "total_windows": 1,
                                 "common_evaluable_windows": 0,
                                 "reason": ";".join(reasons) or "nonfinite_score",
                                 **{f"finite_{model}": int(group.loc[group.model.eq(model), "score"].map(np.isfinite).iloc[0])
                                    for model in models}})
    window_counts = (predictions.assign(finite=np.isfinite(predictions.score.to_numpy(dtype=float)))
                     .groupby(["subject_id", "condition", "noise_type", "snr_target"],
                              dropna=False, sort=False))
    common_index_set = set(common_indices)
    for _, group in window_counts:
        first = group.iloc[0]
        windows = group.groupby("event_id", dropna=False)
        summary_rows.append({"row_type": "count", "event_id": "",
                             **{k: first[k] for k in ("subject_id", "condition", "noise_type", "snr_target")},
                             "total_windows": len(windows),
                             "common_evaluable_windows": sum(all(i in common_index_set for i in g.index)
                                                             for _, g in windows),
                             "reason": "",
                             **{f"finite_{model}": int(group.loc[group.model.eq(model), "finite"].sum())
                                for model in models}})
    summary = pd.DataFrame(summary_rows)
    counts = {"common_evaluable_windows": common_count,
              "excluded_windows": len(grouped) - common_count,
              "excluded_by_subject": excluded_by_subject}
    return predictions.loc[sorted(common_indices)].copy(), summary, counts


def evaluate(predictions: pd.DataFrame, settings: dict, cfg_l3: dict,
             cfg_f: dict, cfg_a: dict, output: Path) -> dict:
    metric_cfg = copy.deepcopy(cfg_f)
    metric_cfg["scope"] = cfg_a["scope"]
    metric_cfg["reporting"]["bootstrap"] = {"kind": "subject_cluster", **settings["bootstrap"]}
    cfg = copy.deepcopy(cfg_l3)
    cfg["comparisons"]["primary"] = settings["primary_comparison"]
    cfg["comparisons"]["secondary"] = settings["secondary_comparisons"]
    common, evaluable_summary, counts = common_evaluable(predictions)
    atomic_write_csv(Path(output) / "evaluable_set_summary.csv", evaluable_summary)
    pair_exclusions = {}
    for metric, base, changed in PAIR_RULES:
        orphans = _orphan_pairs(common, base, changed)
        pair_exclusions[metric] = len(orphans) // 5
    counts["metric_specific_pair_exclusions"] = pair_exclusions
    tables = _metric_tables_with_pairs(common, metric_cfg)
    comparison = model_comparison(common, cfg, metric_cfg, cfg_a)
    for metric, base, changed in PAIR_RULES:
        orphans = _orphan_pairs(common, base, changed)
        if orphans:
            replacement = model_comparison(common.drop(index=orphans), cfg, metric_cfg, cfg_a)
            comparison = _replace_metric_rows(comparison, replacement, metric)
    comparison["status"] = np.where(np.isfinite(comparison["difference"]), "ok", "not_evaluable")
    comparison["reason"] = np.where(comparison["status"].eq("ok"), "", "missing_class_or_score")
    comparison.loc[comparison.status.eq("ok") & comparison.ci_low.isna(), "status"] = "not_evaluable"
    comparison.loc[comparison.status.eq("not_evaluable") & comparison.reason.eq(""), "reason"] = "bootstrap_undefined_over_5_percent"
    result = {}
    for name, frame in zip(("metrics_window", "metrics_subject", "per_subject_metrics",
                            "bootstrap_ci"), tables):
        atomic_write_csv(Path(output) / f"{name}.csv", frame)
        result[name] = len(frame)
    available = _metric_tables_with_pairs(
        predictions.loc[np.isfinite(predictions.score.to_numpy(dtype=float))], metric_cfg)
    for name, frame in zip(("metrics_window", "metrics_subject", "per_subject_metrics",
                            "bootstrap_ci"), available):
        atomic_write_csv(Path(output) / "metrics_all_available" / f"{name}.csv", frame)
    atomic_write_csv(Path(output) / "paired_model_comparison.csv", comparison)
    result["paired_model_comparison"] = len(comparison)
    result.update(counts)
    return result
