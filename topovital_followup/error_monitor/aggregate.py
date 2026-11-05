"""Subject-equal full-run reports from freshly computed outer folds."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from .evaluation import accepted_indices

MODELS = ("M0", "M1", "M2", "M3", "M4")
PAIRS = (("M4", "M2", "primary"), ("M4", "M1", "key_secondary"),
         ("M4", "M0", "descriptive"), ("M4", "M3", "descriptive"),
         ("M2", "M1", "descriptive"))
METRICS = ("R80", "R90", "AURC_0.5_1", "standardized_R80", "operating_risk", "operating_coverage")


def _rows(result, subject):
    csv = result["csv"]
    count = pd.DataFrame(csv["risk_coverage_count"])
    standard = pd.DataFrame(csv["risk_coverage_standardized"])
    operating = pd.DataFrame(csv["operating_threshold_metrics"])
    safety = pd.DataFrame(csv["class_condition_safety"])
    rows = []
    for model in MODELS:
        c = count.loc[count.model == model].sort_values("coverage_target")
        c80, c90 = (c.loc[np.isclose(c.coverage_target, target)].iloc[0] for target in (.8, .9))
        st = standard.loc[(standard.model == model) & np.isclose(standard.coverage_target, .8)].iloc[0]
        op = operating.loc[operating.model == model].iloc[0]
        retro = safety.loc[(safety.model == model) & (safety["set"] == "retrospective_80")].iloc[0]
        rows.append({"subject_id": subject, "model": model, "N_s": len(result["predictions"]),
                     "k_s": int(c80.accepted), "coverage_80": c80.coverage_realized,
                     "coverage_90": c90.coverage_realized, "R80": c80.risk, "R90": c90.risk,
                     "AURC_0.5_1": float(np.trapezoid(c.risk, c.coverage_target)),
                     "standardized_R80": st.standardized_risk,
                     "standardized_coverage": st.standardized_coverage,
                     "error_auroc": c80.auroc, "error_auprc": c80.auprc,
                     "error_metric_reason": c80.reason, "error_prevalence": c80.prevalence,
                     "operating_threshold": op.threshold, "operating_coverage": op.coverage,
                     "operating_risk": op.risk,
                     **{key: retro[key] for key in retro.index if key not in ("model", "set")},
                     **{f"operating_{key}": op[key] for key in op.index if key not in ("model", "threshold", "coverage", "risk")}})
    return rows


def bootstrap(metrics, cfg):
    subjects = sorted(metrics.subject_id.unique())
    draws = np.random.default_rng(cfg["full_run"]["bootstrap"]["seed"]).integers(
        0, len(subjects), size=(cfg["full_run"]["bootstrap"]["resamples"], len(subjects)))
    indexed = metrics.set_index(["subject_id", "model"])
    rows = []
    for a, b, flag in PAIRS:
        for metric in METRICS:
            diff = np.array([indexed.loc[(s, a), metric] - indexed.loc[(s, b), metric] for s in subjects], float)
            values = diff[draws]
            usable = np.isfinite(values).all(axis=1)
            samples = values[usable].mean(axis=1)
            lo, hi = np.percentile(samples, [2.5, 97.5]) if len(samples) else (np.nan, np.nan)
            rows.append({"pair": f"{a} - {b}", "flag": flag, "metric": metric,
                         "point_estimate": float(np.mean(diff)) if np.isfinite(diff).all() else np.nan,
                         "ci_lower": lo, "ci_upper": hi, "n_subjects": len(subjects),
                         "n_evaluable_draws": int(usable.sum()), "n_non_evaluable_draws": int((~usable).sum()),
                         "per_subject_differences": json.dumps([float(x) if np.isfinite(x) else None for x in diff])})
    return pd.DataFrame(rows), draws


def _decomposition(frame, accepted, subject, model, set_name):
    rows = []
    groups = {"pvc": frame.y.eq(1), "non_pvc": frame.y.eq(0)}
    groups.update({c: frame.condition.eq(c) for c in ("C0", "C1", "C2", "C3")})
    for group, mask in groups.items():
        subset = frame.loc[mask]
        take = accepted[np.asarray(mask)]
        correct = subset.e.to_numpy() == 0
        labels = ("accepted_correct", "accepted_missed", "abstained_correct", "abstained_missed") if group in ("pvc", "C1", "C3") else (
            "accepted_correct", "accepted_false_positive", "abstained_correct", "abstained_false_positive")
        shares = [np.mean(take & correct), np.mean(take & ~correct), np.mean(~take & correct), np.mean(~take & ~correct)]
        rows.append({"subject_id": subject, "model": model, "set": set_name, "class_group": group,
                     "n_windows": len(subset), **dict(zip(labels, shares))})
    return rows


def _swap(frame, masks, subject, a, b):
    a_only, b_only = masks[a] & ~masks[b], masks[b] & ~masks[a]
    if a_only.sum() != b_only.sum():
        raise BlockedError("swap set sizes differ")
    delta = (frame.e.to_numpy()[a_only].sum() - frame.e.to_numpy()[b_only].sum()) / masks[a].sum()
    risk_delta = frame.e.to_numpy()[masks[a]].mean() - frame.e.to_numpy()[masks[b]].mean()
    if not np.isclose(delta, risk_delta, atol=1e-12, rtol=0):
        raise BlockedError("swap risk identity failed")
    rows = []
    for side, mask in (("A_only", a_only), ("B_only", b_only)):
        for row in frame.loc[mask].itertuples():
            rows.append({"pair": f"{a} vs {b}", "subject_id": subject, "side": side,
                         "event_id": row.event_id, "condition": row.condition,
                         "noise_type": row.noise_type, "snr_target": row.snr_target,
                         "y": row.y, "y_hat": row.y_hat, "e": row.e,
                         "error_type": "correct" if not row.e else ("FP" if row.y == 0 else "FN")})
    return rows


def aggregate_folds(folds, cfg, output_dir, *, pilot=None):
    """Write reports; pilot is comparison evidence only and is never included."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(folds) != len(set(folds)):
        raise BlockedError("duplicate outer subject")
    metrics, decomposition, swaps = [], [], []
    for subject, result in sorted(folds.items()):
        frame, pred = result["outer_frame"], result["predictions"]
        if set(frame.subject_id) != {subject}:
            raise BlockedError("outer subject mismatch")
        metrics.extend(_rows(result, subject))
        for model in MODELS:
            scores = pred[f"{model}_score"].to_numpy()
            take = np.zeros(len(frame), bool)
            take[accepted_indices(frame, scores, .8, cfg["seeds"]["base"])] = True
            decomposition.extend(_decomposition(frame, take, subject, model, "retrospective_80"))
            decomposition.extend(_decomposition(frame, scores <= pred[f"{model}_operating_threshold"].iloc[0], subject, model, "operating"))
            if model == "M0":
                masks = {}
            masks[model] = take
        for a, b in (("M4", "M2"), ("M4", "M1")):
            swaps.extend(_swap(frame, masks, subject, a, b))
    metrics = pd.DataFrame(metrics)
    if metrics.duplicated(["subject_id", "model"]).any():
        raise BlockedError("subject aggregated twice")
    metrics.to_csv(output_dir / "per_subject_metrics.csv", index=False, lineterminator="\n")
    numeric = metrics.select_dtypes(include="number").columns
    summary = metrics.groupby("model", as_index=False)[list(numeric)].mean()
    for metric in ("error_auroc", "error_auprc"):
        summary[f"{metric}_n_evaluable"] = metrics.groupby("model")[metric].count().reindex(summary.model).to_numpy()
    summary.to_csv(output_dir / "summary_metrics.csv", index=False, lineterminator="\n")
    paired, _ = bootstrap(metrics, cfg)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False, lineterminator="\n")
    dec = pd.DataFrame(decomposition)
    means = dec.groupby(["model", "set", "class_group"], as_index=False)[list(dec.select_dtypes(include="number").columns)].mean()
    means["subject_id"] = "MEAN"
    pd.concat([dec, means], ignore_index=True).to_csv(output_dir / "pvc_decomposition.csv", index=False, lineterminator="\n")
    swap = pd.DataFrame(swaps, columns=["pair", "subject_id", "side", "event_id", "condition", "noise_type", "snr_target", "y", "y_hat", "e", "error_type"])
    swap.to_parquet(output_dir / "swap_windows.parquet", index=False)
    swap["signed_error"] = swap.e * np.where(swap.side == "A_only", 1, -1)
    groups = [["pair", "subject_id"], ["pair", "subject_id", "error_type"],
              ["pair", "subject_id", "condition"], ["pair", "subject_id", "noise_type", "snr_target"]]
    pd.concat([swap.groupby(keys, dropna=False).signed_error.sum().reset_index(name="net_error_change").assign(group_by="|".join(keys))
               for keys in groups], ignore_index=True).to_csv(output_dir / "swap_summary.csv", index=False, lineterminator="\n")
    events = swap.groupby(["pair", "subject_id", "event_id"], dropna=False).signed_error.sum().reset_index(name="net_error_change")
    events = events.sort_values(["pair", "net_error_change"], key=lambda col: col.abs() if col.name == "net_error_change" else col, ascending=[True, False])
    events["cumulative_abs_share"] = events.groupby("pair").net_error_change.transform(lambda x: x.abs().cumsum() / x.abs().sum() if x.abs().sum() else np.nan)
    events.to_csv(output_dir / "swap_events.csv", index=False, lineterminator="\n")
    concentration = []
    for pair, part in events.groupby("pair"):
        total = part.net_error_change.sum()
        for n in (1, 3, 5):
            concentration.append({"pair": pair, "scope": f"top_{n}_events", "net_error_change": part.head(n).net_error_change.sum(),
                                  "share_of_net_change": part.head(n).net_error_change.sum() / total if total else np.nan})
        for subject, value in part.groupby("subject_id").net_error_change.sum().items():
            concentration.append({"pair": pair, "scope": f"subject:{subject}", "net_error_change": value,
                                  "share_of_net_change": value / total if total else np.nan})
    pd.DataFrame(concentration).to_csv(output_dir / "swap_concentration.csv", index=False, lineterminator="\n")
    return metrics
