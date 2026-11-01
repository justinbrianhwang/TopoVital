"""Development-only nested factorial C0-C3 PVC analysis."""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import atomic_write_bytes, atomic_write_csv, atomic_write_json, atomic_write_parquet, peak_rss_mb

from .factorial_metrics import comparison_with_level2, metric_tables, stratified_auroc
from .level2 import (_feature_matrix, _fit_score, _load_windows, _range_log, _region_rows,
                     assert_training_subjects, candidate_grid, choose_candidate, fit_weights)
from .level2_metrics import unbiased_mmd2
from .level1 import analysis_code_hash, validate_events
from .persistence_images import estimate_regions
from .stats import loso_threshold


def _key(value):
    return None if value is None or pd.isna(value) else value


def factorial_weights(subjects, conditions, cells, events):
    subjects, conditions, events = map(np.asarray, (subjects, conditions, events))
    cells = [_key(c) if not isinstance(c, (tuple, list)) else tuple(_key(v) for v in c) for c in cells]
    if not len(subjects) or any(len(a) != len(subjects) for a in (conditions, cells, events)):
        raise ValueError("invalid_factorial_rows")
    if not set(conditions) <= {"C0", "C1", "C2", "C3"}:
        raise ValueError("unexpected_condition")
    cell_names = sorted({c for k, c in zip(conditions, cells) if k in ("C2", "C3")})
    if len(cell_names) != 18:
        raise ValueError("expected_18_cells")
    names = sorted(set(subjects))
    weights = np.zeros(len(subjects), dtype=float)
    for name in names:
        sm = subjects == name
        for condition in ("C0", "C1", "C2", "C3"):
            cm = sm & (conditions == condition)
            if not cm.any():
                raise ValueError("training_subject_missing_condition")
            if condition in ("C0", "C1"):
                weights[cm] = 1 / (len(names) * 4 * cm.sum())
            else:
                for cell in cell_names:
                    mask = cm & np.array([c == cell for c in cells])
                    if not mask.any():
                        raise ValueError("training_subject_missing_cell")
                    weights[mask] = 1 / (len(names) * 4 * 18 * mask.sum())
    return weights


def weight_checks(subjects, conditions, cells, events, weights=None):
    subjects, conditions, events = map(np.asarray, (subjects, conditions, events))
    cells = [tuple(c) if isinstance(c, (tuple, list)) else c for c in cells]
    w = factorial_weights(subjects, conditions, cells, events) if weights is None else np.asarray(weights, dtype=float)
    names = sorted(set(subjects))
    cell_names = sorted({c for k, c in zip(conditions, cells) if k in ("C2", "C3")})
    sums = np.array([w[subjects == s].sum() for s in names])
    result = {"total": float(w.sum()), "min_subject_sum": float(sums.min()),
              "max_subject_sum": float(sums.max()), "fit_weight_mean": float(fit_weights(w).mean())}
    for condition in ("C0", "C1", "C2", "C3"):
        result[f"condition_{condition}_sum"] = float(w[conditions == condition].sum())
    for condition in ("C2", "C3"):
        totals = [w[(conditions == condition) & np.array([c == cell for c in cells])].sum() for cell in cell_names]
        result[f"{condition}_min_cell_sum"] = float(min(totals))
        result[f"{condition}_max_cell_sum"] = float(max(totals))
    spreads = []
    for name in names:
        for condition in ("C0", "C1", "C2", "C3"):
            for cell in ([None] if condition in ("C0", "C1") else cell_names):
                mask = (subjects == name) & (conditions == condition)
                if cell is not None:
                    mask &= np.array([c == cell for c in cells])
                values = w[mask]
                spreads.append(float((values.max() - values.min()) / values.mean()))
    result["max_event_relative_spread"] = max(spreads)
    pvc = np.isin(conditions, ("C1", "C3")).astype(float)
    artefact = np.isin(conditions, ("C2", "C3")).astype(float)
    result["weighted_cov_pvc_artefact"] = float(np.sum(w * (pvc - np.sum(w * pvc)) * (artefact - np.sum(w * artefact))))
    assert np.isclose(result["total"], 1) and np.allclose(sums, 1 / len(names))
    assert all(np.isclose(result[f"condition_{k}_sum"], .25) for k in ("C0", "C1", "C2", "C3"))
    assert all(np.isclose(result[f"{k}_{edge}_cell_sum"], 1 / 72) for k in ("C2", "C3") for edge in ("min", "max"))
    assert result["max_event_relative_spread"] < 1e-12 and abs(result["weighted_cov_pvc_artefact"]) < 1e-12
    assert np.isclose(result["fit_weight_mean"], 1)
    return result


def threshold_weighted_pvc(frame, target):
    cells = sorted({(r.noise_type, r.snr_target) for r in frame.itertuples() if r.condition == "C3"})
    if len(cells) != 18 or frame.empty or not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        return np.nan
    score = frame.score.to_numpy(dtype=float)
    condition = frame.condition.to_numpy()
    subjects = frame.subject_id.to_numpy()
    noise = frame.noise_type.to_numpy()
    snr = frame.snr_target.to_numpy()
    positive = np.unique(score[np.isin(condition, ("C1", "C3"))])
    if not len(positive):
        return np.nan
    thresholds = positive[::-1]
    values = []
    names = sorted(frame.subject_id.unique())
    for subject in names:
        subject_mask = subjects == subject
        c1 = np.sort(score[subject_mask & (condition == "C1")])
        if not len(c1):
            return np.nan
        c1_rate = (len(c1) - np.searchsorted(c1, thresholds, side="left")) / len(c1)
        c3_rates = []
        for cell_noise, cell_snr in cells:
            c3 = np.sort(score[subject_mask & (condition == "C3") &
                               (noise == cell_noise) & (snr == cell_snr)])
            if not len(c3):
                return np.nan
            c3_rates.append((len(c3) - np.searchsorted(c3, thresholds, side="left")) / len(c3))
        values.append(.5 * c1_rate + .5 * np.mean(c3_rates, axis=0))
    sensitivity = np.mean(values, axis=0)
    eligible = np.flatnonzero(sensitivity >= target)
    if len(eligible):
        return float(thresholds[eligible[0]])
    return np.nan


def threshold_c1(frame, target):
    return loso_threshold(frame, "C1", target)


def validate_config(cfg_f, cfg_a):
    expected = {"positive": ["C1", "C3"], "negative": ["C0", "C2"], "windows": "all_c0_c3"}
    if cfg_f.get("inherit_from") != "configs/stage1_analysis.yaml" or any(cfg_f.get("training", {}).get(k) != v for k, v in expected.items()):
        raise BlockedError("Unexpected factorial training configuration")
    if cfg_f["training"]["weights"].get("condition") != {k: .25 for k in ("C0", "C1", "C2", "C3")}:
        raise BlockedError("Unexpected factorial condition weights")
    weight_fields = {"subject": "equal", "cell": "equal_within_c2_c3", "event": "equal_within_cell",
                     "fit_weight_scale": "mean_1",
                     "checks": ["total_sums", "subject_sums", "condition_sums", "cell_sums", "event_sums",
                                "weighted_cov_pvc_artefact_zero", "fit_mean_one"]}
    if any(cfg_f["training"]["weights"].get(k) != v for k, v in weight_fields.items()):
        raise BlockedError("Unexpected factorial weight rule")
    if cfg_f["thresholds"]["primary"].get("target") != .9 or cfg_f["thresholds"]["secondary"].get("target") != .9:
        raise BlockedError("Unexpected factorial sensitivity targets")
    if cfg_f["thresholds"]["primary"].get("name") != "weighted_pvc_sensitivity_90" or cfg_f["thresholds"]["secondary"].get("name") != "c1_sensitivity_90":
        raise BlockedError("Unexpected factorial threshold rules")
    if cfg_f["selection"].get("metric") != "stratified_auroc" or cfg_f["selection"].get("scope") != "nested_loso_inner_only":
        raise BlockedError("Unexpected factorial selection rule")
    if cfg_a["scope"].get("conditions") != ["C0", "C1", "C2", "C3"] or cfg_a["scope"].get("split") != "development_only":
        raise BlockedError("Unsupported inherited scope")
    cells = cfg_a["scope"]["regime1_cells"]
    if len(cells["noise_types"]) * len(cells["snr_db"]) != 18:
        raise BlockedError("Expected eighteen inherited cells")


def _arrays(windows, indices):
    selected = [windows[i] for i in indices]
    return ([w["subject_id"] for w in selected], [w["condition"] for w in selected],
            [(w["noise_type"], w["snr_target"]) if w["condition"] in ("C2", "C3") else None for w in selected],
            [w["event_id"] for w in selected])


def _cross_validate_factorial(windows, cfg, subjects, reps, quantiles, fits, region_rows, range_rows, fold_prefix, log=print):
    models = cfg["level2"]["features"]["models"]
    results = {m: {c: [] for c in candidate_grid(cfg, m)} for m in models}
    predictions = {m: {c: [] for c in results[m]} for m in models}
    checks = []
    visible = [i for i, w in enumerate(windows) if w["subject_id"] in subjects]
    for held in subjects:
        fold_start, fit_start = time.perf_counter(), len(fits)
        train = [i for i in visible if windows[i]["subject_id"] != held]
        source = [i for i in train if windows[i]["condition"] in ("C0", "C1")]
        test = [i for i in visible if windows[i]["subject_id"] == held]
        for indices in (train, source):
            assert_training_subjects([windows[i]["subject_id"] for i in indices], held)
        regions = estimate_regions(windows, source, reps, quantiles, excluded_subject=held)
        fold = f"{fold_prefix}__inner_{held}" if fold_prefix.startswith("outer_") else f"{fold_prefix}__held_{held}"
        region_rows.extend(_region_rows(regions, fold))
        range_rows.extend(_range_log(windows, visible, regions, fold))
        args = _arrays(windows, train)
        weights = factorial_weights(*args)
        checks.append({"fold": fold, "held_out_subject": held, "n_rows": len(train), **weight_checks(*args, weights)})
        y = np.array([windows[i]["condition"] in ("C1", "C3") for i in train])
        amp_train = _feature_matrix(windows, train, "amplitude_only", regions, (None, None, 1.), reps)
        amp_test = _feature_matrix(windows, test, "amplitude_only", regions, (None, None, 1.), reps)
        pi_cache = {}
        for model in models:
            for candidate in results[model]:
                if model == "amplitude_only":
                    x_train, x_test = amp_train, amp_test
                else:
                    geometry = candidate[:2]
                    if geometry not in pi_cache:
                        pi_cache[geometry] = (_feature_matrix(windows, train, "pi_only", regions, candidate, reps),
                                              _feature_matrix(windows, test, "pi_only", regions, candidate, reps))
                    p_train, p_test = pi_cache[geometry]
                    x_train, x_test = (p_train, p_test) if model == "pi_only" else (np.concatenate((p_train, amp_train), axis=1), np.concatenate((p_test, amp_test), axis=1))
                scores, _ = _fit_score(x_train, y, weights, x_test, candidate[2], fits, label=f"{fold}:{model}:R{candidate[0]}:sigma{candidate[1]}:C{candidate[2]}")
                frame = pd.DataFrame([{**{k: windows[i][k] for k in ("event_id", "subject_id", "condition", "noise_type", "snr_target")}, "score": score} for i, score in zip(test, scores)])
                results[model][candidate].append(stratified_auroc(frame))
                predictions[model][candidate].append(frame)
        log(f"{fold}: elapsed_s={time.perf_counter() - fold_start:.3f} fits={len(fits) - fit_start}", flush=True)
    return results, predictions, checks


def _selection(scores, predictions, cfg, cfg_f, forbidden=None):
    selected, rows = {}, []
    tie = cfg["level2"]["cross_validation"]["tie_break"]
    for model, candidates in scores.items():
        means = {c: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for c, v in candidates.items()}
        chosen = choose_candidate(means, tie)
        frame = pd.concat(predictions[model][chosen], ignore_index=True)
        if forbidden is not None:
            assert_training_subjects(frame.subject_id, forbidden)
        primary = threshold_weighted_pvc(frame, cfg_f["thresholds"]["primary"]["target"])
        c1 = threshold_c1(frame, cfg_f["thresholds"]["secondary"]["target"])
        if not np.isfinite(primary) or not np.isfinite(c1):
            raise BlockedError("Factorial inner OOF threshold is not evaluable")
        selected[model] = (chosen, primary, c1, means[chosen])
        for candidate, value in means.items():
            rows.append({"model": model, "resolution": candidate[0], "sigma_pixels": candidate[1], "C": candidate[2],
                         "subject_mean_auroc": value, "not_evaluable_subjects": int(np.sum(~np.isfinite(candidates[candidate]))), "selected": candidate == chosen})
    return selected, rows


def _refit(windows, train, test, selected, cfg, reps, quantiles, fits, checks, label, held=None, save=None):
    source = [i for i in train if windows[i]["condition"] in ("C0", "C1")]
    if held is not None:
        for indices in (train, source):
            assert_training_subjects([windows[i]["subject_id"] for i in indices], held)
    regions = estimate_regions(windows, source, reps, quantiles, excluded_subject=held)
    args = _arrays(windows, train)
    weights = factorial_weights(*args)
    checks.append({"fold": label, "held_out_subject": held or "", "n_rows": len(train), **weight_checks(*args, weights)})
    y = np.array([windows[i]["condition"] in ("C1", "C3") for i in train])
    rows, mmd = [], []
    for model, (candidate, primary, c1, cv_score) in selected.items():
        x_train = _feature_matrix(windows, train, model, regions, candidate, reps)
        x_test = _feature_matrix(windows, test, model, regions, candidate, reps)
        scores, estimator = _fit_score(x_train, y, weights, x_test, candidate[2], fits, label=f"{label}:{model}")
        for i, score in zip(test, scores):
            w = windows[i]
            rows.append({**{k: w[k] for k in ("event_id", "subject_id", "condition", "noise_type", "snr_target")},
                         "fold": label, "model": model, "score": score, "threshold_primary": primary,
                         "threshold_c1": c1, "resolution": candidate[0], "sigma_pixels": candidate[1], "C": candidate[2]})
        if model == "pi_plus_amplitude":
            standardized = estimator.named_steps["standardscaler"].transform(x_test)
            mmd.append((label, test, standardized))
        if save is not None:
            scaler, classifier = estimator.named_steps["standardscaler"], estimator.named_steps["logisticregression"]
            artifact = io.BytesIO()
            np.savez_compressed(artifact, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                                logistic_coef=classifier.coef_, logistic_intercept=classifier.intercept_,
                                region_names=np.asarray(reps),
                                regions=np.array([[np.nan if x is None else x for x in regions[r]] for r in reps]),
                                candidate=np.array([np.nan if x is None else x for x in candidate]))
            atomic_write_bytes(save / "final_models" / f"{model}.npz", artifact.getvalue())
    return rows, mmd, regions


def run_factorial(events, cfg_f, cfg_a, root, output, *, pilot=False, log=print):
    started = time.perf_counter()
    validate_config(cfg_f, cfg_a)
    if not pilot and not cfg_f["approved_for_full_run"]:
        raise BlockedError("Factorial full run requires approved_for_full_run")
    validate_events(events, int(cfg_a["scope"]["events"]))
    root, output = Path(root), Path(output)
    expected_name = "factorial_pilot" if pilot else "factorial"
    if output.name != expected_name or "level2" in [p.lower() for p in output.parts]:
        raise BlockedError("Factorial output directory must be isolated")
    stage = {}
    t = time.perf_counter()
    windows = _load_windows(events, cfg_a, root)
    stage["features"] = time.perf_counter() - t
    subjects = sorted({w["subject_id"] for w in windows})
    if len(subjects) != 10:
        raise BlockedError("Expected ten Development subjects")
    if cfg_f["pilot_outer_subject"] not in subjects:
        raise BlockedError("Pilot outer subject missing")
    reps = list(cfg_a["inputs"]["representations"])
    quantiles = cfg_a["level2"]["persistence_image"]["range_quantiles"]
    fits, regions_rows, range_rows, checks, selections, predictions, mmd_data = [], [], [], [], [], [], []
    outer_subjects = [cfg_f["pilot_outer_subject"]] if pilot else subjects
    inner_times, outer_fit_times = {}, {}
    for held in outer_subjects:
        fold_start, fit_start = time.perf_counter(), len(fits)
        training_subjects = [s for s in subjects if s != held]
        t = time.perf_counter()
        scores, frames, inner_checks = _cross_validate_factorial(windows, cfg_a, training_subjects, reps, quantiles, fits, regions_rows, range_rows, f"outer_{held}", log)
        checks.extend(inner_checks)
        selected, rows = _selection(scores, frames, cfg_a, cfg_f, held)
        selections.extend([{"fold": f"outer_{held}", **r} for r in rows])
        inner_times[held] = time.perf_counter() - t
        train = [i for i, w in enumerate(windows) if w["subject_id"] != held]
        test = [i for i, w in enumerate(windows) if w["subject_id"] == held]
        t = time.perf_counter()
        out, mmd, regions = _refit(windows, train, test, selected, cfg_a, reps, quantiles, fits, checks, f"outer_{held}", held)
        predictions.extend(out)
        mmd_data.extend(mmd)
        regions_rows.extend(_region_rows(regions, f"outer_{held}"))
        range_rows.extend(_range_log(windows, test, regions, f"outer_{held}"))
        outer_fit_times[held] = time.perf_counter() - t
        log(f"outer_{held}: held-out={held} elapsed_s={time.perf_counter() - fold_start:.3f} fits={len(fits) - fit_start}", flush=True)
    outer_fits = len(fits)
    predictions = pd.DataFrame(predictions)
    t = time.perf_counter()
    metrics = metric_tables(predictions, cfg_f | {"scope": cfg_a["scope"]})
    stage["metrics"] = time.perf_counter() - t
    mmd_rows = []
    for condition_a, condition_b, cells in (("C1", "C0", [None]), ("C3", "C2", [(n, s) for n in cfg_a["scope"]["regime1_cells"]["noise_types"] for s in cfg_a["scope"]["regime1_cells"]["snr_db"]])):
        for cell in cells:
            groups = []
            for condition in (condition_a, condition_b):
                parts = []
                for _, indices, x in mmd_data:
                    selected = [j for j, i in enumerate(indices) if windows[i]["condition"] == condition and
                                (cell is None or (windows[i]["noise_type"], windows[i]["snr_target"]) == cell)]
                    if selected:
                        parts.append(x[selected])
                groups.append(np.vstack(parts) if parts else np.empty((0, 0)))
            value, bandwidth = unbiased_mmd2(*groups) if all(len(g) for g in groups) else (np.nan, np.nan)
            mmd_rows.append({"pair": f"{condition_a}_vs_{condition_b}", "noise_type": cell[0] if cell else "clean",
                             "snr_target": cell[1] if cell else "all", "mmd2_unbiased": value,
                             "bandwidth_squared": bandwidth, "n_first": len(groups[0]), "n_second": len(groups[1]),
                             "status": "ok" if np.isfinite(value) else "not_evaluable",
                             "reason": "" if np.isfinite(value) else "fewer_than_two_windows"})
    t = time.perf_counter()
    comparison = comparison_with_level2(predictions, root, cfg_f)
    stage["comparison"] = time.perf_counter() - t
    final_fits = 0
    stage["final"] = 0.0
    if not pilot:
        t = time.perf_counter()
        scores, frames, final_checks = _cross_validate_factorial(windows, cfg_a, subjects, reps, quantiles, fits, regions_rows, range_rows, "final_dev", log)
        checks.extend(final_checks)
        selected, rows = _selection(scores, frames, cfg_a, cfg_f)
        train = list(range(len(windows)))
        _, _, regions = _refit(windows, train, train[:1], selected, cfg_a, reps, quantiles, fits, checks, "final_dev_refit", save=output)
        atomic_write_csv(output / "final_selection_cv.csv", pd.DataFrame(rows))
        atomic_write_json(output / "dev_selection_record.json", {m: {"resolution": c[0][0], "sigma_pixels": c[0][1], "C": c[0][2], "cv_score": c[3], "threshold_primary": c[1], "threshold_c1": c[2], "pi_regions": {r: list(regions[r]) for r in reps}, "selection_basis": "separate LOSO over all Development subjects; outer-fold results not used"} for m, c in selected.items()})
        stage["final"] = time.perf_counter() - t
        final_fits = len(fits) - outer_fits
    atomic_write_csv(output / "weights_check.csv", pd.DataFrame(checks))
    atomic_write_csv(output / "pi_regions.csv", pd.DataFrame(regions_rows))
    atomic_write_parquet(output / "pi_out_of_range.parquet", pd.DataFrame(range_rows))
    atomic_write_csv(output / "inner_cv_selection.csv", pd.DataFrame(selections))
    atomic_write_parquet(output / "oof_predictions.parquet", predictions)
    for name, table in zip(("metrics_window", "metrics_subject", "per_subject_metrics", "bootstrap_ci"), metrics):
        atomic_write_csv(output / f"{name}.csv", table)
    atomic_write_csv(output / "mmd_exploratory.csv", pd.DataFrame(mmd_rows))
    atomic_write_csv(output / "comparison_with_level2.csv", comparison)
    times = np.array([f["wall_time_s"] for f in fits])
    stage["inner_cv_per_fold"] = inner_times
    stage["outer_fit"] = outer_fit_times
    summary = {"fits": len(fits), "outer_fits": outer_fits, "final_selection_fits": final_fits,
               "stage_times_s": stage, "fit_time_median_s": float(np.median(times)),
               "fit_time_min_s": float(times.min()), "fit_time_max_s": float(times.max()),
               "fit_time_total_s": float(times.sum()), "convergence_warnings": int(sum(f["convergence_warnings"] for f in fits)),
               "wall_time_s": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
               "config_sha256": {name: hashlib.sha256((root / "configs" / name).read_bytes()).hexdigest() for name in ("stage1_factorial.yaml", "stage1_analysis.yaml")},
               "code_hash": analysis_code_hash(), "outer_folds": len(outer_subjects), "windows": len(windows), "pilot": pilot}
    atomic_write_json(output / "run_summary.json", summary)
    return summary
