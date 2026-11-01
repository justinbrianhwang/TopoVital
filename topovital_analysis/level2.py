"""Development-only nested LOSO persistence image analysis."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from topovital.cohort import BlockedError
from topovital.util import atomic_write_bytes, atomic_write_csv, atomic_write_json, atomic_write_parquet, peak_rss_mb

from .distances import finite_diagram
from .level1 import _safe_job, analysis_code_hash, validate_events, window_specs
from .level2_metrics import metric_tables, unbiased_mmd2
from .persistence_images import amplitude_features, estimate_regions, out_of_range, pi_features
from .stats import loso_threshold, signed_auroc


def assert_training_subjects(train, held_out):
    assert held_out not in set(train), "held-out subject leaked into training fold"


def sample_weights(subjects, conditions):
    subjects, conditions = np.asarray(subjects), np.asarray(conditions)
    names = sorted(set(subjects))
    weights = np.empty(len(subjects), dtype=float)
    for subject in names:
        mask = subjects == subject
        counts = {c: int(np.sum(mask & (conditions == c))) for c in ("C1", "C2")}
        if not all(counts.values()) or np.any(mask & ~np.isin(conditions, ("C1", "C2"))):
            raise ValueError("training_subject_missing_class")
        for condition in ("C1", "C2"):
            selected = mask & (conditions == condition)
            weights[selected] = 1 / (2 * len(names) * counts[condition])
            assert np.isclose(weights[selected].sum(), 1 / (2 * len(names)))
        assert np.isclose(weights[mask].sum(), 1 / len(names))
    assert np.isclose(weights.sum(), 1)
    assert np.isclose(weights[conditions == "C1"].sum(), .5)
    assert np.isclose(weights[conditions == "C2"].sum(), .5)
    return weights


def fit_weights(w):
    w = np.asarray(w, dtype=float)
    scaled = w * len(w)
    assert np.isclose(scaled.mean(), 1)
    assert np.allclose(scaled / scaled[0], w / w[0])
    return scaled


def candidate_grid(cfg, model):
    l2 = cfg["level2"]
    c_values = l2["classifier"]["c_grid"]
    if model == "amplitude_only":
        return [(None, None, float(c)) for c in c_values]
    image = l2["persistence_image"]["candidates"]
    return [(int(r), float(s), float(c)) for r in image["resolution"]
            for s in image["sigma_pixels"] for c in c_values]


def choose_candidate(scores, tie_break):
    if tie_break != ["smaller_resolution", "larger_sigma", "smaller_C"]:
        raise BlockedError("Unsupported Level 2 tie break configuration")
    finite = [(candidate, score) for candidate, score in scores.items() if np.isfinite(score)]
    if not finite:
        raise BlockedError("No evaluable inner CV candidate")
    return min(finite, key=lambda item: (-item[1], item[0][0] or 0, -(item[0][1] or 0), item[0][2]))[0]


def threshold_from_inner(frame, target):
    return loso_threshold(frame, "C1", target)


def require_full_approval(cfg, pilot):
    if not pilot and not cfg["level2"]["approved_for_full_run"]:
        raise BlockedError("Level 2 full run requires approved_for_full_run")


def _load_windows(events, cfg, root):
    allowed = (root / cfg["inputs"]["clean_jobs"], root / cfg["inputs"]["regime1_jobs"])
    keys = cfg["inputs"]["representations"]
    loaded = {}

    def read(path, spec, half):
        path = _safe_job(path, allowed)
        if path not in loaded:
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"].item()))
                if meta.get("validation_status") != "ok" or meta.get("event_id") != spec["event_id"] or meta.get("half") != half:
                    raise ValueError("invalid_job_metadata")
                diagrams = {rep: finite_diagram(saved[key], vr_h0=key == "vr_h0", vr_h1=key == "vr_h1")
                            for rep, key in keys.items()}
                low = np.asarray(saved["cubical_sub_essential_births"], dtype=float)
                high = np.asarray(saved["cubical_super_essential_births"], dtype=float)
                if len(low) != 1 or len(high) != 1:
                    raise ValueError("invalid_cubical_essential_births")
                loaded[path] = (diagrams, (float(low[0]), float(-high[0])), meta)
        data = loaded[path]
        expected = spec["condition"] if half == "post" or spec["condition"] in ("C0", "C1") else cfg["inputs"]["pre_half_of"][spec["condition"]]
        if data[2].get("condition") != expected:
            raise ValueError("job_condition_mismatch")
        return data[:2]

    windows = []
    for spec in window_specs(events, cfg, root):
        pre, pre_amp = read(spec["pre_path"], spec, "pre")
        post, post_amp = read(spec["post_path"], spec, "post")
        windows.append({**{k: spec[k] for k in ("event_id", "subject_id", "condition", "noise_type", "snr_target")},
                        "pre": pre, "post": post, "amplitude": {"pre": pre_amp, "post": post_amp}})
    return windows


def _feature_matrix(windows, indices, model, regions, candidate, reps):
    r, sigma, _ = candidate
    vectors = []
    for i in indices:
        amp = amplitude_features(windows[i])
        if model == "amplitude_only":
            vectors.append(amp)
        else:
            pi = pi_features(windows[i], regions, r, sigma, reps)
            vectors.append(pi if model == "pi_only" else np.concatenate((pi, amp)))
    return np.stack(vectors)


def _fit_score(x_train, y_train, weights, x_eval, c, fits, *, label=None):
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=5000))
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, logisticregression__sample_weight=fit_weights(weights))
    elapsed = time.perf_counter() - start
    fits.append({"fit": len(fits) + 1, "label": label or "", "wall_time_s": elapsed,
                 "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in seen)})
    return model.predict_proba(x_eval)[:, 1], model


def _range_log(windows, indices, regions, fold, detail=None):
    rows = []
    groups = {}
    for i in indices:
        w = windows[i]
        for half in ("pre", "post"):
            for rep, region in regions.items():
                count, persistence_sum = out_of_range(w[half][rep], region)
                if detail is not None:
                    detail.append({"fold": fold, "event_id": w["event_id"], "subject_id": w["subject_id"],
                                   "condition": w["condition"], "noise_type": w["noise_type"],
                                   "snr_target": w["snr_target"], "half": half, "representation": rep,
                                   "outside_count": count, "outside_persistence_sum": persistence_sum})
                key = (w["condition"], half, rep)
                if key not in groups:
                    groups[key] = [0, 0., 0]
                groups[key][0] += count
                groups[key][1] += persistence_sum
                groups[key][2] += 1
    for (condition, half, rep), (count, persistence_sum, n_halves) in groups.items():
        rows.append({"fold": fold, "condition": condition, "half": half, "representation": rep,
                     "outside_count": count, "outside_persistence_sum": persistence_sum, "n_halves": n_halves})
    return rows


def _region_rows(regions, fold):
    return [{"fold": fold, "representation": rep, "birth_min": region[0], "birth_max": region[1],
             "persistence_min": region[2], "persistence_max": region[3]} for rep, region in regions.items()]


def _target(cfg):
    rule = cfg["level2"]["cross_validation"]["threshold_rule"]
    match = re.search(r">=\s*(0(?:\.\d+)?|1(?:\.0+)?)", rule)
    if not match:
        raise BlockedError("Cannot parse Level 2 sensitivity threshold")
    return float(match.group(1))


def _cross_validate_candidates(windows, cfg, subjects, reps, quantiles, fits,
                               region_rows, range_rows, range_detail, fold_prefix):
    """Run the same held-out-subject fits for inner and final Development CV."""
    scores_by_model = {model: {candidate: [] for candidate in candidate_grid(cfg, model)}
                       for model in cfg["level2"]["features"]["models"]}
    predictions_by_model = {model: {candidate: [] for candidate in candidates}
                            for model, candidates in scores_by_model.items()}
    visible = [i for i, w in enumerate(windows) if w["subject_id"] in subjects]
    for held in subjects:
        train_subjects = [s for s in subjects if s != held]
        assert_training_subjects(train_subjects, held)
        train = [i for i in visible if windows[i]["subject_id"] != held and windows[i]["condition"] in ("C1", "C2")]
        region_source = [i for i in visible if windows[i]["subject_id"] != held and windows[i]["condition"] in ("C0", "C1")]
        eval_indices = [i for i in visible if windows[i]["subject_id"] == held and windows[i]["condition"] in ("C1", "C2")]
        regions = estimate_regions(windows, region_source, reps, quantiles, excluded_subject=held)
        fold = f"{fold_prefix}__inner_{held}" if fold_prefix.startswith("outer_") else f"{fold_prefix}__held_{held}"
        region_rows.extend(_region_rows(regions, fold))
        range_rows.extend(_range_log(windows, visible, regions, fold, range_detail))
        y = np.array([windows[i]["condition"] == "C1" for i in train])
        weights = sample_weights([windows[i]["subject_id"] for i in train], [windows[i]["condition"] for i in train])
        amplitude_train = _feature_matrix(windows, train, "amplitude_only", regions, (None, None, 1.), reps)
        amplitude_eval = _feature_matrix(windows, eval_indices, "amplitude_only", regions, (None, None, 1.), reps)
        pi_cache, combined_cache = {}, {}
        for model, candidates in scores_by_model.items():
            for candidate in candidates:
                geometry = candidate[:2]
                if model == "amplitude_only":
                    x_train, x_eval = amplitude_train, amplitude_eval
                else:
                    if geometry not in pi_cache:
                        pi_cache[geometry] = (_feature_matrix(windows, train, "pi_only", regions, candidate, reps),
                                              _feature_matrix(windows, eval_indices, "pi_only", regions, candidate, reps))
                    pi_train, pi_eval = pi_cache[geometry]
                    if model == "pi_only":
                        x_train, x_eval = pi_train, pi_eval
                    else:
                        if geometry not in combined_cache:
                            combined_cache[geometry] = (np.concatenate((pi_train, amplitude_train), axis=1),
                                                        np.concatenate((pi_eval, amplitude_eval), axis=1))
                        x_train, x_eval = combined_cache[geometry]
                scores, _ = _fit_score(x_train, y, weights, x_eval, candidate[2], fits,
                                        label=f"{fold}:{model}:R{candidate[0]}:sigma{candidate[1]}:C{candidate[2]}")
                frame = pd.DataFrame({"subject_id": held,
                                      "condition": [windows[i]["condition"] for i in eval_indices], "score": scores})
                auc = signed_auroc(frame.loc[frame.condition.eq("C1"), "score"],
                                   frame.loc[frame.condition.eq("C2"), "score"])
                scores_by_model[model][candidate].append(auc)
                predictions_by_model[model][candidate].append(frame)
    return scores_by_model, predictions_by_model


def final_dev_selection(windows, cfg, output: Path, fits):
    """Choose and refit from Development windows and config alone."""
    started = time.perf_counter()
    first_fit = len(fits)
    subjects = sorted({w["subject_id"] for w in windows})
    if len(subjects) != 10:
        raise BlockedError("Expected ten Development subjects")
    reps = list(cfg["inputs"]["representations"])
    quantiles = cfg["level2"]["persistence_image"]["range_quantiles"]
    scores, predictions = _cross_validate_candidates(windows, cfg, subjects, reps, quantiles,
                                                       fits, [], [], None, "final_dev")
    target = _target(cfg)
    rows, record = [], {}
    region_source = [i for i, w in enumerate(windows) if w["condition"] in ("C0", "C1")]
    regions = estimate_regions(windows, region_source, reps, quantiles)
    train = [i for i, w in enumerate(windows) if w["condition"] in ("C1", "C2")]
    y = np.array([windows[i]["condition"] == "C1" for i in train])
    weights = sample_weights([windows[i]["subject_id"] for i in train], [windows[i]["condition"] for i in train])
    for model, candidates in scores.items():
        means = {candidate: float(np.mean(valid)) if len(valid) else np.nan
                 for candidate, values in candidates.items()
                 for valid in [np.asarray(values)[np.isfinite(values)]]}
        chosen = choose_candidate(means, cfg["level2"]["cross_validation"]["tie_break"])
        for candidate, score in means.items():
            rows.append({"model": model, "resolution": candidate[0], "sigma_pixels": candidate[1],
                         "C": candidate[2], "subject_mean_auroc": score,
                         "not_evaluable_subjects": int(np.sum(~np.isfinite(candidates[candidate]))),
                         "selected": candidate == chosen})
        threshold = threshold_from_inner(pd.concat(predictions[model][chosen], ignore_index=True), target)
        if not np.isfinite(threshold):
            raise BlockedError("Final Development OOF threshold is not evaluable")
        x_train = _feature_matrix(windows, train, model, regions, chosen, reps)
        _, estimator = _fit_score(x_train, y, weights, x_train[:1], chosen[2], fits,
                                   label=f"final_dev_refit:{model}:R{chosen[0]}:sigma{chosen[1]}:C{chosen[2]}")
        scaler = estimator.named_steps["standardscaler"]
        classifier = estimator.named_steps["logisticregression"]
        region_array = np.array([[np.nan if value is None else value for value in regions[rep]] for rep in reps], dtype=float)
        artifact = io.BytesIO()
        np.savez_compressed(artifact, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                            logistic_coef=classifier.coef_, logistic_intercept=classifier.intercept_,
                            region_names=np.asarray(reps), regions=region_array,
                            candidate=np.array([np.nan if value is None else value for value in chosen], dtype=float))
        atomic_write_bytes(output / "final_models" / f"{model}.npz", artifact.getvalue())
        record[model] = {"resolution": chosen[0], "sigma_pixels": chosen[1], "C": chosen[2],
                         "cv_score": means[chosen], "operating_threshold": threshold,
                         "target_sensitivity": target,
                         "pi_regions": {rep: list(regions[rep]) for rep in reps},
                         "feature_count": int(x_train.shape[1]), "n_training_windows": len(train),
                         "selection_basis": "separate LOSO over all Development subjects; outer-fold results not used"}
    atomic_write_csv(output / "final_selection_cv.csv", pd.DataFrame(rows))
    atomic_write_json(output / "dev_selection_record.json", record)
    return {"final_selection_fits": len(fits) - first_fit,
            "final_selection_wall_time_s": time.perf_counter() - started}


def run_level2(events, cfg, root: Path, output: Path, *, pilot=False):
    started = time.perf_counter()
    require_full_approval(cfg, pilot)
    validate_events(events, int(cfg["scope"]["events"]))
    if cfg["scope"]["split"] != "development_only" or cfg["inputs"]["pairs"] != "finite_only":
        raise BlockedError("Unsupported Level 2 scope")
    windows = _load_windows(events, cfg, root)
    subjects = sorted(set(w["subject_id"] for w in windows))
    if len(subjects) != 10:
        raise BlockedError("Expected ten Development subjects")
    reps = list(cfg["inputs"]["representations"])
    quantiles = cfg["level2"]["persistence_image"]["range_quantiles"]
    models = cfg["level2"]["features"]["models"]
    target = _target(cfg)
    region_rows, range_rows, selection_rows, prediction_rows, fits, mmd_rows = [], [], [], [], [], []
    range_detail = [] if pilot else None
    outer_subjects = subjects[:1] if pilot else subjects
    for outer in outer_subjects:
        train_subjects = [s for s in subjects if s != outer]
        assert_training_subjects(train_subjects, outer)
        outer_train = [i for i, w in enumerate(windows) if w["subject_id"] != outer]
        outer_eval = [i for i, w in enumerate(windows) if w["subject_id"] == outer]
        inner_scores, inner_predictions = _cross_validate_candidates(
            windows, cfg, train_subjects, reps, quantiles, fits, region_rows,
            range_rows, range_detail, f"outer_{outer}")
        selected = {}
        for model in models:
            means = {candidate: float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                     for candidate, values in inner_scores[model].items()}
            candidate = choose_candidate(means, cfg["level2"]["cross_validation"]["tie_break"])
            selected[model] = candidate
            for choice, score in means.items():
                selection_rows.append({"fold": f"outer_{outer}", "model": model, "resolution": choice[0],
                                       "sigma_pixels": choice[1], "C": choice[2], "inner_subject_mean_auroc": score,
                                       "not_evaluable_subjects": int(np.sum(~np.isfinite(inner_scores[model][choice]))),
                                       "selected": choice == candidate})
            threshold = threshold_from_inner(pd.concat(inner_predictions[model][candidate], ignore_index=True), target)
            if not np.isfinite(threshold):
                raise BlockedError("Inner OOF threshold is not evaluable")
            region_source = [i for i in outer_train if windows[i]["condition"] in ("C0", "C1")]
            regions = estimate_regions(windows, region_source, reps, quantiles, excluded_subject=outer)
            if model == models[0]:
                region_rows.extend(_region_rows(regions, f"outer_{outer}"))
                range_rows.extend(_range_log(windows, outer_train + outer_eval, regions, f"outer_{outer}", range_detail))
            train = [i for i in outer_train if windows[i]["condition"] in ("C1", "C2")]
            x_train = _feature_matrix(windows, train, model, regions, candidate, reps)
            x_eval = _feature_matrix(windows, outer_eval, model, regions, candidate, reps)
            y = np.array([windows[i]["condition"] == "C1" for i in train])
            weights = sample_weights([windows[i]["subject_id"] for i in train], [windows[i]["condition"] for i in train])
            scores, estimator = _fit_score(x_train, y, weights, x_eval, candidate[2], fits,
                                            label=f"outer_{outer}:{model}:R{candidate[0]}:sigma{candidate[1]}:C{candidate[2]}")
            standardized = estimator.named_steps["standardscaler"].transform(x_eval) if model == "pi_plus_amplitude" else None
            for position, i in enumerate(outer_eval):
                w = windows[i]
                prediction_rows.append({**{k: w[k] for k in ("event_id", "subject_id", "condition", "noise_type", "snr_target")},
                                        "fold": f"outer_{outer}", "model": model, "score": scores[position],
                                        "threshold": threshold, "resolution": candidate[0], "sigma_pixels": candidate[1], "C": candidate[2]})
            if standardized is not None:
                for noise in cfg["scope"]["regime1_cells"]["noise_types"]:
                    for snr in cfg["scope"]["regime1_cells"]["snr_db"]:
                        c1 = standardized[[j for j, i in enumerate(outer_eval) if windows[i]["condition"] == "C1"]]
                        c2 = standardized[[j for j, i in enumerate(outer_eval) if windows[i]["condition"] == "C2" and windows[i]["noise_type"] == noise and windows[i]["snr_target"] == snr]]
                        mmd_rows.append({"fold": f"outer_{outer}", "noise_type": noise, "snr_target": snr,
                                         "c1_features": c1, "c2_features": c2})
    predictions = pd.DataFrame(prediction_rows)
    metric_window, metric_subject, per_subject, ci = metric_tables(predictions, cfg)
    mmd_output = []
    for noise in cfg["scope"]["regime1_cells"]["noise_types"]:
        for snr in cfg["scope"]["regime1_cells"]["snr_db"]:
            parts = [r for r in mmd_rows if r["noise_type"] == noise and r["snr_target"] == snr]
            x = np.vstack([r["c1_features"] for r in parts])
            y = np.vstack([r["c2_features"] for r in parts])
            value, bandwidth2 = unbiased_mmd2(x, y)
            mmd_output.append({"noise_type": noise, "snr_target": snr, "mmd2_unbiased": value,
                               "bandwidth_squared": bandwidth2, "n_c1": len(x), "n_c2": len(y),
                               "status": "ok" if np.isfinite(value) else "not_evaluable",
                               "reason": "" if np.isfinite(value) else "fewer_than_two_windows"})
    atomic_write_csv(output / "pi_regions.csv", pd.DataFrame(region_rows))
    atomic_write_parquet(output / "pi_out_of_range.parquet", pd.DataFrame(range_rows))
    if range_detail is not None:
        atomic_write_parquet(output / "pi_out_of_range_detail.parquet", pd.DataFrame(range_detail))
    atomic_write_csv(output / "inner_cv_selection.csv", pd.DataFrame(selection_rows))
    atomic_write_parquet(output / "oof_predictions.parquet", predictions)
    atomic_write_csv(output / "metrics_window.csv", metric_window)
    atomic_write_csv(output / "metrics_subject.csv", metric_subject)
    atomic_write_csv(output / "per_subject_metrics.csv", per_subject)
    atomic_write_csv(output / "bootstrap_ci.csv", ci)
    atomic_write_csv(output / "mmd_exploratory.csv", pd.DataFrame(mmd_output))
    outer_fit_count = len(fits)
    final_result = {"final_selection_fits": 0, "final_selection_wall_time_s": 0.0}
    if not pilot:
        final_result = final_dev_selection(windows, cfg, output, fits)
    times = np.array([f["wall_time_s"] for f in fits])
    candidate_count = sum(len(candidate_grid(cfg, m)) for m in models)
    full_fits = (len(subjects) * (len(subjects) - 1) * candidate_count + len(subjects) * len(models)
                 + len(subjects) * candidate_count + len(models))
    summary = {"fits": len(fits), "outer_fits": outer_fit_count, **final_result, "fit_records": fits,
               "fit_time_median_s": float(np.median(times)), "fit_time_min_s": float(times.min()),
               "fit_time_max_s": float(times.max()), "fit_time_total_s": float(times.sum()),
               "projected_full_run_fits": full_fits, "projected_full_run_fit_time_s": float(full_fits * np.median(times)),
               "convergence_warnings": int(sum(f["convergence_warnings"] for f in fits)),
               "wall_time_s": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
               "config_sha256": hashlib.sha256((root / "configs" / "stage1_analysis.yaml").read_bytes()).hexdigest(),
               "code_hash": analysis_code_hash(), "outer_folds": len(outer_subjects), "windows": len(windows),
               "sample_weight_scale": "mean_1 (relative weights from sample_weights)"}
    atomic_write_json(output / "run_summary.json", summary)
    return summary
