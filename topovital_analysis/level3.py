"""Development-only classical and persistence-image comparison."""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import atomic_write_bytes, atomic_write_csv, atomic_write_json, atomic_write_parquet, peak_rss_mb

from .factorial import _arrays, _selection, factorial_weights, weight_checks
from .factorial_metrics import _scope, metric_tables, metric_values, stratified_auroc
from .level1 import analysis_code_hash, validate_events
from .level2 import (_feature_matrix, _fit_score, _load_windows, _range_log, _region_rows,
                     assert_training_subjects, candidate_grid)
from .persistence_images import estimate_regions

KEYS = ("event_id", "subject_id", "condition", "noise_type", "snr_target")


def _models(cfg_l3):
    return [name for name, spec in cfg_l3["models"].items()
            if isinstance(spec, dict) and "classical" in spec and "pi" in spec]


def _guard(root, output, name):
    expected = Path(root) / "outputs/stage1_analysis/development" / name
    if Path(output).resolve() != expected.resolve():
        raise BlockedError(f"Level 3 output directory must be {name} under Development")


def model_columns(frame, model, cfg_l3):
    spec = cfg_l3["models"][model]
    if spec["classical"] == "none":
        return []
    cols = [c for c in frame if c not in KEYS]
    if not spec["boundary"]:
        cols = [c for c in cols if not c.startswith("boundary__") and c != "boundary_missing"]
    return cols


def impute_train(x_train, x_test, columns, fold, log_rows):
    train = np.asarray(x_train, dtype=float).copy()
    test = np.asarray(x_test, dtype=float).copy()
    for j, name in enumerate(columns):
        valid = train[np.isfinite(train[:, j]), j]
        median = float(np.median(valid)) if len(valid) else 0.
        count = int((~np.isfinite(train[:, j])).sum())
        test_count = int((~np.isfinite(test[:, j])).sum())
        log_rows.append({"fold": fold, "column": name, "median": median,
                         "n_missing_train": count, "n_missing_test": test_count,
                         "all_missing": not len(valid)})
        train[~np.isfinite(train[:, j]), j] = median
        test[~np.isfinite(test[:, j]), j] = median
    return train, test


def _classical_blocks(classical, train, test, fold, log_rows):
    columns = [c for c in classical if c not in KEYS]
    start = len(log_rows)
    a, b = impute_train(classical.iloc[train][columns].to_numpy(dtype=float),
                        classical.iloc[test][columns].to_numpy(dtype=float), columns, fold, log_rows)
    medians = {row["column"]: row["median"] for row in log_rows[start:]}
    return a, b, {name: j for j, name in enumerate(columns)}, medians


def _model_blocks(windows, train, test, classical, model, regions, candidate, reps, cfg_l3,
                  classical_train, classical_test, column_index, pi_cache):
    spec = cfg_l3["models"][model]
    a_parts, b_parts, names = [], [], []
    if spec["pi"]:
        geometry = candidate[:2]
        if geometry not in pi_cache:
            pi_cache[geometry] = (_feature_matrix(windows, train, "pi_only", regions, candidate, reps),
                                  _feature_matrix(windows, test, "pi_only", regions, candidate, reps))
        pi_train, pi_test = pi_cache[geometry]
        a_parts.append(pi_train)
        b_parts.append(pi_test)
        names.extend(f"pi_{i}" for i in range(pi_train.shape[1]))
    classical_names = model_columns(classical, model, cfg_l3)
    if classical_names:
        positions = [column_index[name] for name in classical_names]
        a_parts.append(classical_train[:, positions])
        b_parts.append(classical_test[:, positions])
        names.extend(classical_names)
    return np.concatenate(a_parts, axis=1), np.concatenate(b_parts, axis=1), names, classical_names


def _grid(cfg_a, cfg_l3, model):
    return candidate_grid(cfg_a, "pi_only" if cfg_l3["models"][model]["pi"] else "amplitude_only")


def _cross_validate_level3(windows, classical, cfg_l3, cfg_a, subjects, reps, quantiles, fits,
                           region_rows, range_rows, imputation_rows, fold_prefix, log=print, fold_times=None):
    results = {m: {c: [] for c in _grid(cfg_a, cfg_l3, m)} for m in _models(cfg_l3)}
    predictions = {m: {c: [] for c in results[m]} for m in results}
    checks = []
    visible = [i for i, w in enumerate(windows) if w["subject_id"] in subjects]
    for held in subjects:
        start, first_fit = time.perf_counter(), len(fits)
        train = [i for i in visible if windows[i]["subject_id"] != held]
        source = [i for i in train if windows[i]["condition"] in ("C0", "C1")]
        test = [i for i in visible if windows[i]["subject_id"] == held]
        assert_training_subjects([windows[i]["subject_id"] for i in train], held)
        regions = estimate_regions(windows, source, reps, quantiles, excluded_subject=held)
        fold = f"{fold_prefix}__inner_{held}" if fold_prefix.startswith("outer_") else f"{fold_prefix}__held_{held}"
        region_rows.extend(_region_rows(regions, fold))
        range_rows.extend(_range_log(windows, visible, regions, fold))
        args = _arrays(windows, train)
        weights = factorial_weights(*args)
        checks.append({"fold": fold, "held_out_subject": held, "n_rows": len(train), **weight_checks(*args, weights)})
        y = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
        classical_train, classical_test, column_index, _ = _classical_blocks(
            classical, train, test, fold, imputation_rows)
        pi_cache = {}
        for model in results:
            for candidate in results[model]:
                a, b, _, _ = _model_blocks(windows, train, test, classical, model, regions,
                                          candidate, reps, cfg_l3, classical_train, classical_test,
                                          column_index, pi_cache)
                scores, _ = _fit_score(a, y, weights, b, candidate[2], fits,
                                        label=f"{fold}:{model}:R{candidate[0]}:sigma{candidate[1]}:C{candidate[2]}")
                frame = pd.DataFrame([{**{k: windows[i][k] for k in KEYS}, "score": score}
                                      for i, score in zip(test, scores)])
                results[model][candidate].append(stratified_auroc(frame))
                predictions[model][candidate].append(frame)
        elapsed = time.perf_counter() - start
        if fold_times is not None:
            fold_times[fold] = elapsed
        log(f"{fold}: elapsed_s={elapsed:.3f} fits={len(fits)-first_fit}", flush=True)
    return results, predictions, checks


def _refit_level3(windows, classical, train, test, selected, cfg_l3, cfg_a, reps, quantiles,
                  fits, checks, imputation_rows, label, held=None, save=None):
    source = [i for i in train if windows[i]["condition"] in ("C0", "C1")]
    if held is not None:
        assert_training_subjects([windows[i]["subject_id"] for i in train], held)
    regions = estimate_regions(windows, source, reps, quantiles, excluded_subject=held)
    args = _arrays(windows, train)
    weights = factorial_weights(*args)
    checks.append({"fold": label, "held_out_subject": held or "", "n_rows": len(train), **weight_checks(*args, weights)})
    y = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
    rows = []
    classical_train, classical_test, column_index, median_by_name = _classical_blocks(
        classical, train, test, label, imputation_rows)
    pi_cache = {}
    for model, (candidate, primary, c1, cv_score) in selected.items():
        a, b, columns, classical_names = _model_blocks(windows, train, test, classical, model,
            regions, candidate, reps, cfg_l3, classical_train, classical_test, column_index, pi_cache)
        medians = np.asarray([median_by_name[name] for name in classical_names])
        scores, estimator = _fit_score(a, y, weights, b, candidate[2], fits, label=f"{label}:{model}")
        for i, score in zip(test, scores):
            rows.append({**{k: windows[i][k] for k in KEYS}, "fold": label, "model": model,
                         "score": score, "threshold_primary": primary, "threshold_c1": c1,
                         "resolution": candidate[0], "sigma_pixels": candidate[1], "C": candidate[2]})
        if save is not None:
            scaler = estimator.named_steps["standardscaler"]
            classifier = estimator.named_steps["logisticregression"]
            artifact = io.BytesIO()
            np.savez_compressed(artifact, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                                logistic_coef=classifier.coef_, logistic_intercept=classifier.intercept_,
                                region_names=np.asarray(reps),
                                regions=np.asarray([[np.nan if x is None else x for x in regions[r]] for r in reps]),
                                candidate=np.asarray([np.nan if x is None else x for x in candidate]),
                                feature_names=np.asarray(columns),
                                imputation_columns=np.asarray(classical_names), imputation_medians=medians)
            atomic_write_bytes(save / "final_models" / f"{model}.npz", artifact.getvalue())
    return rows, regions


def _paired_draws(a, b, subjects, cells, rule, aggregation, n, seed, metrics):
    rng = np.random.default_rng(seed)
    a_parts = {s: a.loc[a.subject_id.eq(s)] for s in subjects}
    b_parts = {s: b.loc[b.subject_id.eq(s)] for s in subjects}
    a_per = {s: metric_values(a_parts[s], cells, rule) for s in subjects}
    b_per = {s: metric_values(b_parts[s], cells, rule) for s in subjects}
    draws = {metric: [] for metric in metrics}
    cache = {}
    for _ in range(n):
        indices = rng.integers(0, len(subjects), len(subjects))
        key = tuple(np.bincount(indices, minlength=len(subjects)))
        if key not in cache:
            if aggregation == "subject_equal_weight":
                values = {metric: float(np.mean([a_per[subjects[i]][metric] - b_per[subjects[i]][metric]
                                                   for i in indices])) for metric in metrics}
            else:
                selected = [subjects[i] for i in indices]
                a_draw = pd.concat([a_parts[s].assign(subject_id=f"draw_{j}") for j, s in enumerate(selected)], ignore_index=True)
                b_draw = pd.concat([b_parts[s].assign(subject_id=f"draw_{j}") for j, s in enumerate(selected)], ignore_index=True)
                av, bv = metric_values(a_draw, cells, rule), metric_values(b_draw, cells, rule)
                values = {metric: av[metric] - bv[metric] for metric in metrics}
            cache[key] = values
        for metric in metrics:
            draws[metric].append(cache[key][metric])
    return draws


def model_comparison(predictions, cfg_l3, cfg_f, cfg_a):
    cells = [(n, s) for n in cfg_a["scope"]["regime1_cells"]["noise_types"]
             for s in cfg_a["scope"]["regime1_cells"]["snr_db"]]
    scopes = [("overall", "all", cells), ("protocol_9_cell", "12_6_0", [c for c in cells if c[1] in (12, 6, 0)])]
    scopes += [(n, s, [(n, s)]) for n, s in cells]
    pairs = [cfg_l3["comparisons"]["primary"], *cfg_l3["comparisons"]["secondary"]]
    boot = cfg_f["reporting"]["bootstrap"]
    rows = []
    for expression in pairs:
        model_a, model_b = (part.strip() for part in expression.split(" - "))
        a_all = predictions.loc[predictions.model.eq(model_a)]
        b_all = predictions.loc[predictions.model.eq(model_b)]
        match_keys = [*KEYS, "fold"]
        if (a_all.duplicated(match_keys).any() or b_all.duplicated(match_keys).any() or
                len(a_all) != len(b_all) or not a_all[match_keys].reset_index(drop=True).equals(
                    b_all[match_keys].reset_index(drop=True))):
            raise BlockedError(f"Level 3 comparison rows differ: {model_a}, {model_b}")
        for noise, snr, selected in scopes:
            a = _scope(a_all, selected)
            b = _scope(b_all, selected)
            subjects = sorted(set(a.subject_id) & set(b.subject_id))
            for rule in ("primary", "c1"):
                for aggregation in ("window_level", "subject_equal_weight"):
                    if aggregation == "window_level":
                        av, bv = metric_values(a, selected, rule), metric_values(b, selected, rule)
                    else:
                        av = {m: float(np.mean([metric_values(a.loc[a.subject_id.eq(s)], selected, rule)[m]
                                                for s in subjects])) for m in cfg_l3["comparisons"]["metrics"]}
                        bv = {m: float(np.mean([metric_values(b.loc[b.subject_id.eq(s)], selected, rule)[m]
                                                for s in subjects])) for m in cfg_l3["comparisons"]["metrics"]}
                    draws = _paired_draws(a, b, subjects, selected, rule, aggregation,
                                          int(boot["resamples"]), int(boot["seed"]), cfg_l3["comparisons"]["metrics"])
                    for metric in cfg_l3["comparisons"]["metrics"]:
                        finite = np.asarray(draws[metric], dtype=float)
                        finite = finite[np.isfinite(finite)]
                        ok = len(finite) >= .95 * int(boot["resamples"])
                        rows.append({"model_a": model_a, "model_b": model_b, "noise_type": noise,
                                     "snr_target": snr, "rule": rule, "aggregation": aggregation,
                                     "metric": metric, "model_a_value": av[metric], "model_b_value": bv[metric],
                                     "difference": av[metric] - bv[metric],
                                     "ci_low": float(np.quantile(finite, .025)) if ok else np.nan,
                                     "ci_high": float(np.quantile(finite, .975)) if ok else np.nan,
                                     "non_evaluable_resamples": int(boot["resamples"]) - len(finite),
                                     "robustness_rule": cfg_l3["comparisons"]["claim_rule"]})
    return pd.DataFrame(rows)


def _verify_manifest(root, cfg_l3):
    reference = Path(root) / cfg_l3["ph_reproducibility"]["reference"]
    manifest = reference.parent / "hashes/factorial_outputs.sha256"
    if not manifest.is_file():
        raise BlockedError("Factorial output SHA256 manifest missing")
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if not relative.startswith("factorial/"):
            continue
        if relative in entries:
            raise BlockedError("Invalid factorial SHA256 manifest")
        entries[relative] = digest.lower()
    required = ["oof_predictions.parquet", "inner_cv_selection.csv"]
    for name in required:
        relative = f"factorial/{name}"
        if relative not in entries:
            raise BlockedError(f"Factorial SHA256 missing: {relative}")
    for relative, digest in entries.items():
        path = reference.parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest().lower() != digest:
            raise BlockedError(f"Factorial SHA256 mismatch: {relative}")
    return reference


def ph_reproducibility(predictions, selections, output, root, cfg_l3, *, full=False, record=None, log=print):
    reference = _verify_manifest(root, cfg_l3)
    old = pd.read_parquet(reference / "oof_predictions.parquet")
    new = predictions.loc[predictions.model.eq("ph_only")].copy()
    old = old.loc[old.model.eq("pi_only")].copy()
    keys = [*KEYS, "fold"]
    old = old.loc[old.fold.isin(new.fold.unique())]
    if old.duplicated(keys).any() or new.duplicated(keys).any() or len(old) != len(new):
        raise BlockedError("PH reproducibility rows not one-to-one")
    joined = new.merge(old, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_factorial"))
    if not joined._merge.eq("both").all():
        raise BlockedError("PH reproducibility rows differ")
    rows = []
    tol = float(cfg_l3["ph_reproducibility"]["tolerance_abs"])
    for name, col in (("oof_score", "score"), ("threshold_primary", "threshold_primary"),
                      ("threshold_c1", "threshold_c1"), ("selected_resolution", "resolution"),
                      ("selected_sigma", "sigma_pixels"), ("selected_C", "C")):
        a = joined[col].to_numpy(dtype=float)
        b = joined[f"{col}_factorial"].to_numpy(dtype=float)
        diff = np.abs(a-b)
        equal = bool(np.all((diff <= tol) | (np.isnan(a) & np.isnan(b))))
        rows.append({"item": name, "n_compared": len(a), "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.,
                     "all_equal": equal})
    old_sel = pd.read_csv(reference / "inner_cv_selection.csv")
    new_sel = selections.loc[selections.model.eq("ph_only")].copy()
    old_sel = old_sel.loc[old_sel.model.eq("pi_only") & old_sel.fold.isin(new_sel.fold.unique())]
    sel_keys = ["fold", "resolution", "sigma_pixels", "C"]
    joined_sel = new_sel.merge(old_sel, on=sel_keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_factorial"))
    if len(joined_sel) != len(new_sel) or not joined_sel._merge.eq("both").all():
        raise BlockedError("PH inner candidate rows differ")
    diff = np.abs(joined_sel.subject_mean_auroc - joined_sel.subject_mean_auroc_factorial)
    rows.append({"item": "inner_candidate_scores", "n_compared": len(diff),
                 "max_abs_diff": float(np.nanmax(diff)), "all_equal": bool(np.all(diff <= tol))})
    if full:
        old_record = json.loads((reference / "dev_selection_record.json").read_text(encoding="utf-8"))["pi_only"]
        new_record = record["ph_only"]
        values = ["resolution", "sigma_pixels", "C", "cv_score", "threshold_primary", "threshold_c1"]
        differences = [abs(float(new_record[k]) - float(old_record[k])) for k in values]
        for rep in new_record["pi_regions"]:
            if rep not in old_record["pi_regions"]:
                raise BlockedError("PH final PI regions differ")
            for a, b in zip(new_record["pi_regions"][rep], old_record["pi_regions"][rep]):
                differences.append(0. if a is None and b is None else abs(float(a) - float(b)))
        rows.append({"item": "dev_selection_record", "n_compared": len(differences),
                     "max_abs_diff": max(differences), "all_equal": all(d <= tol for d in differences)})
    result = pd.DataFrame(rows)
    atomic_write_csv(output / "ph_reproducibility_check.csv", result)
    if not result.all_equal.all():
        log("PH reproducibility FAILED: see ph_reproducibility_check.csv", flush=True)
    return result


def run_level3(events, cfg_l3, cfg_f, cfg_a, root, output, *, pilot=False, log=print):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    _guard(root, output, "level3_pilot" if pilot else "level3")
    if not pilot and not cfg_l3["approved_for_full_run"]:
        raise BlockedError("Level 3 full run requires approved_for_full_run")
    path = root / "outputs/stage1_analysis/development/level3_features"
    summary_path, feature_path = path / "extraction_summary.json", path / "classical_features.parquet"
    if not summary_path.is_file() or not feature_path.is_file():
        raise BlockedError("Level 3 feature extraction missing")
    extraction = json.loads(summary_path.read_text(encoding="utf-8"))
    if not extraction.get("stop_rule_passed"):
        raise BlockedError("Level 3 feature stop rule failed")
    feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    if feature_sha != extraction.get("classical_features_sha256"):
        raise BlockedError("Level 3 feature SHA256 mismatch")
    validate_events(events, int(cfg_a["scope"]["events"]))
    windows = _load_windows(events, cfg_a, root)
    classical = pd.read_parquet(feature_path)
    keys = list(KEYS)
    window_keys = pd.DataFrame([{k: w[k] for k in keys} for w in windows])
    if classical.duplicated(keys).any() or window_keys.duplicated(keys).any() or len(classical) != len(windows):
        raise BlockedError("Level 3 feature keys are not one-to-one")
    classical = window_keys.merge(classical, on=keys, how="left", validate="one_to_one", indicator=True)
    if not classical._merge.eq("both").all():
        raise BlockedError("Level 3 feature keys do not match PH windows")
    classical = classical.drop(columns="_merge")
    subjects = sorted({w["subject_id"] for w in windows})
    if len(subjects) != 10 or cfg_l3["pilot_outer_subject"] not in subjects:
        raise BlockedError("Expected ten Development subjects and the pilot subject")
    reps = list(cfg_a["inputs"]["representations"])
    quantiles = cfg_a["level2"]["persistence_image"]["range_quantiles"]
    fits, region_rows, range_rows, checks, selections, predictions, imputation_rows = [], [], [], [], [], [], []
    times, inner_times = {}, {}
    outer_subjects = [cfg_l3["pilot_outer_subject"]] if pilot else subjects
    for held in outer_subjects:
        start, first_fit = time.perf_counter(), len(fits)
        training_subjects = [s for s in subjects if s != held]
        scores, frames, inner_checks = _cross_validate_level3(windows, classical, cfg_l3, cfg_a,
            training_subjects, reps, quantiles, fits, region_rows, range_rows, imputation_rows,
            f"outer_{held}", log, inner_times)
        checks.extend(inner_checks)
        selected, rows = _selection(scores, frames, cfg_a, cfg_f, held)
        selections.extend([{"fold": f"outer_{held}", **r} for r in rows])
        train = [i for i, w in enumerate(windows) if w["subject_id"] != held]
        test = [i for i, w in enumerate(windows) if w["subject_id"] == held]
        out, regions = _refit_level3(windows, classical, train, test, selected, cfg_l3, cfg_a,
                                     reps, quantiles, fits, checks, imputation_rows, f"outer_{held}", held)
        predictions.extend(out)
        region_rows.extend(_region_rows(regions, f"outer_{held}"))
        range_rows.extend(_range_log(windows, test, regions, f"outer_{held}"))
        times[held] = time.perf_counter() - start
        log(f"outer_{held}: held-out={held} elapsed_s={times[held]:.3f} fits={len(fits)-first_fit}", flush=True)
    outer_fits = len(fits)
    predictions = pd.DataFrame(predictions)
    selections = pd.DataFrame(selections)
    metrics = metric_tables(predictions, cfg_f | {"scope": cfg_a["scope"]})
    comparison = model_comparison(predictions, cfg_l3, cfg_f, cfg_a)
    output.mkdir(parents=True, exist_ok=True)
    record = None
    if not pilot:
        scores, frames, final_checks = _cross_validate_level3(windows, classical, cfg_l3, cfg_a,
            subjects, reps, quantiles, fits, region_rows, range_rows, imputation_rows, "final_dev", log, inner_times)
        checks.extend(final_checks)
        selected, rows = _selection(scores, frames, cfg_a, cfg_f)
        all_rows = list(range(len(windows)))
        _, regions = _refit_level3(windows, classical, all_rows, all_rows[:1], selected, cfg_l3,
            cfg_a, reps, quantiles, fits, checks, imputation_rows, "final_dev_refit", save=output)
        record = {m: {"resolution": c[0][0], "sigma_pixels": c[0][1], "C": c[0][2],
                      "cv_score": c[3], "threshold_primary": c[1], "threshold_c1": c[2],
                      "pi_regions": {r: list(regions[r]) for r in reps},
                      "selection_basis": "separate LOSO over all Development subjects; outer-fold results not used"}
                  for m, c in selected.items()}
        atomic_write_csv(output / "final_selection_cv.csv", pd.DataFrame(rows))
        atomic_write_json(output / "dev_selection_record.json", record)
    atomic_write_csv(output / "weights_check.csv", pd.DataFrame(checks))
    atomic_write_csv(output / "pi_regions.csv", pd.DataFrame(region_rows))
    atomic_write_parquet(output / "pi_out_of_range.parquet", pd.DataFrame(range_rows))
    atomic_write_csv(output / "inner_cv_selection.csv", selections)
    atomic_write_parquet(output / "oof_predictions.parquet", predictions)
    atomic_write_csv(output / "imputation_log.csv", pd.DataFrame(imputation_rows))
    atomic_write_csv(output / "model_comparison.csv", comparison)
    for name, table in zip(("metrics_window", "metrics_subject", "per_subject_metrics", "bootstrap_ci"), metrics):
        atomic_write_csv(output / f"{name}.csv", table)
    reproducibility = ph_reproducibility(predictions, selections, output, root, cfg_l3,
                                         full=not pilot, record=record, log=log)
    fit_times = np.asarray([f["wall_time_s"] for f in fits])
    summary = {"fits": len(fits), "outer_fits": outer_fits, "final_selection_fits": len(fits)-outer_fits,
               "outer_folds": len(outer_subjects), "windows": len(windows), "pilot": pilot,
               "per_fold_time_s": {**inner_times, **{f"outer_{k}": v for k, v in times.items()}},
               "fit_time_median_s": float(np.median(fit_times)), "fit_time_min_s": float(fit_times.min()),
               "fit_time_max_s": float(fit_times.max()), "fit_time_total_s": float(fit_times.sum()),
               "convergence_warnings": int(sum(f["convergence_warnings"] for f in fits)),
               "wall_time_s": time.perf_counter()-started, "peak_rss_mb": peak_rss_mb(),
               "config_sha256": {name: hashlib.sha256((root / "configs" / name).read_bytes()).hexdigest()
                                 for name in ("stage1_level3.yaml", "stage1_factorial.yaml", "stage1_analysis.yaml")},
               "code_hash": analysis_code_hash(), "classical_features_sha256": feature_sha,
               "ph_reproducibility_passed": bool(reproducibility.all_equal.all())}
    atomic_write_json(output / "run_summary.json", summary)
    return summary
