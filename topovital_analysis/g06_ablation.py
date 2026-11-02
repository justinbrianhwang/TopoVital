"""Exploratory Development feature-group ablation of the Level 3 classical models."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import atomic_write_bytes, atomic_write_csv, atomic_write_json, atomic_write_parquet, peak_rss_mb

from .factorial import _arrays, _selection, factorial_weights, weight_checks
from .factorial_metrics import metric_tables, metric_values, stratified_auroc, _scope
from .level1 import analysis_code_hash, validate_events
from .level2 import _fit_score, _load_windows, assert_training_subjects, candidate_grid
from .level3 import KEYS, impute_train, model_comparison
from .runtime import ANALYSIS_CODE_VERSION, blas_thread_report


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_group_manifest(columns, cfg):
    """Validate an exact partition of the parquet's classical columns."""
    columns = [c for c in columns if c not in KEYS]
    if len(columns) != len(set(columns)):
        raise BlockedError("Duplicate classical columns")
    groups = cfg["feature_groups"]
    listed = [i for spec in groups.values() for i in spec["indicators"]]
    if len(listed) != len(set(listed)):
        raise BlockedError("Indicator assigned to more than one group")
    if set(listed) - set(columns):
        raise BlockedError(f"Listed indicator missing: {sorted(set(listed) - set(columns))}")
    rows = []
    counts = {name: 0 for name in groups}
    for column in columns:
        matches = [(name, "indicator" if column in spec["indicators"] else "feature")
                   for name, spec in groups.items()
                   if column in spec["indicators"] or any(column.startswith(p) for p in spec["prefixes"])]
        if len(matches) != 1:
            raise BlockedError(f"Classical column must belong to exactly one group: {column} ({len(matches)})")
        name, kind = matches[0]
        if column in listed and kind != "indicator":
            raise BlockedError(f"Indicator assignment conflict: {column}")
        counts[name] += 1
        rows.append({"group": name, "column": column, "kind": kind})
    expected = cfg["expected_column_counts"]
    if any(counts[name] != expected[name] for name in groups) or len(columns) != expected["total"]:
        raise BlockedError(f"Feature-group column counts differ: {counts}; total={len(columns)}")
    return pd.DataFrame([{**row, "n_in_group": counts[row["group"]]} for row in rows],
                        columns=["group", "column", "kind", "n_in_group"])


def model_column_lists(manifest, cfg):
    all_columns = manifest.column.tolist()
    by_group = {name: set(manifest.loc[manifest.group.eq(name), "column"]) for name in cfg["feature_groups"]}
    references = {}
    for name, spec in cfg["references"].items():
        if spec["groups"] == "all":
            references[name] = all_columns.copy()
        elif spec["groups"] == "all_except_boundary":
            references[name] = [c for c in all_columns if c not in by_group["boundary"]]
        else:
            raise BlockedError(f"Unsupported reference groups: {name}")
    models, not_applicable = dict(references), []
    for ref in cfg["ablations"]["from"]:
        for group in cfg["ablations"]["remove"]:
            name = f"{ref}__minus_{group}"
            columns = [c for c in references[ref] if c not in by_group[group]]
            if not columns or len(columns) == len(references[ref]):
                not_applicable.append({"model": name, "reason": "removal leaves no column" if not columns else "removal removes no column"})
            else:
                models[name] = columns
    return models, not_applicable


def missingness_by_group(classical, manifest):
    records = []
    for (condition, noise, snr), part in classical.groupby(["condition", "noise_type", "snr_target"], dropna=False, sort=True):
        for row in manifest.itertuples(index=False):
            values = part[row.column].to_numpy(dtype=float)
            rate = float(np.mean(values == 1)) if row.kind == "indicator" else float(np.mean(~np.isfinite(values)))
            records.append({"group": row.group, "condition": condition, "noise_type": noise,
                            "snr_target": snr, "column": row.column, "kind": row.kind,
                            "rate": rate, "n_windows": len(part)})
    return pd.DataFrame(records)


def _fold_data(windows, classical, train, test, columns, fold, imputation_rows):
    assert_training_subjects([windows[i]["subject_id"] for i in train], windows[test[0]]["subject_id"])
    args = _arrays(windows, train)
    weights = factorial_weights(*args)
    check = {"fold": fold, "held_out_subject": windows[test[0]]["subject_id"],
             "n_rows": len(train), **weight_checks(*args, weights)}
    a, b = impute_train(classical.iloc[train][columns].to_numpy(dtype=float),
                        classical.iloc[test][columns].to_numpy(dtype=float), columns, fold, imputation_rows)
    y = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
    return a, b, y, weights, check


def _inner_cv(windows, classical, subjects, models, cfg_a, fits, checks, imputation_rows, times, prefix, log):
    grid = candidate_grid(cfg_a, "amplitude_only")
    scores = {m: {c: [] for c in grid} for m in models}
    frames = {m: {c: [] for c in grid} for m in models}
    visible = [i for i, w in enumerate(windows) if w["subject_id"] in subjects]
    full_columns = [c for c in classical if c not in KEYS]
    positions = {c: i for i, c in enumerate(full_columns)}
    for held in subjects:
        start = time.perf_counter()
        train = [i for i in visible if windows[i]["subject_id"] != held]
        test = [i for i in visible if windows[i]["subject_id"] == held]
        fold = f"{prefix}__inner_{held}"
        a, b, y, weights, check = _fold_data(windows, classical, train, test, full_columns, fold, imputation_rows)
        checks.append(check)
        for model, names in models.items():
            ix = [positions[c] for c in names]
            for candidate in grid:
                values, _ = _fit_score(a[:, ix], y, weights, b[:, ix], candidate[2], fits,
                                       label=f"{fold}:{model}:RNone:sigmaNone:C{candidate[2]}")
                frame = pd.DataFrame([{**{k: windows[i][k] for k in KEYS}, "score": value}
                                      for i, value in zip(test, values)])
                scores[model][candidate].append(stratified_auroc(frame))
                frames[model][candidate].append(frame)
        times[fold] = time.perf_counter() - start
        log(f"{fold}: elapsed_s={times[fold]:.3f}", flush=True)
    return scores, frames


def _outer_refit(windows, classical, held, models, selected, fits, checks, imputation_rows):
    train = [i for i, w in enumerate(windows) if w["subject_id"] != held]
    test = [i for i, w in enumerate(windows) if w["subject_id"] == held]
    full_columns = [c for c in classical if c not in KEYS]
    positions = {c: i for i, c in enumerate(full_columns)}
    fold = f"outer_{held}"
    a, b, y, weights, check = _fold_data(windows, classical, train, test, full_columns, fold, imputation_rows)
    checks.append(check)
    rows = []
    for model, (candidate, primary, c1, _) in selected.items():
        ix = [positions[c] for c in models[model]]
        values, _ = _fit_score(a[:, ix], y, weights, b[:, ix], candidate[2], fits, label=f"{fold}:{model}")
        rows.extend({**{k: windows[i][k] for k in KEYS}, "fold": fold, "model": model,
                     "score": value, "threshold_primary": primary, "threshold_c1": c1,
                     "resolution": None, "sigma_pixels": None, "C": candidate[2]}
                    for i, value in zip(test, values))
    return rows


def reference_reproduction(predictions, selections, old_predictions, old_selections, cfg):
    tol = float(cfg["reference_reproduction"]["tolerance_abs"])
    rows = []
    for model in cfg["references"]:
        new = predictions.loc[predictions.model.eq(model)]
        old = old_predictions.loc[old_predictions.model.eq(model) & old_predictions.fold.isin(new.fold.unique())]
        keys = [*KEYS, "fold"]
        valid = (not new.duplicated(keys).any() and not old.duplicated(keys).any() and len(new) == len(old))
        if valid:
            joined = new.merge(old, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_level3"))
            valid = joined._merge.eq("both").all()
        for item, column in (("oof_score", "score"), ("threshold_primary", "threshold_primary"),
                             ("threshold_c1", "threshold_c1"), ("selected_C", "C")):
            diff = np.abs(joined[column].to_numpy(dtype=float) - joined[f"{column}_level3"].to_numpy(dtype=float)) if valid else np.array([])
            equal = bool(valid and np.all((diff <= tol) | (np.isnan(joined[column]) & np.isnan(joined[f"{column}_level3"]))))
            rows.append({"model": model, "item": item, "n_compared": len(diff),
                         "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0., "all_equal": equal,
                         "reason": "" if valid else "oof_keys_differ"})
        a = selections.loc[selections.model.eq(model)]
        b = old_selections.loc[old_selections.model.eq(model) & old_selections.fold.isin(a.fold.unique())]
        sel_keys = ["fold", "C"]
        sel_valid = not a.duplicated(sel_keys).any() and not b.duplicated(sel_keys).any() and len(a) == len(b)
        if sel_valid:
            joined_sel = a.merge(b, on=sel_keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_level3"))
            sel_valid = joined_sel._merge.eq("both").all()
        diff = np.abs(joined_sel.subject_mean_auroc.to_numpy(dtype=float) - joined_sel.subject_mean_auroc_level3.to_numpy(dtype=float)) if sel_valid else np.array([])
        equal = bool(sel_valid and np.all((diff <= tol) | (np.isnan(joined_sel.subject_mean_auroc) & np.isnan(joined_sel.subject_mean_auroc_level3))))
        rows.append({"model": model, "item": "inner_candidate_scores", "n_compared": len(diff),
                     "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0., "all_equal": equal,
                     "reason": "" if sel_valid else "inner_candidate_keys_differ"})
    return pd.DataFrame(rows)


def _paired_comparison(predictions, old, cfg_g, cfg_l3, cfg_f, cfg_a):
    pairs = [f"{ref} - {ref}__minus_{group}" for ref in cfg_g["ablations"]["from"]
             for group in cfg_g["ablations"]["remove"]
             if f"{ref}__minus_{group}" in set(predictions.model)]
    reference_rows = old.loc[old.model.isin(cfg_g["references"])].copy()
    reference_rows = reference_rows.loc[reference_rows.fold.isin(predictions.fold.unique())]
    ablated = predictions.loc[~predictions.model.isin(cfg_g["references"])]
    combined = pd.concat([reference_rows, ablated], ignore_index=True)
    comparison_cfg = {**cfg_l3, "comparisons": {**cfg_l3["comparisons"], "primary": pairs[0], "secondary": pairs[1:]}}
    result = model_comparison(combined, comparison_cfg, cfg_f, cfg_a)
    counts = []
    cells = [(n, s) for n in cfg_a["scope"]["regime1_cells"]["noise_types"]
             for s in cfg_a["scope"]["regime1_cells"]["snr_db"]]
    scopes = [("overall", "all", cells), ("protocol_9_cell", "12_6_0", [c for c in cells if c[1] in (12, 6, 0)])]
    scopes += [(n, s, [(n, s)]) for n, s in cells]
    for pair in pairs:
        ref, model = pair.split(" - ")
        for noise, snr, selected in scopes:
            a = _scope(combined.loc[combined.model.eq(ref)], selected)
            b = _scope(combined.loc[combined.model.eq(model)], selected)
            subjects = sorted(set(a.subject_id) & set(b.subject_id))
            for rule in ("primary", "c1"):
                for metric in cfg_l3["comparisons"]["metrics"]:
                    invalid = [s for s in subjects if not (np.isfinite(metric_values(a.loc[a.subject_id.eq(s)], selected, rule)[metric])
                                                           and np.isfinite(metric_values(b.loc[b.subject_id.eq(s)], selected, rule)[metric]))]
                    counts.append({"model_a": ref, "model_b": model, "noise_type": noise, "snr_target": snr,
                                   "rule": rule, "metric": metric, "n_subjects": len(subjects),
                                   "non_evaluable_subjects": len(invalid),
                                   "non_evaluable_reason": "missing_class_or_score" if invalid else ""})
    return result.merge(pd.DataFrame(counts), on=["model_a", "model_b", "noise_type", "snr_target", "rule", "metric"],
                        how="left", validate="many_to_one")


def _postrun(output):
    files = {p.relative_to(output).as_posix(): _sha(p) for p in output.rglob("*")
             if p.is_file() and p.relative_to(output).as_posix() != "hashes/postrun.json"}
    atomic_write_json(output / "hashes/postrun.json", files)


def run_g06_ablation(events, cfg_g, cfg_l3, cfg_f, cfg_a, root, output, *, max_outer_folds=None, log=print):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    expected = root / cfg_g["output"]
    if max_outer_folds is None and output.resolve() != expected.resolve():
        raise BlockedError("G-06 output directory must be the configured Development directory")
    if max_outer_folds is not None and (max_outer_folds < 1 or output.resolve() == expected.resolve()):
        raise BlockedError("Debug folds require a positive count and a separate temporary output")
    if (output / "run_summary.json").exists():
        raise BlockedError("G-06 run_summary.json already exists; refusing to overwrite")
    if not cfg_g["approved_for_full_run"]:
        raise BlockedError("G-06 full run requires approved_for_full_run")
    if cfg_g["analysis_code_version_required"] != ANALYSIS_CODE_VERSION:
        raise BlockedError("G-06 analysis code version mismatch")
    if [float(c) for c in cfg_g["training"]["c_grid"]] != [c[2] for c in candidate_grid(cfg_a, "amplitude_only")]:
        raise BlockedError("G-06 C grid differs from Level 3")
    source = root / "outputs/stage1_analysis/development/level3_features"
    feature_path = source / "classical_features.parquet"
    summary_path = source / "extraction_summary.json"
    l3_output = root / "outputs/stage1_analysis/development/level3"
    old_oof_path, old_sel_path = l3_output / "oof_predictions.parquet", l3_output / "inner_cv_selection.csv"
    for path in (feature_path, summary_path, old_oof_path, old_sel_path):
        if not path.is_file():
            raise BlockedError(f"Required G-06 input missing: {path}")
    extraction = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_hash = _sha(feature_path)
    if not extraction.get("stop_rule_passed") or extraction.get("classical_features_sha256") != feature_hash:
        raise BlockedError("Level 3 feature stop rule or SHA256 failed")
    validate_events(events, int(cfg_a["scope"]["events"]))
    windows = _load_windows(events, cfg_a, root)
    classical = pd.read_parquet(feature_path)
    window_keys = pd.DataFrame([{k: w[k] for k in KEYS} for w in windows])
    if classical.duplicated(list(KEYS)).any() or window_keys.duplicated(list(KEYS)).any() or len(classical) != len(windows):
        raise BlockedError("G-06 feature keys are not one-to-one")
    classical = window_keys.merge(classical, on=list(KEYS), how="left", validate="one_to_one", indicator=True)
    if not classical._merge.eq("both").all():
        raise BlockedError("G-06 feature keys differ from Level 3 windows")
    classical = classical.drop(columns="_merge")
    manifest = feature_group_manifest(classical.columns, cfg_g)
    models, not_applicable = model_column_lists(manifest, cfg_g)
    old_oof = pd.read_parquet(old_oof_path)
    old_sel = pd.read_csv(old_sel_path)
    for ref in cfg_g["references"]:
        old = old_oof.loc[old_oof.model.eq(ref)]
        if old.duplicated([*KEYS, "fold"]).any() or len(old) != len(windows):
            raise BlockedError(f"Level 3 reference OOF keys invalid: {ref}")
        matched = window_keys.merge(old[list(KEYS)], on=list(KEYS), how="outer", indicator=True)
        if len(matched) != len(windows) or not matched._merge.eq("both").all():
            raise BlockedError(f"Level 3 reference OOF windows differ: {ref}")
    config_hashes = {name: _sha(root / "configs" / name) for name in
                     ("stage1_g06_ablation.yaml", "stage1_level3.yaml", "stage1_factorial.yaml", "stage1_analysis.yaml")}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    threadpool = blas_thread_report()
    preflight = {"code_hash": analysis_code_hash(), "analysis_code_version": ANALYSIS_CODE_VERSION,
                 "config_sha256": config_hashes, "classical_features_sha256": feature_hash,
                 "level3_input_sha256": {"oof_predictions.parquet": _sha(old_oof_path),
                                         "inner_cv_selection.csv": _sha(old_sel_path)},
                 "blas_thread_report": threadpool, "head_commit": head}
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "hashes/preflight.json", preflight)
    atomic_write_bytes(output / "ablation_config.yaml", (root / "configs/stage1_g06_ablation.yaml").read_bytes())
    atomic_write_csv(output / "feature_group_manifest.csv", manifest)
    atomic_write_csv(output / "missingness_by_group.csv", missingness_by_group(classical, manifest))
    fits, checks, imputation_rows, selection_rows, prediction_rows, times = [], [], [], [], [], {}
    subjects = sorted({w["subject_id"] for w in windows})
    if len(subjects) != 10:
        raise BlockedError("Expected ten Development subjects")
    for held in subjects[:max_outer_folds]:
        start = time.perf_counter()
        training_subjects = [s for s in subjects if s != held]
        scores, frames = _inner_cv(windows, classical, training_subjects, models, cfg_a, fits,
                                   checks, imputation_rows, times, f"outer_{held}", log)
        selected, rows = _selection(scores, frames, cfg_a, cfg_f, held)
        selection_rows.extend([{"fold": f"outer_{held}", **r} for r in rows])
        prediction_rows.extend(_outer_refit(windows, classical, held, models, selected, fits, checks, imputation_rows))
        times[f"outer_{held}"] = time.perf_counter() - start
        log(f"outer_{held}: elapsed_s={times[f'outer_{held}']:.3f}", flush=True)
    predictions, selections = pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)
    atomic_write_parquet(output / "oof_predictions.parquet", predictions)
    atomic_write_csv(output / "inner_cv_selection.csv", selections)
    atomic_write_csv(output / "weights_check.csv", pd.DataFrame(checks))
    atomic_write_csv(output / "imputation_log.csv", pd.DataFrame(imputation_rows))
    reproduction = reference_reproduction(predictions, selections, old_oof, old_sel, cfg_g)
    atomic_write_csv(output / "reference_reproduction_check.csv", reproduction)
    passed = bool(reproduction.all_equal.all())
    metrics = metric_tables(predictions, cfg_f | {"scope": cfg_a["scope"]})
    for name, table in zip(("metrics_window", "metrics_subject", "per_subject_metrics", "bootstrap_ci"), metrics):
        atomic_write_csv(output / f"{name}.csv", table)
    if passed:
        atomic_write_csv(output / "paired_comparison.csv", _paired_comparison(predictions, old_oof, cfg_g, cfg_l3, cfg_f, cfg_a))
    summary = {"fits": len(fits), "convergence_warnings": sum(f["convergence_warnings"] for f in fits),
               "per_fold_time_s": times, "wall_time_s": time.perf_counter()-started,
               "peak_rss_mb": peak_rss_mb(), "blas_thread_report": threadpool,
               "analysis_code_version": ANALYSIS_CODE_VERSION, "code_hash": preflight["code_hash"],
               "config_sha256": config_hashes, "classical_features_sha256": feature_hash,
               "models": [{"model": name, "n_columns": len(names), "columns": names} for name, names in models.items()],
               "not_applicable": not_applicable, "reference_reproduction_passed": passed,
               "outer_folds": len(predictions.fold.unique()), "windows": len(windows)}
    atomic_write_json(output / "run_summary.json", summary)
    _postrun(output)
    if not passed:
        raise BlockedError("G-06 reference reproduction failed; see reference_reproduction_check.csv")
    return summary
