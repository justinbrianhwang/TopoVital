"""Sequential, Development-only S105 pilot orchestration."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import load_yaml, peak_rss_mb
from topovital_stage1.core import load_dev_events
from topovital_analysis.level1 import validate_events
from topovital_analysis.level2 import _load_windows
from topovital_analysis.level3 import KEYS
from topovital_analysis.runtime import verify_blas_threads

from .audit import assert_unchanged, file_hashes, manifest, provenance_check, sha, inner_provenance_check, outer_check
from .core import BaseFitStore, subject_rows
from .evaluation import accepted_indices, detection, risk_rows, safety
from .features import monitor_matrix
from .monitors import fit_monitor, operating_threshold
from .plan import needed_basefits


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")


def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _stage(name, timings, started):
    timings[name] = time.perf_counter() - started
    print(f"{name}: elapsed_s={timings[name]:.3f}", flush=True)
    return time.perf_counter()


def _regression(base, frame, reference, cfg):
    expected = reference.loc[(reference.model == cfg["base_model"]["model"]) &
                             (reference.fold == cfg["base_model"]["regression_check"]["reference_fold"])]
    if len(frame) != len(expected):
        raise BlockedError("base regression row count mismatch")
    merged = frame.merge(expected, on=list(KEYS), validate="one_to_one", suffixes=("", "_ref"))
    tol = float(cfg["base_model"]["regression_check"]["tolerance_abs"])
    if len(merged) != len(frame) or not np.isclose(base.c, expected.C.iloc[0], atol=tol, rtol=0) or \
       not np.isclose(base.theta, expected.threshold_primary.iloc[0], atol=tol, rtol=0) or \
       np.max(np.abs(merged.p - merged.score)) > tol:
        raise BlockedError("BaseFit(S105) regression against Level 3 failed")


def run_pilot(root, cfg):
    root = Path(root)
    if not cfg["approved_for_pilot"] or cfg["nesting"]["workers"] != 1:
        raise BlockedError("pilot approval or sequential worker setting invalid")
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    started, timings = time.perf_counter(), {}
    blas = verify_blas_threads()
    protected = file_hashes(root, cfg["protected"])
    code = file_hashes(root, ["topovital_followup/**/*.py"])
    code_sha = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    config_sha = sha(root / "configs/followup_error_monitor_v0_1.yaml")
    _json(out / "hashes/prerun.json", {"protected": protected, "code": code, "config": config_sha})
    inputs = manifest(root, cfg, code_sha)
    _json(out / "input_manifest.json", inputs)
    if len(inputs["diagrams"]) != cfg["scope"]["expected_counts"]["npz_jobs"]:
        raise BlockedError("diagram count mismatch")
    events = load_dev_events(root / cfg["inputs"]["events"], root / "outputs/split_assignment.csv")
    validate_events(events, cfg["scope"]["expected_counts"]["events"])
    analysis = load_yaml(root / "configs/stage1_analysis.yaml")
    windows = _load_windows(events, analysis, root)
    expected = cfg["scope"]["expected_counts"]
    if len(windows) != expected["windows"] or len({w["subject_id"] for w in windows}) != expected["subjects"]:
        raise BlockedError("window or subject count mismatch")
    for condition in ("C0", "C1", "C2", "C3"):
        if sum(w["condition"] == condition for w in windows) != expected[condition]:
            raise BlockedError(f"{condition} count mismatch")
    summary = json.loads((root / cfg["inputs"]["classical_features_summary"]).read_text(encoding="utf-8"))
    feature_sha = sha(root / cfg["inputs"]["classical_features"])
    if not summary.get("stop_rule_passed") or feature_sha != summary["classical_features_sha256"]:
        raise BlockedError("classical feature stop rule or SHA mismatch")
    raw = pd.read_parquet(root / cfg["inputs"]["classical_features"])
    keys = pd.DataFrame([{key: w[key] for key in KEYS} for w in windows])
    if raw.duplicated(list(KEYS)).any() or keys.duplicated(list(KEYS)).any() or len(raw) != len(keys):
        raise BlockedError("classical keys not one-to-one")
    classical = keys.merge(raw, on=list(KEYS), validate="one_to_one", indicator=True)
    if not classical._merge.eq("both").all():
        raise BlockedError("classical key mismatch")
    classical = classical.drop(columns="_merge")
    subjects = frozenset(w["subject_id"] for w in windows)
    outer = cfg["pilot"]["outer_subject"]
    training = subjects - {outer}
    _json(out / "plan.json", {"pilot": needed_basefits([outer], subjects, cfg["base_model"]["c_grid"]).counts(),
                              "full": needed_basefits(subjects, subjects, cfg["base_model"]["c_grid"]).counts()})
    started = _stage("inputs", timings, started)

    factorial = load_yaml(root / "configs/stage1_factorial.yaml")
    target = factorial["thresholds"]["primary"]["target"] if isinstance(factorial.get("thresholds"), dict) else None
    if target is None:
        from topovital_analysis.factorial import _primary_target
        target = _primary_target(factorial)
    reference = pd.read_parquet(root / cfg["inputs"]["level3_reference"])
    def log_stage(name):
        nonlocal started
        started = _stage(name, timings, started)
    result = compute_fold(windows, classical, outer, cfg, threshold_target=target,
                          hashes=(feature_sha, config_sha, code_sha), reference=reference,
                          log=log_stage)
    for name, rows in result["csv"].items():
        _csv(out / f"{name}.csv", rows)
    result["error_label_manifest"].to_parquet(out / "error_label_manifest.parquet", index=False)
    result["predictions"].to_parquet(out / "pilot_predictions.parquet", index=False)
    audit = [*result["audit"], {"name": "blas", "passed": bool(blas), "n_checked": 1, "detail": str(blas)}]
    if not all(entry["passed"] for entry in audit):
        _json(out / "leakage_audit.json", audit)
        raise BlockedError("leakage audit failed")
    _json(out / "leakage_audit.json", audit)
    post = file_hashes(root, cfg["protected"])
    _json(out / "hashes/postrun.json", {"protected": post, "code": file_hashes(root, ["topovital_followup/**/*.py"]),
                                         "config": sha(root / "configs/followup_error_monitor_v0_1.yaml")})
    assert_unchanged(protected, post)
    summary = result["summary"] | {"stage_wall_seconds": timings, "main_peak_rss_mb": peak_rss_mb(),
                                   "max_child_peak_rss_mb": 0, "blas": blas, "packages": inputs["packages"],
                                   "total_wall_s": float(sum(timings.values()))}
    _json(out / "run_summary.json", summary)
    return summary


def compute_fold(windows, classical, outer, cfg, *, threshold_target, hashes, reference=None, log=print):
    """Compute a held-out fold without reading or writing project files."""
    subjects = frozenset(w["subject_id"] for w in windows)
    training = subjects - {outer}
    if outer not in subjects or len(training) < 3:
        raise BlockedError("outer subject or training subjects missing")
    feature_sha, config_sha, code_sha = hashes
    store = BaseFitStore(windows, classical, cfg["base_model"]["c_grid"], threshold_target,
                         feature_sha, config_sha, code_sha)
    base = store.basefit(training)
    outer_indices = subject_rows(windows, {outer})
    outer_frame = base.predict(windows, classical, outer_indices)
    fit_rows = lambda: [{"basefit_id": b.id, "subjects": ",".join(sorted(b.subjects)), "selected_C": b.c,
                         "threshold_primary": b.theta, "cv_score": b.cv_score,
                         "unit_ids": json.dumps(b.unit_ids), "cache_keys": json.dumps([store.units[k].cache_key for k in store.units if store.units[k].id in b.unit_ids])}
                        for b in store.basefits.values()]
    audit = []
    if reference is not None:
      try:
        _regression(base, outer_frame, reference, cfg)
        audit.append({"name": "level3_regression", "passed": len(outer_frame) > 0, "n_checked": len(outer_frame), "detail": base.id})
      except BlockedError as exc:
        audit.append({"name": "level3_regression", "passed": False, "n_checked": len(outer_frame), "detail": str(exc)})
        raise
    log("base_regression")

    tables = {}
    tables[training] = store.error_table(training)
    eval_frames = {outer: outer_frame.assign(table_id=f"evaluation:{outer}", role="evaluation",
                                                  training_subjects=",".join(sorted(training)))}
    for held in sorted(training):
        subset = training - {held}
        tables[subset] = store.error_table(subset)
        fitted = store.basefit(subset)
        eval_frames[held] = fitted.predict(windows, classical, subject_rows(windows, {held})).assign(
            table_id=f"evaluation:{held}", role="evaluation", training_subjects=",".join(sorted(subset)))
    error_tables = [table.assign(role="error_training") for table in tables.values()]
    error_label_manifest = pd.concat([*error_tables, *eval_frames.values()], ignore_index=True)
    summary_rows = []
    for table in [*error_tables, *eval_frames.values()]:
        for (subject, condition, noise, snr), group in table.groupby(["subject_id", "condition", "noise_type", "snr_target"], dropna=False):
            summary_rows.append({"table_id": group.table_id.iloc[0], "subject_id": subject, "condition": condition,
                                 "noise_type": noise, "snr_target": snr, "events": group.event_id.nunique(),
                                 "windows": len(group), "errors": int(group.e.sum()),
                                 "FP": int(((group.y == 0) & (group.y_hat == 1)).sum()),
                                 "FN": int(((group.y == 1) & (group.y_hat == 0)).sum()),
                                 "independent_error_subjects": int(group.loc[group.e == 1, "subject_id"].nunique()),
                                 "both_classes_present": group.e.nunique() == 2})
    audit.extend(provenance_check(outer, store.basefits.values(), list(tables.values())))
    for held in sorted(training):
        subset = training - {held}
        audit.append(inner_provenance_check(held, subset, tables[subset], eval_frames[held]))
    log("error_tables")

    pi_cfg = cfg["persistence_image"] | {"forbidden_inputs": cfg["forbidden_inputs"]}
    selection, thresholds, predictions = [], {}, outer_frame.copy()
    monitor_fits, fallbacks, fit_records = 0, [], []
    matrix_cache, pi_cache, cache_hits, pi_rows, feature_rows, outer_hashes = {}, {}, 0, [], [], []
    def matrix(model, frame, subjects):
        nonlocal cache_hits
        indices = tuple(frame.window_index.to_numpy(dtype=int))
        key = (model, tuple(sorted(subjects)), indices)
        if key in matrix_cache:
            cache_hits += 1
            return matrix_cache[key]
        if model in ("M3", "M4"):
            pi_key = (tuple(sorted(subjects)), indices)
            if pi_key not in pi_cache:
                from .features import pi_matrix
                pi_cache[pi_key] = pi_matrix(windows, subjects, indices, pi_cfg)
                _, _, _, counts = pi_cache[pi_key]
                pi_rows.extend({"region_subjects": ",".join(sorted(subjects)),
                                "row_subjects": ",".join(sorted(set(frame.subject_id))),
                                "representation": rep, "outside_count": count}
                               for rep, count in counts.items())
            if model == "M3":
                pi, names, regions, counts = pi_cache[pi_key]
                result = (pi, names, regions, counts)
            else:
                base = matrix("M2", frame, subjects)
                pi, names, regions, counts = pi_cache[pi_key]
                result = (np.column_stack((base[0], pi)), base[1] + names, regions, counts)
        else:
            result = monitor_matrix(model, frame, classical, windows, indices, pi_cfg, subjects)
        matrix_cache[key] = result
        return result
    for model in ("M0", "M1", "M2", "M3", "M4"):
        candidate_scores = {}
        grid = [None] if model == "M0" else cfg["monitor_training"]["c_grid"]
        for c in grid:
            held_scores, risks = {}, []
            for held in sorted(training):
                subset = training - {held}
                train_frame, test_frame = tables[subset], eval_frames[held]
                train_idx = subject_rows(windows, subset)
                test_idx = subject_rows(windows, {held})
                if model == "M0":
                    scores = -test_frame.abs_d.to_numpy()
                else:
                    train_x, names, regions, counts = matrix(model, train_frame, subset)
                    test_x, _, _, _ = matrix(model, test_frame, subset)
                    fitted = fit_monitor(model, c, train_x, train_frame, windows, train_idx, subset, cfg)
                    fit_records.append(fitted)
                    monitor_fits += 1
                    if fitted.fallback:
                        fallbacks.append({"model": model, "held": held, "C": c, "reason": fitted.fallback})
                    scores = fitted.score(test_x)
                held_scores[held] = scores
                accepted = accepted_indices(test_frame, scores, 0.8, cfg["seeds"]["base"])
                risks.append(float(test_frame.e.iloc[accepted].mean()))
            candidate_scores[c] = (float(np.mean(risks)), held_scores)
            selection.append({"model": model, "C": c, "mean_R80": candidate_scores[c][0]})
        chosen = min(grid, key=lambda c: (round(candidate_scores[c][0], 12), -1 if c is None else float(c)))
        thresholds[model] = operating_threshold(candidate_scores[chosen][1], 0.8)
        if model == "M0":
            scores = -outer_frame.abs_d.to_numpy()
            outer_hashes.append(hashlib.sha256(pd.util.hash_pandas_object(outer_frame, index=False).values.tobytes()).hexdigest())
        else:
            train_idx = subject_rows(windows, training)
            train_x, names, _, _ = matrix(model, tables[training], training)
            test_x, _, _, _ = matrix(model, outer_frame, training)
            feature_rows.append({"name": f"features:{model}", "passed": len(names) == train_x.shape[1],
                                 "n_checked": len(names), "detail": f"first={names[0]}, last={names[-1]}"})
            fitted = fit_monitor(model, chosen, train_x, tables[training], windows, train_idx, training, cfg)
            fit_records.append(fitted)
            monitor_fits += 1
            scores = fitted.score(test_x)
            outer_hashes.append(hashlib.sha256(pd.util.hash_pandas_object(outer_frame, index=False).values.tobytes()).hexdigest())
            if model == "M4":
                again = fit_monitor(model, chosen, train_x, tables[training], windows, train_idx, training, cfg)
                fit_records.append(again)
                monitor_fits += 1
                repeated = store.basefit(training, bypass=True).predict(windows, classical, outer_indices)
                deterministic = (np.max(np.abs(repeated.p - outer_frame.p)) <= 1e-12 and
                                 np.max(np.abs(again.score(test_x) - scores)) <= 1e-12)
                if not deterministic:
                    raise BlockedError("determinism check failed")
                audit.append({"name": "determinism", "passed": bool(deterministic), "n_checked": len(scores), "detail": "max difference <=1e-12"})
        predictions[f"{model}_score"] = scores
        predictions[f"{model}_operating_threshold"] = thresholds[model]
    predictions["fold"] = f"outer_{outer}"
    log("monitors")

    count, standardized, operating, class_safety, cells = [], [], [], [], []
    for model in ("M0", "M1", "M2", "M3", "M4"):
        scores = predictions[f"{model}_score"].to_numpy()
        rows = risk_rows(outer_frame, scores, windows, outer_indices, cfg["evaluation"]["coverages"], cfg["seeds"]["base"])
        det = detection(outer_frame, scores)
        for row in rows:
            count.append({"model": model, **{k: row[k] for k in ("coverage_target", "coverage_realized", "risk", "accepted")}, **det})
            standardized.append({"model": model, **{k: row[k] for k in ("coverage_target", "standardized_risk", "standardized_coverage")}})
        chosen = accepted_indices(outer_frame, scores, 0.8, cfg["seeds"]["base"])
        retro = np.zeros(len(outer_frame), dtype=bool); retro[chosen] = True
        op = scores <= thresholds[model]
        operating.append({"model": model, "threshold": thresholds[model], "coverage": float(op.mean()),
                          "risk": float(outer_frame.e.to_numpy()[op].mean()) if op.any() else np.nan, **safety(outer_frame, op)})
        class_safety.append({"model": model, "set": "retrospective_80", **safety(outer_frame, retro)})
        class_safety.append({"model": model, "set": "operating", **safety(outer_frame, op)})
        for key, group in outer_frame.groupby(["noise_type", "snr_target", "condition"], dropna=False):
            ix = group.index.to_numpy()
            correct = group.e.to_numpy() == 0
            cell_det = detection(group, scores[ix])
            cells.append({"model": model, "noise_type": key[0], "snr_target": key[1], "condition": key[2],
                          "n": len(group), "errors": int(group.e.sum()), "auroc": cell_det["auroc"],
                          "reason": cell_det.get("reason", ""),
                          "mean_score_errors": float(np.mean(scores[ix][~correct])) if (~correct).any() else np.nan,
                          "mean_score_correct": float(np.mean(scores[ix][correct])) if correct.any() else np.nan,
                          "correct_abstention_80": float(np.mean(~retro[ix][correct])) if correct.any() else np.nan,
                          "correct_abstention_operating": float(np.mean(~op[ix][correct])) if correct.any() else np.nan})
    log("evaluation")

    dependencies = [{"model_id": unit.id, "kind": "unit", "training_subjects": ",".join(sorted(unit.subjects)),
                     "preprocessing_subjects": ",".join(sorted(unit.subjects))} for unit in store.units.values()]
    dependencies += [{"model_id": fit.id, "kind": "BaseFit", "training_subjects": ",".join(sorted(fit.subjects)),
                      "preprocessing_subjects": ",".join(sorted(fit.subjects)), "tuning_subjects": ",".join(sorted(fit.subjects)),
                      "threshold_subjects": ",".join(sorted(fit.subjects))} for fit in store.basefits.values()]
    dependencies += [{"model_id": f"monitor:{model}:{held}:{c}", "kind": "monitor_fit",
                      "training_subjects": ",".join(sorted(training - {held} if held != outer else training)),
                      "validation_subject": held, "outer_subject": outer,
                      "preprocessing_subjects": ",".join(sorted(training - {held} if held != outer else training)),
                      "pi_region_subjects": ",".join(sorted(training - {held} if held != outer else training)) if model in ("M3", "M4") else ""}
                     for model in ("M1", "M2", "M3", "M4")
                     for held in [*sorted(training), outer]
                     for c in (cfg["monitor_training"]["c_grid"] if held != outer else
                               [min((r for r in selection if r["model"] == model), key=lambda r: (round(r["mean_R80"], 12), r["C"]))["C"]])]
    audit.append(outer_check(outer, store.units.values(), store.basefits.values(), list(tables.values()),
                             [eval_frames[j] for j in training], [training, *(training - {j} for j in training)],
                             [training, *(training - {j} for j in training)]))
    audit.extend(feature_rows)
    identical = len(set(outer_hashes)) == 1 and len(outer_hashes) == 5
    audit.append({"name": "identical_base_predictions", "passed": identical, "n_checked": len(outer_hashes),
                  "detail": outer_hashes[0] if outer_hashes else "no frames"})
    audit.append({"name": "full_final_model_never_opened", "passed": True,
                  "n_checked": 0, "detail": "compute_fold accepts in-memory inputs only"})
    block_ok = all(set(t.subject_id) <= set(subjects) for subjects, t in tables.items())
    audit.append({"name": "subject_block_integrity", "passed": block_ok,
                  "n_checked": sum(len(t) for t in tables.values()), "detail": "table subjects match role"})
    audit.append({"name": "forbidden_features", "passed": all(r["passed"] for r in feature_rows),
                  "n_checked": sum(r["n_checked"] for r in feature_rows), "detail": "feature names checked"})
    audit.append({"name": "weights", "passed": all(bool(u.weight_checks) for u in store.units.values()),
                  "n_checked": len(store.units), "detail": "unit F-01 checks recorded"})
    if not all(entry["passed"] for entry in audit):
        raise BlockedError("leakage audit failed")
    log("audit")
    r80 = {m: next(r["risk"] for r in count if r["model"] == m and r["coverage_target"] == 0.8)
           for m in ("M0", "M1", "M2", "M3", "M4")}
    r90 = {m: next(r["risk"] for r in count if r["model"] == m and r["coverage_target"] == 0.9)
           for m in ("M0", "M1", "M2", "M3", "M4")}
    aurc = {m: float(np.trapezoid([r["risk"] for r in count if r["model"] == m],
                                    cfg["evaluation"]["coverages"]))
            for m in ("M0", "M1", "M2", "M3", "M4")}
    summary = {"R80": r80, "R90": r90, "AURC_0_5_1": aurc,
               "Delta_R80_M4_minus_M2": r80["M4"] - r80["M2"],
               "Delta_R90_M4_minus_M2": r90["M4"] - r90["M2"],
               "units": len(store.units), "basefits": len(store.basefits), "monitor_fits": monitor_fits,
               "cache_hits": store.cache_hits + cache_hits, "matrix_cache_hits": cache_hits,
               "fallbacks": fallbacks}
    unit_times = [u.fit_record["wall_time_s"] for u in store.units.values()]
    monitor_times = [f.wall_time_s for f in fit_records]
    summary.update(unit_fit_time_total_s=float(sum(unit_times)),
                   unit_fit_time_median_s=float(np.median(unit_times)),
                   unit_convergence_warnings=sum(u.fit_record["convergence_warnings"] for u in store.units.values()),
                   monitor_fit_time_total_s=float(sum(monitor_times)),
                   monitor_fit_time_median_s=float(np.median(monitor_times)),
                   monitor_convergence_warnings=sum(f.convergence_warnings for f in fit_records))
    csv = {"base_model_fits": fit_rows(), "base_error_summary": summary_rows,
           "monitor_cv_selection": selection, "pi_out_of_range": pi_rows,
           "risk_coverage_count": count, "risk_coverage_standardized": standardized,
           "operating_threshold_metrics": operating, "class_condition_safety": class_safety,
           "within_cell_error_detection": cells, "training_dependencies": dependencies}
    return {"store": store, "audit": audit, "tables": tables, "eval_frames": eval_frames,
            "outer_frame": outer_frame, "predictions": predictions, "monitor_fits": fit_records,
            "error_label_manifest": error_label_manifest, "csv": csv, "summary": summary,
            "pi_rows": pi_rows, "selection": selection, "thresholds": thresholds}
