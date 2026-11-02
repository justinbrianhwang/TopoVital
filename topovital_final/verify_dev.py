"""Development-only replay and final model verification."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info

from topovital.cohort import BlockedError
from topovital.util import atomic_write_json, load_yaml
from topovital_analysis.level2 import _load_windows
from topovital_analysis.level3 import KEYS, _refit_level3
from topovital_analysis.level1 import analysis_code_hash
from topovital_analysis.runtime import ANALYSIS_CODE_VERSION
from topovital_stage1.core import load_dev_events
from topovital_stage1.features import DIAGRAM_KEYS, load_job

from .allocation import test_allocation_table
from .apply import apply_models, design_matrix, load_models, score_matrix
from . import final_code_hash
from .features import job_paths, load_windows, run_features
from .gate import _sha
from .waveforms import make_waveforms


def _allocation_preview(test_events: pd.DataFrame, settings: dict,
                        development: pd.DataFrame) -> dict:
    proposed = dict(settings)
    proposed["test_noise_allocation"] = {
        **settings["test_noise_allocation"],
        "rule": "non_overlapping_grid_seeded_permutation_test_block"}
    slots = test_allocation_table(test_events, proposed)
    length = int(settings["test_noise_allocation"]["snippet_samples"])
    prior = development.snippet_start.astype(int).to_numpy()
    starts = slots.snippet_start.to_numpy(dtype=int)
    overlap = bool(((starts[:, None] < prior[None, :] + length) &
                    (prior[None, :] < starts[:, None] + length)).any())
    return {"events": len(test_events), "unique_snippets": int(slots.snippet_id.nunique()),
            "min_start": int(starts.min()), "max_end": int(starts.max() + length),
            "overlap_with_development": overlap,
            "reuse_count_max": int(slots.reuse_count.max())}


def _channel_preview(test_events: pd.DataFrame) -> list[dict]:
    records = test_events[["record_id", "mlii_channel_index"]].drop_duplicates()
    return [{"record_id": str(row.record_id),
             "mlii_channel_index": int(row.mlii_channel_index),
             "fixed_channel_0": 0,
             "match_ecg_signal_index": int(row.mlii_channel_index)}
            for row in records.sort_values("record_id", kind="stable").itertuples(index=False)]


def _decision_checks(test_events: pd.DataFrame, settings: dict,
                     development: pd.DataFrame) -> list[dict]:
    preview = _allocation_preview(test_events, settings, development)
    block = settings["test_noise_allocation"]["block"]
    if settings["test_noise_allocation"].get("rule") == "PENDING":
        allocation = {"check": "test_allocation_dry_run", "value": "PENDING",
                      "passed": None, "status": "pending_researcher_decision",
                      "preview": preview}
    else:
        test_allocation_table(test_events, settings, development)
        allocation = {"check": "test_allocation_dry_run", "value": preview,
                      "passed": preview["events"] == int(settings["expected_test_events"]) and
                      preview["unique_snippets"] == len(test_events) * len(settings["noise_types"]) and
                      preview["min_start"] >= block[0] and preview["max_end"] <= block[1] and
                      not preview["overlap_with_development"] and preview["reuse_count_max"] == 1}
    channel_rule = settings["noise_channel_rule"]
    if channel_rule == "PENDING":
        channel = {"check": "noise_channel_rule", "value": "PENDING", "passed": None,
                   "status": "pending_researcher_decision",
                   "preview": _channel_preview(test_events)}
    else:
        channel = {"check": "noise_channel_rule", "value": _channel_preview(test_events),
                   "rule": channel_rule,
                   "passed": channel_rule == "match_ecg_signal_index" and
                   all(row["match_ecg_signal_index"] in (0, 1)
                       for row in _channel_preview(test_events))}
    return [allocation, channel]


def _sklearn_scores(matrix: np.ndarray, saved: dict) -> np.ndarray:
    scaler = StandardScaler()
    scaler.mean_ = saved["scaler_mean"]
    scaler.scale_ = saved["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    classifier = LogisticRegression()
    classifier.coef_ = saved["logistic_coef"]
    classifier.intercept_ = saved["logistic_intercept"]
    classifier.classes_ = np.array([False, True])
    classifier.n_features_in_ = len(scaler.mean_)
    return make_pipeline(scaler, classifier).predict_proba(matrix)[:, 1]


def verify_dev(root: Path, settings: dict, *, workers: int,
               output_dir: str = "outputs/final_test/dev_verification") -> dict:
    root = Path(root)
    relative = Path(output_dir)
    if (relative.is_absolute() or ".." in relative.parts or relative == Path(".") or
            not (root / relative).resolve().is_relative_to(root.resolve())):
        raise BlockedError("verify-dev output directory must be repo-relative")
    output = root / relative
    if (output / "dev_verification_report.json").exists():
        raise BlockedError("Development verification report already exists")
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    def record(name, value, passed, *, preview=None):
        check = {"check": name, "value": value, "passed": None if passed is None else bool(passed)}
        if passed is None:
            check["status"] = "pending_researcher_decision"
            check["preview"] = preview
        checks.append(check)

    cfg_s1 = load_yaml(root / "configs/stage1.yaml")
    cfg_r1 = load_yaml(root / "configs/stage1_regime1.yaml")
    cfg_l3 = load_yaml(root / "configs/stage1_level3.yaml")
    cfg_a = load_yaml(root / "configs/stage1_analysis.yaml")
    cfg_s0 = load_yaml(root / "configs/stage0.yaml")
    try:
        events = load_dev_events(root / "outputs/development/event_audit_dev.parquet",
                                 root / "outputs/split_assignment.csv")
        selected = events.sort_values("event_id").drop_duplicates("subject_id").head(3)
        record("three_distinct_development_subjects", selected.subject_id.tolist(),
               len(selected) == selected.subject_id.nunique() == 3)
        stage = cfg_s1
        parameters = {"points": stage["vietoris_rips"]["n_points_per_half"],
                      "tau": stage["embedding"]["tau_samples"],
                      "m": stage["embedding"]["m"],
                      "max_dimension": stage["vietoris_rips"]["max_dimension"],
                      "r_max": str(stage["vietoris_rips"]["r_max_value"]),
                      "snr_tolerance": cfg_r1["injection"]["tolerance"]["abs_snr_db"]}
        record("frozen_parameters", parameters,
               parameters == {"points": 300, "tau": 28, "m": 5, "max_dimension": 2,
                              "r_max": "inf", "snr_tolerance": 1e-3})
        job_dirs = [root / "outputs/final_test/jobs", root / "outputs/final_test/regime1_jobs"]
        predictions = list((root / "outputs").rglob("*test*predictions*"))
        job_contents = {p.relative_to(root).as_posix():
                        [f.relative_to(root).as_posix() for f in p.iterdir()] if p.exists() else []
                        for p in job_dirs}
        record("no_test_outputs", {"jobs_content": job_contents,
                                   "prediction_paths": [p.relative_to(root).as_posix() for p in predictions]},
               not any(job_contents.values()) and not predictions)
        dev_alloc = pd.read_csv(root / "outputs/stage1_features/development/snippet_allocation.csv")
        test_manifest = root / "outputs/final_test/test_event_manifest.csv"
        if test_manifest.exists():
            test_events = pd.read_csv(test_manifest)
            checks.extend(_decision_checks(test_events, settings, dev_alloc))
        else:
            record("test_allocation_dry_run", "Test event membership manifest missing", False)
            record("noise_channel_rule", "Test event membership manifest missing", False)
        replay_settings = dict(settings)
        replay_settings["snr_db"] = [24, -6]
        selected_alloc = dev_alloc.loc[dev_alloc.event_id.astype(str).isin(selected.event_id.astype(str))]
        waveforms, snr_audit, _ = make_waveforms(selected, selected_alloc, replay_settings,
                                                cfg_r1, root, split="development")
        record("replay_snr", float(np.max(np.abs(snr_audit.snr_actual.dropna() -
                                               snr_audit.snr_target.dropna()))),
               bool(snr_audit.passed.all()))
        replay = output / "replay"
        classical, failures, ledger = run_features(waveforms, replay, cfg_s1, cfg_s0,
                                                   workers=workers, run_id="dev_verification")
        original = root / "outputs/stage1_features/development"
        diagrams_ok, hashes_ok, compared = True, True, 0
        for w in waveforms:
            new_pre, new_post = job_paths(replay, w)
            old_pre, old_post = job_paths(original, w)
            for new_path, old_path in ((new_pre, old_pre), (new_post, old_post)):
                if w["condition"] in ("C2", "C3") and new_path == new_pre:
                    continue
                new_d, new_m = load_job(new_path)
                old_d, old_m = load_job(old_path)
                hashes_ok &= new_m["input_sha256"] == old_m["input_sha256"]
                diagrams_ok &= all(np.array_equal(new_d[k], old_d[k]) for k in DIAGRAM_KEYS)
                compared += 1
        record("replay_ph_diagrams", compared, diagrams_ok)
        record("replay_input_sha256", compared, hashes_ok)
        old_classical = pd.read_parquet(root / "outputs/stage1_analysis/development/level3_features/classical_features.parquet")
        joined = classical.merge(old_classical, on=list(KEYS), suffixes=("_new", "_old"),
                                 how="left", validate="one_to_one", indicator=True)
        numeric = [c for c in classical if c not in KEYS]
        differences = [np.max(np.nan_to_num(np.abs(joined[f"{c}_new"].to_numpy(dtype=float) -
                                                   joined[f"{c}_old"].to_numpy(dtype=float)), nan=0.0), initial=0)
                       for c in numeric if joined[f"{c}_new"].notna().any()]
        maximum = float(np.nanmax(differences)) if differences else 0.0
        record("replay_classical_features", maximum,
               joined._merge.eq("both").all() and maximum <= 1e-12 and
               all(np.array_equal(joined[f"{c}_new"].isna(), joined[f"{c}_old"].isna()) for c in numeric))
        all_windows = _load_windows(events, cfg_a, root)
        classical_all = old_classical
        keys = pd.DataFrame([{k: w[k] for k in KEYS} for w in all_windows])
        classical_all = keys.merge(classical_all, on=list(KEYS), how="left", validate="one_to_one")
        manifest = {"files": {str(Path(settings["selection_record"]).as_posix()):
                              _sha(root / settings["selection_record"])}}
        for name in json.loads((root / settings["selection_record"]).read_text(encoding="utf-8")):
            path = Path(settings["models_from"]) / f"{name}.npz"
            manifest["files"][path.as_posix()] = _sha(root / path)
        models, selection = load_models(root, settings, manifest)
        score_diffs = {}
        applied = apply_models(all_windows, classical_all, models, selection, split="development")
        record("development_application_rows", len(applied),
               len(all_windows) == 4788 and len(applied) == 5 * 4788 and
               applied.score.notna().all())
        for name, saved in models.items():
            matrix = design_matrix(all_windows, classical_all, saved)
            score_diffs[name] = float(np.max(np.abs(score_matrix(matrix, saved) -
                                                    _sklearn_scores(matrix, saved))))
        record("saved_model_sklearn_scores", score_diffs,
               len(all_windows) == 4788 and max(score_diffs.values()) <= 1e-12)
        region_checks = {}
        for name, saved in models.items():
            actual = {rep: [None if np.isnan(x) else float(x) for x in saved["regions"][i]]
                      for i, rep in enumerate(saved["region_names"])}
            region_checks[name] = actual == selection[name]["pi_regions"]
        record("saved_pi_regions", region_checks, all(region_checks.values()))
        record("cubical_finite_essential_handling", list(DIAGRAM_KEYS),
               all(key in DIAGRAM_KEYS for key in ("cubical_sub_finite", "cubical_sub_essential_births",
                                                       "cubical_super_finite", "cubical_super_essential_births")))
        selected_models = {name: ((entry["resolution"], entry["sigma_pixels"], entry["C"]),
                                  entry["threshold_primary"], entry["threshold_c1"], entry["cv_score"])
                           for name, entry in selection.items()}
        refit_dir = output / "refit"
        all_indices = list(range(len(all_windows)))
        _refit_level3(all_windows, classical_all, all_indices, all_indices[:1], selected_models,
                      cfg_l3, cfg_a, list(cfg_a["inputs"]["representations"]),
                      cfg_a["level2"]["persistence_image"]["range_quantiles"],
                      [], [], [], "final_dev_verification", save=refit_dir)
        refit_diff = {}
        for name, saved in models.items():
            with np.load(refit_dir / "final_models" / f"{name}.npz", allow_pickle=False) as rebuilt:
                refit_diff[name] = {key: float(np.max(np.abs(rebuilt[key] - saved[key])))
                                    for key in ("logistic_coef", "logistic_intercept")}
        record("final_refit_parameters", refit_diff,
               max(max(v.values()) for v in refit_diff.values()) <= 1e-10)
        ledgers = [pd.read_csv(original / "jobs_ledger.csv"),
                   pd.read_csv(original / "regime1_ledger.csv")]
        all_jobs = pd.concat(ledgers, ignore_index=True)
        counts = {"clean_jobs": 81 * 2 * 2, "injected_jobs": 81 * 2 * 3 * 6}
        mean_job_s = float(all_jobs.vr_wall_time_s.fillna(0).mean() +
                           all_jobs.cubical_wall_time_s.fillna(0).mean())
        benchmark = {**counts, "mean_job_s": mean_job_s,
                     "replay_mean_job_s": float(ledger.vr_wall_time_s.fillna(0).mean() +
                                                ledger.cubical_wall_time_s.fillna(0).mean()),
                     "cpu_hours": (counts["clean_jobs"] + counts["injected_jobs"]) * mean_job_s / 3600,
                     "wall_hours_at_workers": (counts["clean_jobs"] + counts["injected_jobs"]) * mean_job_s /
                                              (3600 * workers),
                     "peak_rss_mb_per_vr_job": float(all_jobs.vr_child_peak_rss_mb.max()),
                     "workers": workers}
        record("resource_benchmark", benchmark, math.isfinite(mean_job_s) and workers > 0)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for separator in ("\\", "/"):
            message = message.replace(str(root.resolve()) + separator, "")
        record("verification_exception", message.replace("\\", "/"), False)
    report = {"checks": checks,
              "passed": bool(checks) and all(c["passed"] for c in checks if c["passed"] is not None),
              "pending_decisions": [c for c in checks if c["passed"] is None],
              "threadpool_info": [{key: value for key, value in item.items() if key != "filepath"}
                                  for item in threadpool_info()],
              "analysis_code_version": ANALYSIS_CODE_VERSION,
              "final_code_hash": final_code_hash(),
              "analysis_code_hash": analysis_code_hash()}
    atomic_write_json(output / "dev_verification_report.json", report)
    if not report["passed"]:
        raise BlockedError("Development verification failed; see dev_verification_report.json")
    return report
