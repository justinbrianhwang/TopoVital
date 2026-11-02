"""Synthetic checks for the frozen final evaluation pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from topovital.cohort import BlockedError
from topovital_final import final_code_hash
from topovital_final.allocation import test_allocation_table as allocate
from topovital_final.apply import apply_models, score_matrix
from topovital_final.events import load_test_events
from topovital_final.gate import require_test_approval
from topovital_final.verify_dev import _allocation_preview, _channel_preview, _decision_checks
from topovital_final.waveforms import noise_channel

from conftest import requires_data


def _settings(rule="non_overlapping_grid_seeded_permutation_test_block"):
    return {"test_noise_allocation": {"rule": rule, "block": [325000, 650000],
                                       "grid_step_samples": 1800, "snippet_samples": 1800,
                                       "seed": 20261031}, "noise_types": ["bw", "ma", "em"]}


def test_allocation_deterministic_unique_and_disjoint():
    events = pd.DataFrame({"event_id": ["e3", "e1", "e2"]})
    development = pd.DataFrame({"noise_type": ["bw", "ma", "em"],
                                "snippet_start": [0, 1800, 3600]})
    one = allocate(events, _settings(), development)
    two = allocate(events.iloc[::-1], _settings(), development)
    pd.testing.assert_frame_equal(one, two)
    assert one.snippet_id.nunique() == 9
    assert one.snippet_start.min() >= 325000
    assert (one.snippet_start + 1800 <= 650000).all()
    assert not ((one.snippet_start < 325000) & (one.snippet_start + 1800 > 0)).any()


def test_allocation_rejects_pending_and_overlap():
    events = pd.DataFrame({"event_id": ["e1"]})
    with pytest.raises(BlockedError, match="PENDING"):
        allocate(events, _settings("PENDING"))
    first = allocate(events, _settings())
    prior = first[["noise_type", "snippet_start"]]
    with pytest.raises(BlockedError, match="overlap"):
        allocate(events, _settings(), prior)


def test_noise_channel_rules():
    assert noise_channel("fixed_channel_0", 1) == 0
    assert noise_channel("match_ecg_signal_index", 1) == 1
    with pytest.raises(BlockedError, match="PENDING"):
        noise_channel("PENDING", 0)


def test_saved_score_matches_sklearn_pipeline():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 4))
    y = np.r_[np.zeros(30), np.ones(30)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000)).fit(x, y)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    saved = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
             "logistic_coef": classifier.coef_, "logistic_intercept": classifier.intercept_}
    assert np.allclose(score_matrix(x, saved), model.predict_proba(x)[:, 1], atol=1e-12)


def test_apply_models_uses_saved_imputation_and_thresholds():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(20, 2))
    raw[0, 0] = np.nan
    medians = np.nanmedian(raw, axis=0)
    filled = np.where(np.isfinite(raw), raw, medians)
    target = np.r_[np.zeros(10), np.ones(10)]
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(filled, target)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    saved = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
             "logistic_coef": classifier.coef_, "logistic_intercept": classifier.intercept_,
             "candidate": np.array([np.nan, np.nan, 1.0]),
             "feature_names": np.array(["a", "b"]),
             "imputation_columns": np.array(["a", "b"]),
             "imputation_medians": medians}
    windows = [{"event_id": f"e{i}", "subject_id": "S1", "condition": "C0",
                "noise_type": "", "snr_target": None} for i in range(len(raw))]
    classical = pd.DataFrame([{**w, "a": a, "b": b} for w, (a, b) in zip(windows, raw)])
    selection = {"classical_only": {"threshold_primary": .4, "threshold_c1": .5}}
    result = apply_models(windows, classical, {"classical_only": saved}, selection, split="test")
    assert np.allclose(result.score, model.predict_proba(filled)[:, 1], atol=1e-12)
    assert result.split.eq("test").all()
    assert result.threshold_primary.eq(.4).all()
    assert result.threshold_c1.eq(.5).all()


def _event(event_id, subject, record):
    return {"event_id": event_id, "subject_id": subject, "record_id": record,
            "normal_record_id": record, "mlii_channel_index": 0, "fs": 360,
            "g_ms": 250, "in_common_cohort": True, "split": "test",
            "start_abs": 100, "boundary_abs": 1900, "end_abs": 3700,
            "normal_start_sample": 0}


def test_event_loader_counts_and_overlap(tmp_path):
    audit_path = tmp_path / "audit.parquet"
    split_path = tmp_path / "split.csv"
    pd.DataFrame([_event("a", "S100", "100"), _event("b", "S106", "106")]).to_parquet(audit_path)
    pd.DataFrame({"subject_id": ["S100", "S106", "S200"],
                  "split": ["test", "test", "development"]}).to_csv(split_path, index=False)
    settings = {"g_star_ms": 250, "expected_test_events": 2,
                "expected_test_subjects": 2, "test_subjects": ["S100", "S106"]}
    assert len(load_test_events(audit_path, split_path, settings)) == 2
    with pytest.raises(BlockedError, match="count"):
        load_test_events(audit_path, split_path, {**settings, "expected_test_events": 3})
    development = pd.DataFrame({"subject_id": ["S100"], "record_id": ["100"],
                                "normal_record_id": ["100"]})
    with pytest.raises(BlockedError, match="overlap"):
        load_test_events(audit_path, split_path, settings, development)


def _gate_fixture(tmp_path, monkeypatch):
    import topovital_final.gate as gate
    config = tmp_path / "configs" / "final_test_evaluation.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"status": "approved_for_test_execution",
                                     "approved_for_test_execution": True,
                                     "settings": {"rule": "chosen"}}), encoding="utf-8")
    output = tmp_path / "outputs" / "final_test"
    output.mkdir(parents=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(gate, "_inputs", lambda root: {
        "tracked": hashlib.sha256((root / "tracked.txt").read_bytes()).hexdigest()})
    manifest = gate._inputs(tmp_path)
    (output / gate.MANIFEST.name).write_text(json.dumps(manifest), encoding="utf-8")
    return config, output, tracked


def test_gate_status_pending_hash_predictions_incident(tmp_path, monkeypatch):
    config, output, tracked = _gate_fixture(tmp_path, monkeypatch)
    assert require_test_approval(tmp_path)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["status"] = "draft"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(BlockedError, match="approved"):
        require_test_approval(tmp_path)
    data["status"] = "approved_for_test_execution"
    data["settings"]["rule"] = "PENDING"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(BlockedError, match="complete"):
        require_test_approval(tmp_path)
    data["settings"]["rule"] = "chosen"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    tracked.write_text("changed", encoding="utf-8")
    with pytest.raises(BlockedError, match="mismatch"):
        require_test_approval(tmp_path)
    tracked.write_text("frozen", encoding="utf-8")
    predictions = output / "test_predictions.parquet"
    predictions.write_bytes(b"data")
    with pytest.raises(BlockedError, match="quarantine"):
        require_test_approval(tmp_path)
    predictions.unlink()
    profile = output / "resource_profile.csv"
    profile.write_text("command\nrun-test\n", encoding="utf-8")
    with pytest.raises(BlockedError, match="quarantine"):
        require_test_approval(tmp_path)
    profile.unlink()
    (output / "incident_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BlockedError, match="quarantine"):
        require_test_approval(tmp_path)


def test_freeze_inputs_track_config_model_and_code(tmp_path, monkeypatch):
    import topovital_final.gate as gate
    for name in gate.CONFIGS:
        path = tmp_path / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    for name in gate.MEMBERSHIP:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("test_event_manifest.csv"):
            path.write_text("event_id,subject_id,record_id,normal_record_id,mlii_channel_index\n"
                            "e114,S114,114,100,1\n", encoding="utf-8")
        else:
            path.write_text("membership", encoding="utf-8")
    raw_names = [f"mit-bih-arrhythmia-database-1.0.0/{record}.{suffix}"
                 for record in ("100", "114") for suffix in ("dat", "hea", "atr")]
    raw_names += [f"mit-bih-noise-stress-test-database-1.0.0/{noise}.{suffix}"
                  for noise in ("bw", "ma", "em") for suffix in ("dat", "hea")]
    checksum = tmp_path / "outputs/input_checksums.csv"
    with checksum.open("w", encoding="utf-8") as stream:
        stream.write("relative_path,sha256_expected\n")
        for name in raw_names:
            path = tmp_path / "data/raw" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode("utf-8"))
            stream.write(f"{name},{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({f"m{i}": {"threshold_primary": .4,
                                                    "threshold_c1": .5} for i in range(5)}),
                              encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for i in range(5):
        (model_dir / f"m{i}.npz").write_bytes(b"model")
    config = tmp_path / "configs/final_test_evaluation.yaml"
    config.write_text(yaml.safe_dump({"status": "draft",
                                      "approved_for_test_execution": False,
                                      "output": "outputs/final_test/",
                                      "settings": {"models_from": "models",
                                                   "selection_record": "selection.json",
                                                   "workers": 4,
                                                   "test_noise_allocation": {"rule": "non_overlapping_grid_seeded_permutation_test_block"},
                                                   "noise_channel_rule": "match_ecg_signal_index"}}), encoding="utf-8")
    version = {"code": "v1"}
    monkeypatch.setattr(gate, "final_code_hash", lambda: version["code"])
    monkeypatch.setattr(gate.importlib.metadata, "version", lambda name: "1")
    frozen = gate._inputs(tmp_path)
    assert frozen["record_114"] == {"in_test": True, "subject_id": "S114",
                                    "n_events": 1, "nstdb_channel": 1}
    assert frozen["decisions"]["U-02"]["per_record_channels"][0]["nstdb_channel_name"] == "noise2"
    assert len(frozen["raw_data"]) == 12
    assert "configs/final_test_evaluation.yaml" not in frozen["files"]
    assert "final_test_settings_sha256" in frozen
    gate.validate_freeze_manifest(tmp_path, frozen)
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["status"] = "approved_for_test_execution"
    data["approved_for_test_execution"] = True
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    gate.validate_freeze_manifest(tmp_path, frozen)
    data["settings"]["workers"] = 5
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(BlockedError, match="mismatch"):
        gate.validate_freeze_manifest(tmp_path, frozen)
    data["settings"]["workers"] = 4
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    gate.validate_freeze_manifest(tmp_path, frozen)
    (tmp_path / "configs/stage1.yaml").write_text("changed", encoding="utf-8")
    with pytest.raises(BlockedError, match="mismatch"):
        gate.validate_freeze_manifest(tmp_path, frozen)
    (tmp_path / "configs/stage1.yaml").write_text("x", encoding="utf-8")
    (model_dir / "m0.npz").write_bytes(b"changed")
    with pytest.raises(BlockedError, match="mismatch"):
        gate.validate_freeze_manifest(tmp_path, frozen)
    (model_dir / "m0.npz").write_bytes(b"model")
    version["code"] = "v2"
    with pytest.raises(BlockedError, match="mismatch"):
        gate.validate_freeze_manifest(tmp_path, frozen)
    version["code"] = "v1"
    raw = tmp_path / "data/raw" / raw_names[0]
    raw.write_bytes(b"changed")
    with pytest.raises(BlockedError, match="Raw data SHA256 mismatch"):
        gate.validate_freeze_manifest(tmp_path, frozen)


def test_final_code_hash_normalizes_crlf(tmp_path):
    source = tmp_path / "a.py"
    source.write_bytes(b"print(1)\r\n")
    before = final_code_hash(tmp_path)
    source.write_bytes(b"print(1)\n")
    assert final_code_hash(tmp_path) == before
    source.write_bytes(b"print(2)\n")
    assert final_code_hash(tmp_path) != before


def test_entrypoint_sets_threads_before_numpy_import():
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.pop(name, None)
    script = ("import topovital_final.__main__; import os; "
              "print([os.environ.get(x) for x in "
              "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',"
              "'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')])")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, check=False, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
    assert "['1', '1', '1', '1', '1']" in result.stdout


def test_draft_approval_gate_blocks_before_data_access(tmp_path):
    from topovital_final.gate import require_test_approval

    config = tmp_path / "configs/final_test_evaluation.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"status": "frozen_pending_test_approval",
                                      "approved_for_test_execution": False,
                                      "settings": {"noise_channel_rule": "match_ecg_signal_index"}}),
                      encoding="utf-8")
    with pytest.raises(BlockedError, match="approved"):
        require_test_approval(tmp_path)


def test_membership_command_writes_only_membership(tmp_path, monkeypatch):
    import topovital_final.__main__ as cli

    events = pd.DataFrame({"event_id": ["e1"], "subject_id": ["S100"],
                           "record_id": ["100"], "mlii_channel_index": [1]})
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "verify_blas_threads", lambda: None)
    monkeypatch.setattr(cli, "load_yaml", lambda path: {"settings": {"workers": 1}})
    monkeypatch.setattr(cli, "_membership", lambda root, settings: (events, None))
    assert cli.main(["test-membership"]) == 0
    output = tmp_path / "outputs/final_test"
    assert [p.name for p in output.iterdir()] == ["test_event_manifest.csv"]
    pd.testing.assert_frame_equal(pd.read_csv(output / "test_event_manifest.csv",
                                              dtype={"record_id": str}), events)


def test_pending_decision_previews_use_membership_and_slots():
    events = pd.DataFrame({"event_id": ["e2", "e1"], "record_id": ["100", "106"],
                           "mlii_channel_index": [1, 0]})
    development = pd.DataFrame({"snippet_start": [0, 1800],
                                "noise_type": ["bw", "ma"]})
    settings = _settings("PENDING")
    settings["expected_test_events"] = 2
    settings["noise_channel_rule"] = "PENDING"
    preview = _allocation_preview(events, settings, development)
    assert preview == {"events": 2, "unique_snippets": 6,
                       "min_start": preview["min_start"], "max_end": preview["max_end"],
                       "overlap_with_development": False, "reuse_count_max": 1}
    assert preview["min_start"] >= 325000 and preview["max_end"] <= 650000
    assert _channel_preview(events) == [
        {"record_id": "100", "mlii_channel_index": 1,
         "fixed_channel_0": 0, "match_ecg_signal_index": 1},
        {"record_id": "106", "mlii_channel_index": 0,
         "fixed_channel_0": 0, "match_ecg_signal_index": 0}]
    checks = _decision_checks(events, settings, development)
    assert [check["check"] for check in checks] == ["test_allocation_dry_run", "noise_channel_rule"]
    assert all(check["passed"] is None and
               check["status"] == "pending_researcher_decision" for check in checks)
    assert checks[0]["preview"] == preview
    assert checks[1]["preview"] == _channel_preview(events)
    settings["test_noise_allocation"]["rule"] = "non_overlapping_grid_seeded_permutation_test_block"
    settings["noise_channel_rule"] = "match_ecg_signal_index"
    assert all(check["passed"] is True for check in _decision_checks(events, settings, development))


def test_raw_checksum_row_missing_blocks_freeze(tmp_path):
    from topovital_final.gate import _raw_data

    events = pd.DataFrame([{"record_id": "114", "normal_record_id": "100"}])
    checksum = tmp_path / "outputs/input_checksums.csv"
    checksum.parent.mkdir(parents=True)
    checksum.write_text("relative_path,sha256_expected\n", encoding="utf-8")
    with pytest.raises(BlockedError, match="row missing"):
        _raw_data(tmp_path, events)


def test_common_evaluable_excludes_failed_window_for_every_model():
    from topovital_final.evaluate import common_evaluable

    models = [f"m{i}" for i in range(5)]
    rows = [{"event_id": event, "subject_id": "S114", "condition": "C0",
             "noise_type": "", "snr_target": None, "model": model,
             "score": np.nan if event == "failed" and model == "m4" else .5,
             "reason": "ph_job_failed" if event == "failed" and model == "m4" else ""}
            for model in models for event in ("good", "failed")]
    predictions = pd.DataFrame(rows)
    common, summary, counts = common_evaluable(predictions)
    assert len(common) == 5
    assert set(common.event_id) == {"good"}
    assert counts["common_evaluable_windows"] == 1
    assert counts["excluded_windows"] == 1
    assert counts["excluded_by_subject"] == {"S114": 1}
    excluded = summary.loc[summary.row_type.eq("excluded_window")].iloc[0]
    assert excluded["reason"] == "ph_job_failed"
    assert excluded["finite_m4"] == 0
    assert summary.loc[summary.row_type.eq("count"), "total_windows"].iloc[0] == 2


def test_evaluate_zero_failures_passes_previous_rows_to_metrics(tmp_path, monkeypatch):
    import topovital_final.evaluate as evaluation

    predictions = pd.DataFrame([
        {"event_id": "e", "subject_id": "S114", "condition": "C0", "noise_type": "",
         "snr_target": None, "model": f"m{i}", "score": .5, "reason": ""}
        for i in range(5)])
    seen = []
    def metrics(frame, cfg):
        seen.append(frame.copy())
        return tuple(pd.DataFrame({"value": [1]}) for _ in range(4))
    def compare(frame, cfg_l3, cfg_f, cfg_a):
        seen.append(frame.copy())
        return pd.DataFrame({"difference": [.1], "ci_low": [0.]})
    monkeypatch.setattr(evaluation, "metric_tables", metrics)
    monkeypatch.setattr(evaluation, "model_comparison", compare)
    cfg_f = {"reporting": {"bootstrap": {}}}
    cfg_l3 = {"comparisons": {}}
    result = evaluation.evaluate(predictions, {"bootstrap": {}, "primary_comparison": "m1 - m0",
                                               "secondary_comparisons": []}, cfg_l3, cfg_f,
                                 {"scope": {}}, tmp_path)
    assert len(seen) == 3
    for frame in seen:
        pd.testing.assert_frame_equal(frame, predictions)
    assert result["common_evaluable_windows"] == 1
    assert result["excluded_windows"] == 0


def test_paired_metric_drops_only_orphaned_injected_rows(monkeypatch):
    import topovital_final.evaluate as evaluation

    rows = []
    for model in (f"m{i}" for i in range(5)):
        for event, condition in (("paired", "C0"), ("paired", "C2"),
                                 ("orphan", "C2")):
            rows.append({"event_id": event, "subject_id": "S114", "condition": condition,
                         "noise_type": "bw" if condition == "C2" else "",
                         "snr_target": 24 if condition == "C2" else None,
                         "model": model, "score": .5})
    frame = pd.DataFrame(rows)
    seen = []
    def metrics(data, cfg):
        seen.append(data.copy())
        result = pd.DataFrame({"metric": ["paired_change_normal", "other"],
                               "value": [len(data), len(data)]})
        return (result.copy(),)
    monkeypatch.setattr(evaluation, "metric_tables", metrics)
    (result,) = evaluation._metric_tables_with_pairs(frame, {})
    assert [len(item) for item in seen] == [15, 10]
    assert result.value.tolist() == [10, 15]


def test_failure_log_records_ph_and_injection_and_empty_headers():
    from topovital_final.features import FAILURE_COLUMNS, failure_log

    window = {"event_id": "e114", "subject_id": "S114", "record_id": "114",
              "condition": "C2", "noise_type": "bw", "snr_target": 24,
              "physical_post": None, "failure_reason": "injection_failed"}
    ledger = pd.DataFrame([{**{k: window.get(k) for k in FAILURE_COLUMNS[:-2]},
                            "half": "pre", "validation_status": "failed",
                            "validation_messages": "ph_failed"}])
    log = failure_log([window], ledger)
    assert log.reason.tolist() == ["ph_failed", "injection_failed"]
    assert log.half.tolist() == ["pre", "post"]
    assert list(failure_log([], ledger.iloc[0:0]).columns) == FAILURE_COLUMNS


def test_noise_channel_audit_uses_synthetic_two_channel_snippets(monkeypatch, tmp_path):
    import topovital_final.waveforms as waveforms

    event = pd.DataFrame([{"event_id": "e114", "subject_id": "S114",
                           "record_id": "114", "mlii_channel_index": 1}])
    allocation = pd.DataFrame([{"event_id": "e114", "noise_type": noise,
                                "snippet_start": 325000, "snippet_id": f"{noise}_325000",
                                "reuse_count": 1} for noise in ("bw", "ma", "em")])
    half = np.arange(1800, dtype=float)
    physical = np.r_[half, half + 1]
    def segments(events, ecg):
        for condition in ("C0", "C1"):
            for label in ("pre", "post"):
                yield {"event_id": "e114", "condition": condition, "half": label}, half
    def physical_windows(events, ecg):
        for condition in ("C0", "C1"):
            yield events.iloc[0], "114", condition, physical
    class Record:
        fs = waveforms.FS
        def __init__(self, channel):
            self.p_signal = (half if channel == 0 else -2 * half).reshape(-1, 1)
    monkeypatch.setattr(waveforms, "iter_segments", segments)
    monkeypatch.setattr(waveforms, "_physical_windows", physical_windows)
    monkeypatch.setattr(waveforms.wfdb, "rdrecord", lambda *args, channels, **kwargs: Record(channels[0]))
    monkeypatch.setattr(waveforms, "inject", lambda pre, post, snippet, snr, cfg:
                        (post, {"snr_actual": snr, "scale_a": 1,
                                "injection_failure_reason": ""}))
    settings = {**_settings(), "noise_channel_rule": "match_ecg_signal_index", "snr_db": [24]}
    windows, _, audit = waveforms.make_waveforms(event, allocation, settings,
        {"noise": {"snippet_samples": 1800}}, tmp_path, split="test")
    assert len(windows) == 8
    assert len(audit) == 3
    assert audit.channel_used.eq(1).all()
    assert np.allclose(audit.pearson_correlation, -1)
    assert np.allclose(audit.channel_1_rms, 2 * audit.channel_0_rms)


def test_verify_dev_refuses_existing_report(tmp_path):
    from topovital_final.verify_dev import verify_dev

    path = tmp_path / "outputs/final_test/dev_verification_2"
    path.mkdir(parents=True)
    (path / "dev_verification_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BlockedError, match="already exists"):
        verify_dev(tmp_path, {}, workers=1, output_dir="outputs/final_test/dev_verification_2")


@requires_data
def test_saved_development_model_feature_order():
    """Use one Development event to check the real artifact schema, without Test data."""
    from topovital.util import PROJECT_ROOT, load_yaml
    from topovital_stage1.core import load_dev_events
    from topovital_analysis.level2 import _load_windows
    from topovital_final.apply import design_matrix, load_models

    root = PROJECT_ROOT
    events = load_dev_events(root / "outputs/development/event_audit_dev.parquet",
                             root / "outputs/split_assignment.csv").head(1)
    windows = _load_windows(events, load_yaml(root / "configs/stage1_analysis.yaml"), root)
    classical = pd.read_parquet(
        root / "outputs/stage1_analysis/development/level3_features/classical_features.parquet")
    keys = ["event_id", "subject_id", "condition", "noise_type", "snr_target"]
    classical = pd.DataFrame([{k: w[k] for k in keys} for w in windows]).merge(
        classical, on=keys, how="left", validate="one_to_one")
    settings = load_yaml(root / "configs/final_test_evaluation.yaml")["settings"]
    selection = Path(settings["selection_record"])
    files = {selection.as_posix(): hashlib.sha256((root / selection).read_bytes()).hexdigest()}
    for name in json.loads((root / selection).read_text(encoding="utf-8")):
        path = Path(settings["models_from"]) / f"{name}.npz"
        files[path.as_posix()] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    models, chosen = load_models(root, settings, {"files": files})
    scores = apply_models(windows, classical, models, chosen, split="development")
    assert len(scores) == len(windows) * 5
    assert scores.score.notna().all()
    assert all(design_matrix(windows, classical, model).shape[1] == len(model["feature_names"])
               for model in models.values())
