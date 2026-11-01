"""Synthetic checks for resumable Development feature extraction."""
from __future__ import annotations

import copy
import threading
import time

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_stage1 import features


@pytest.fixture
def cfg():
    return load_yaml(features.Path(__file__).resolve().parents[2] / "configs" / "stage1.yaml")


@pytest.mark.parametrize("section,key,value", [
    ("embedding", "tau_samples", 10),
    ("embedding", "m", 3),
    ("vietoris_rips", "r_max", "bounded"),
    ("vietoris_rips", "r_max_value", 1.0),
    ("vietoris_rips", "max_edge_length", 1.0),
])
def test_parameter_guard_rejects_changes(cfg, section, key, value):
    changed = copy.deepcopy(cfg)
    changed[section][key] = value
    with pytest.raises(BlockedError):
        features.require_approved_parameters(changed)


def test_atomic_job_roundtrip_and_cache_key(tmp_path):
    path = tmp_path / "one.npz"
    diagrams = features._empty_diagrams()
    diagrams["vr_h0"] = np.array([[0.0, 1.0], [0.0, np.inf]])
    metadata = {"cache_key": "key-a", "validation_status": "ok", "run_id": "first"}
    features.save_job(path, diagrams, metadata)
    got_diagrams, got_meta = features.load_job(path)
    assert np.isposinf(got_diagrams["vr_h0"][1, 1])
    assert got_meta == metadata
    assert features.cached_metadata(path, "key-a") == metadata
    assert features.cached_metadata(path, "key-b") is None
    metadata["validation_status"] = "failed"
    features.save_job(path, diagrams, metadata)
    assert features.cached_metadata(path, "key-a") is None


def test_validation_flags_wrong_essential_count():
    diagrams = features._empty_diagrams()
    diagrams["vr_h0"] = np.array([[0.0, np.inf], [0.0, np.inf]])
    diagrams["cubical_sub_essential_births"] = np.array([0.0])
    diagrams["cubical_super_essential_births"] = np.array([-1.0])
    metadata = {"vr_status": "success", "cubical_status": "success",
                "num_simplices": features.EXPECTED_SIMPLICES}
    messages = features.validate_job(diagrams, metadata, np.array([0.0, 1.0]))
    assert "vr_h0_essential_count" in messages


def test_compute_job_saves_valid_synthetic_diagrams(monkeypatch, tmp_path, cfg):
    def fake_vr(points, include_diagrams):
        assert include_diagrams and points.shape == (300, 5)
        h0 = np.vstack((np.column_stack((np.zeros(299), np.ones(299))), [[0.0, np.inf]]))
        return {"vr_status": "success", "vr_failure_reason": "", "vr_wall_time_s": 0.1,
                "vr_child_peak_rss_mb": 10.0, "num_simplices": features.EXPECTED_SIMPLICES,
                "h0": h0, "h1": np.empty((0, 2))}
    monkeypatch.setattr(features, "monitored_vr", fake_vr)
    half = np.sin(np.linspace(0, 20, 1800))
    info = {"event_id": "e1", "subject_id": "S1", "record_id": "100",
            "condition": "C0", "half": "pre"}
    path = tmp_path / "e1.npz"
    metadata = features.compute_job(info, half, path, "key", "sha", "cfg", "code", "run",
                                    features.require_approved_parameters(cfg))
    saved, loaded = features.load_job(path)
    assert metadata["validation_status"] == "ok"
    assert loaded["h0_finite"] == 299 and loaded["cubical_sub_essential"] == 1
    assert np.isposinf(saved["vr_h0"][-1, 1])


def test_representative_events_are_deterministic_and_one_per_subject():
    events = pd.DataFrame({"subject_id": ["S2", "S1", "S2", "S1", "S3"],
                           "event_id": ["z", "b", "a", "a", "c"]})
    selected = features.representative_events(events, 2)
    assert selected[["subject_id", "event_id"]].values.tolist() == [["S1", "a"], ["S2", "a"]]


def test_scheduler_limits_active_jobs_and_recomputes_changed_key(monkeypatch, tmp_path, cfg):
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "subject_id": ["S1", "S1", "S2"]})
    half = np.arange(1800, dtype=float)
    def segments(events_arg, _):
        for ev in events_arg.itertuples():
            for condition in ("C0", "C1"):
                for part in ("pre", "post"):
                    yield {"event_id": ev.event_id, "subject_id": ev.subject_id,
                           "record_id": "100", "condition": condition, "half": part}, half
    monkeypatch.setattr(features, "iter_segments", segments)
    monkeypatch.setattr(features, "check_memory", lambda workers: workers)
    lock = threading.Lock()
    active = 0
    maximum = 0
    calls = 0
    def fake_compute(info, half_arg, path, key, sha, cfg_hash, code_hash, run_id, scientific):
        nonlocal active, maximum, calls
        with lock:
            active += 1
            calls += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        metadata = {**info, "cache_key": key, "validation_status": "ok",
                    "vr_child_peak_rss_mb": 2.0, "run_id": run_id}
        features.save_job(path, features._empty_diagrams(), metadata)
        with lock:
            active -= 1
        return metadata
    monkeypatch.setattr(features, "compute_job", fake_compute)
    out = tmp_path / "results"
    first = features.run_features(events, tmp_path, out, cfg, 2, "run1")
    assert first["total_jobs"] == 12 and first["cached"] == 0
    assert maximum <= 2 and calls == 12
    second = features.run_features(events, tmp_path, out, cfg, 2, "run2")
    assert second["cached"] == 12 and calls == 12
    changed = copy.deepcopy(cfg)
    changed["embedding"]["ami_method"] = "different non-computational setting"
    third = features.run_features(events, tmp_path, out, changed, 2, "run3")
    assert third["cached"] == 0 and calls == 24


def test_memory_guard_reports_safe_maximum():
    with pytest.raises(BlockedError, match="safe maximum: 2"):
        features.check_memory(3, available=2_200_000_000)
