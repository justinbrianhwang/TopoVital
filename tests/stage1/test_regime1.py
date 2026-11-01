"""Synthetic Regime 1 allocation, injection, reuse, gating, and cache checks."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.noise import NoiseBlockError
from topovital.util import PROJECT_ROOT, load_yaml
from topovital_stage1 import features, regime1


@pytest.fixture
def cfg():
    return load_yaml(PROJECT_ROOT / "configs" / "stage1_regime1.yaml")


@pytest.fixture
def stage1_cfg():
    return load_yaml(PROJECT_ROOT / "configs" / "stage1.yaml")


def test_allocation_is_deterministic_unique_and_development_only(cfg):
    events = pd.DataFrame({"event_id": [f"e{i:03d}" for i in range(126)]})
    first = regime1.allocation_table(events.sample(frac=1, random_state=1), cfg)
    second = regime1.allocation_table(events.sample(frac=1, random_state=2), cfg)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 126 * 3
    assert first.snippet_id.nunique() == len(first)
    assert first.reuse_count.eq(1).all()
    assert first.snippet_start.ge(0).all()
    assert (first.snippet_start + cfg["noise"]["snippet_samples"] <= 325000).all()
    assert len(range(0, 325000 - 1800 + 1, 1800)) == 180
    for _, group in first.groupby("noise_type"):
        assert group.snippet_start.nunique() == 126


def test_test_block_snippet_rejected_before_slice(cfg):
    signal = np.zeros(325000)
    with pytest.raises(NoiseBlockError):
        regime1.development_snippet(signal, 325000, cfg)


def test_taper_and_snr_for_each_target_with_separate_scales(cfg):
    w = regime1.taper(1800, cfg["noise"]["onset_taper"]["samples"])
    assert w[0] == 0
    assert w[35] < 1
    assert np.array_equal(w[36:], np.ones(1800 - 36))
    t = np.linspace(0, 8 * np.pi, 1800)
    pre = 0.2 * np.sin(t)
    normal = 0.5 * np.sin(t + 0.3)
    pvc = normal + 1.0 * np.exp(-((np.arange(1800) - 900) / 50) ** 2)
    snippet = 0.05 * np.cos(t * 2.5)
    for target in cfg["injection"]["snr_db"]:
        c2, r2 = regime1.inject(pre, normal, snippet, target, cfg)
        c3, r3 = regime1.inject(pre, pvc, snippet, target, cfg)
        assert c2 is not None and c3 is not None
        assert r2["snr_check_passed"] and r3["snr_check_passed"]
        assert abs(r2["snr_actual"] - target) < 1e-3
        assert abs(r3["snr_actual"] - target) < 1e-3
        assert r2["scale_a"] != r3["scale_a"]
        assert r2["noise_norm"] == r3["noise_norm"]


def test_full_run_refused_without_approval(cfg):
    cfg = deepcopy(cfg)
    cfg["approved_for_full_run"] = False
    with pytest.raises(BlockedError, match="approved_for_full_run"):
        regime1.require_run_scope(cfg, None)
    regime1.require_run_scope(cfg, 1)


def test_full_run_allowed_with_approval(cfg):
    cfg = deepcopy(cfg)
    cfg["approved_for_full_run"] = True
    regime1.require_run_scope(cfg, None)


def test_pre_mismatch_is_failure_and_identical_post_key_is_cached(
        monkeypatch, tmp_path, cfg, stage1_cfg):
    events = pd.DataFrame({"event_id": ["e1"], "subject_id": ["S1"]})
    pre = 0.2 * np.sin(np.linspace(0, 10, 1800))
    post = 0.3 * np.sin(np.linspace(0, 12, 1800))
    window = np.concatenate((pre, post))
    pre_norm, _ = regime1.normalize_window(window)
    source = tmp_path / "source" / "jobs"
    monkeypatch.setattr(regime1, "resolve", lambda _: source.parent)
    monkeypatch.setattr(regime1, "_physical_windows", lambda selected, root: iter([
        (type("Ev", (), {"event_id": "e1", "subject_id": "S1"})(), "100", "C0", window),
        (type("Ev", (), {"event_id": "e1", "subject_id": "S1"})(), "100", "C1", window),
    ]))
    # A Development-sized synthetic signal allows every allocated slot to be sliced.
    noise_signal = 0.03 * np.sin(np.linspace(0, 1000, 325000))
    monkeypatch.setattr(regime1, "_noise_signals", lambda config, root: {
        name: noise_signal for name in config["noise"]["types"]})
    monkeypatch.setattr(features, "check_memory", lambda workers: workers)
    calls = []

    def fake_compute(info, half, path, key, sha, cfg_hash, code_sha, run_id, scientific):
        calls.append(path)
        metadata = {**info, "cache_key": key, "input_sha256": sha,
                    "validation_status": "ok", "vr_child_peak_rss_mb": 2.0,
                    "run_id": run_id}
        features.save_job(path, features._empty_diagrams(), metadata)
        return metadata

    monkeypatch.setattr(features, "compute_job", fake_compute)
    for condition in ("C0", "C1"):
        path = features.job_path(source, {"event_id": "e1", "condition": condition, "half": "pre"})
        features.save_job(path, features._empty_diagrams(), {"input_sha256": "wrong"})
    out = tmp_path / "results"
    first = regime1.run_regime1(events, events, tmp_path, tmp_path, out, cfg, stage1_cfg, 2, "run1")
    ledger = pd.read_csv(out / "regime1_ledger.csv")
    assert first["failed"] == first["total_jobs"] == 36
    assert first["snr_checks_passed"] == 36
    assert ledger.pre_reuse_status.eq("mismatch").all()
    assert not calls
    for condition in ("C0", "C1"):
        path = features.job_path(source, {"event_id": "e1", "condition": condition, "half": "pre"})
        features.save_job(path, features._empty_diagrams(),
                          {"input_sha256": features.input_hash(pre_norm)})
    second = regime1.run_regime1(events, events, tmp_path, tmp_path, out, cfg, stage1_cfg, 2, "run2")
    assert second["success"] == 36 and second["cached"] == 0 and len(calls) == 36
    assert any("__m6__post.npz" in path.name for path in calls)
    third = regime1.run_regime1(events, events, tmp_path, tmp_path, out, cfg, stage1_cfg, 2, "run3")
    assert third["success"] == 36 and third["cached"] == 36 and len(calls) == 36
    assert (out / "snippet_allocation.csv").is_file()
    assert (out / "regime1_injection_log.csv").is_file()
