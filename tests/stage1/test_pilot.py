"""Synthetic checks for the Stage 1 Development pilot primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital_stage1 import core
from topovital_stage1.ph import monitored_vr, vr_persistence


def test_normalization_uses_pre_statistics_for_both_halves():
    pre = np.arange(1800, dtype=float)
    post = pre + 10000
    a, b = core.normalize_window(np.r_[pre, post])
    iqr = np.percentile(pre, 75, method="linear") - np.percentile(pre, 25, method="linear")
    assert np.allclose(a, (pre - np.median(pre)) / iqr)
    assert np.allclose(b, (post - np.median(pre)) / iqr)
    assert np.median(b) > 10


def test_delay_vectors_stay_inside_each_half():
    pre = np.arange(1800, dtype=float)
    post = np.arange(1800, 3600, dtype=float)
    for half in (pre, post):
        embedded = core.delay_vectors(half, tau=10, m=5)
        assert embedded.shape == (1800 - 4 * 10, 5)
        assert embedded[0, 0] == half[0]
        assert embedded[-1, -1] == half[-1]
        assert embedded.min() >= half.min()
        assert embedded.max() <= half.max()


def test_fnn_sine_has_small_dimension():
    sine = np.sin(np.linspace(0, 40 * np.pi, 500))
    assert core.fnn_fraction(sine, tau=10, m=3) < 0.05


def test_uniform_subsample_has_300_unique_sorted_indices():
    idx = core.uniform_indices(1760)
    assert len(idx) == 300
    assert idx[0] == 0 and idx[-1] == 1759
    assert np.all(np.diff(idx) > 0)


def test_loader_rejects_test_subject_before_signal_read(monkeypatch, tmp_path):
    audit = pd.DataFrame({
        "event_id": [f"100_{i}" for i in range(126)],
        "subject_id": ["S100"] * 126,
        "record_id": ["100"] * 126,
        "normal_record_id": ["100"] * 126,
        "mlii_channel_index": [0] * 126,
        "fs": [360] * 126,
        "g_ms": [250] * 126,
        "in_common_cohort": [True] * 126,
        "split": ["development"] * 126,
        "start_abs": [0] * 126,
        "boundary_abs": [1800] * 126,
        "end_abs": [3600] * 126,
        "normal_start_sample": [7200] * 126,
    })
    split = pd.DataFrame({"subject_id": ["S100"], "split": ["test"]})
    monkeypatch.setattr(core.pd, "read_parquet", lambda _: audit)
    monkeypatch.setattr(core.pd, "read_csv", lambda *args, **kwargs: split)
    monkeypatch.setattr(core, "read_record_signal", lambda *args: pytest.fail("signal read"))
    with pytest.raises(BlockedError, match="non-Development or Test"):
        core.load_dev_events(tmp_path / "audit.parquet", tmp_path / "split.csv")


def test_tiny_vr_cloud_has_one_infinite_h0():
    result = vr_persistence(np.array([[0.0], [1.0], [2.0]]))
    assert result["h0_infinite"] == 1
    assert result["h0_pairs"] == 3


def test_monitored_vr_child_returns_peak_memory():
    result = monitored_vr(np.array([[0.0], [1.0], [2.0]]))
    assert result["vr_status"] == "success"
    assert result["h0_infinite"] == 1
    assert result["vr_child_peak_rss_mb"] > 0
