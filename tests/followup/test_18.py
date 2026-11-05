import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_followup.error_monitor.aggregate import aggregate_folds, bootstrap, _decomposition, _swap
from topovital_followup.error_monitor.full import check_full
from topovital_followup.error_monitor import full as full_module
from topovital_followup.error_monitor.core import BaseFitStore
from topovital_followup.error_monitor.pilot import compute_fold, threshold_target
from conftest import synthetic


def test_threshold_target_real_config():
    root = Path(__file__).resolve().parents[2]
    assert threshold_target(load_yaml(root / "configs/stage1_factorial.yaml")) == 0.9
    with pytest.raises(BlockedError, match="target missing"):
        threshold_target({})


def test_full_startup_reaches_target(tmp_path, monkeypatch):
    class TargetReached(Exception):
        pass

    cfg = {"full_run": {"output_dir": "full"}, "protected": []}
    identity = {"code_files": {}, "config_sha256": "config", "code_sha256": "code"}
    monkeypatch.setattr(full_module, "check_full", lambda root, config: (identity, True))
    monkeypatch.setattr(full_module, "file_hashes", lambda root, paths: {})
    monkeypatch.setattr(full_module, "_inputs", lambda root, config, code: ([{"subject_id": "S00"}], None, "feature", {}))
    factorial = load_yaml(Path(__file__).resolve().parents[2] / "configs/stage1_factorial.yaml")
    monkeypatch.setattr(full_module, "load_yaml", lambda path: factorial)

    def reached(factorial_cfg):
        assert threshold_target(factorial_cfg) == 0.9
        raise TargetReached

    monkeypatch.setattr(full_module, "threshold_target", reached)
    with pytest.raises(TargetReached):
        full_module.run_full(tmp_path, cfg)


def test_full_guards(tmp_path):
    cfg = {"approved_for_full_run": False, "full_run": {"freeze": {"record": "record.json"}}}
    with pytest.raises(BlockedError, match="approved"):
        check_full(tmp_path, cfg)
    cfg["approved_for_full_run"] = True
    with pytest.raises(BlockedError, match="missing"):
        check_full(tmp_path, cfg)


def test_full_freeze_mismatch_and_dirty(tmp_path, monkeypatch):
    cfg = {"approved_for_full_run": True, "full_run": {"freeze": {"record": "record.json"}}}
    (tmp_path / "record.json").write_text(json.dumps({"git_commit": "old"}))
    monkeypatch.setattr(full_module, "_identity", lambda root: {"git_commit": "new"})
    with pytest.raises(BlockedError, match="identity"):
        check_full(tmp_path, cfg)
    (tmp_path / "record.json").write_text(json.dumps({"git_commit": "new"}))
    monkeypatch.setattr(full_module, "_git", lambda root, *args: " M topovital_followup/x.py")
    with pytest.raises(BlockedError, match="dirty"):
        check_full(tmp_path, cfg)


def test_freeze_refuses_overwrite(tmp_path, monkeypatch):
    cfg = {"full_run": {"freeze": {"record": "record.json"}}}
    (tmp_path / "record.json").write_text("{}")
    monkeypatch.setattr(full_module, "_git", lambda root, *args: "")
    with pytest.raises(BlockedError, match="already exists"):
        full_module.freeze_full(tmp_path, cfg)


def test_bootstrap_shared_draws_and_nan(small_cfg):
    small_cfg["full_run"]["bootstrap"]["resamples"] = 10
    rows = []
    for subject, value in (("S00", 1.), ("S01", 2.)):
        for model in ("M0", "M1", "M2", "M3", "M4"):
            rows.append({"subject_id": subject, "model": model,
                         **{key: value if model == "M4" else 0. for key in
                            ("R80", "R90", "AURC_0.5_1", "standardized_R80", "operating_risk", "operating_coverage")}})
    table, draws = bootstrap(pd.DataFrame(rows), small_cfg)
    first = table.iloc[0]
    assert first.point_estimate == 1.5
    expected = np.array([1., 2.])[draws].mean(axis=1)
    assert first.ci_lower == pytest.approx(np.percentile(expected, 2.5))
    assert table.loc[(table.pair == "M4 - M2") & (table.metric == "R90"), "ci_lower"].iloc[0] == first.ci_lower
    rows[9]["R80"] = np.nan
    table, _ = bootstrap(pd.DataFrame(rows), small_cfg)
    assert table.loc[(table.pair == "M4 - M2") & (table.metric == "R80"), "n_non_evaluable_draws"].iloc[0] > 0


def test_decomposition_and_swap(fold):
    frame = fold[2]["outer_frame"]
    a = np.arange(len(frame)) < int(.8 * len(frame))
    b = np.arange(len(frame)) >= len(frame) - int(.8 * len(frame))
    for row in _decomposition(frame, a, "S04", "M4", "retrospective_80"):
        shares = [v for k, v in row.items() if k.startswith(("accepted_", "abstained_"))]
        assert sum(shares) == pytest.approx(1)
    swap = _swap(frame, {"M4": a, "M2": b}, "S04", "M4", "M2")
    assert sum(r["side"] == "A_only" for r in swap) == sum(r["side"] == "B_only" for r in swap)


def test_aggregate_counts_pilot_once(fold, small_cfg, tmp_path):
    metrics = aggregate_folds({"S04": fold[2]}, small_cfg, tmp_path, pilot=fold[2]["predictions"])
    assert len(metrics) == 5
    assert metrics.subject_id.nunique() == 1
    for name in ("per_subject_metrics.csv", "summary_metrics.csv", "paired_comparisons.csv",
                 "pvc_decomposition.csv", "swap_windows.parquet", "swap_summary.csv",
                 "swap_events.csv", "swap_concentration.csv"):
        assert (tmp_path / name).exists()


def test_two_fold_shared_loop(small_cfg, tmp_path):
    windows, classical = synthetic(n_subjects=5, events_per_subject=1)
    hashes = ("feature", "config", "code")
    store = BaseFitStore(windows, classical, small_cfg["base_model"]["c_grid"], .9, *hashes)
    tables, folds = {}, {}
    for subject in ("S00", "S01"):
        folds[subject] = compute_fold(windows, classical, subject, small_cfg,
                                      threshold_target=.9, hashes=hashes, log=lambda _: None,
                                      store=store, table_cache=tables)
    metrics = aggregate_folds(folds, small_cfg, tmp_path)
    assert len(metrics) == 10
    assert len(tables) > 0
