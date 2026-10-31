import json
import shutil

import pytest

from conftest import MITDB, MITDB_RECORDS, requires_data
from topovital import cohort as C
from topovital.cli import EXIT_BLOCKED, main
from topovital.gate import (build_locked_config, freeze, pending_settings,
                            require_dev_sweep_allowed, require_test_allowed)
from topovital.util import PROJECT_ROOT, atomic_write_yaml

TEST_ONLY_SEED = 777  # mechanism test only; not a protocol value


def test_pending_settings_detected(cfg):
    assert pending_settings(cfg) == []
    require_dev_sweep_allowed(cfg)
    cfg["split"]["seed"] = None
    with pytest.raises(C.BlockedError):
        require_dev_sweep_allowed(cfg)


def test_test_split_g_sweep_always_blocked(tmp_path, cfg):
    with pytest.raises(C.BlockedError):
        require_test_allowed(tmp_path / "locked_config.yaml", g_sweep=True)


def test_test_blocked_without_or_with_pending_lock(tmp_path, cfg):
    lp = tmp_path / "locked_config.yaml"
    with pytest.raises(C.BlockedError):
        require_test_allowed(lp, g_sweep=False)
    atomic_write_yaml(lp, build_locked_config(cfg, "pending", None, pending_settings(cfg)))
    with pytest.raises(C.BlockedError, match="stage0_freeze_status='pending'"):
        require_test_allowed(lp, g_sweep=False)


def test_test_blocked_if_frozen_config_tampered(tmp_path, cfg):
    lp = tmp_path / "locked_config.yaml"
    locked = build_locked_config(cfg, "frozen", 250, [])
    locked["stage0"]["time"]["g_star_ms"] = 300  # edit after freezing
    atomic_write_yaml(lp, locked)
    with pytest.raises(C.BlockedError, match="modified"):
        require_test_allowed(lp, g_sweep=False)


def test_freeze_blocked_while_pending(tmp_path, cfg):
    dec = tmp_path / "d.yaml"
    atomic_write_yaml(dec, {"g_star": 250})
    with pytest.raises(C.BlockedError):
        freeze(cfg, dec, tmp_path)
    assert not (tmp_path / "locked_config.yaml").exists()


def test_cli_test_sweep_returns_blocked_exit(tmp_path):
    assert main(["--out", str(tmp_path), "stage0", "--split", "test", "--g-sweep"]) == EXIT_BLOCKED
    assert main(["--out", str(tmp_path), "stage0", "--split", "test"]) == EXIT_BLOCKED
    assert "blocked" in (tmp_path / "resource_profile.csv").read_text(encoding="utf-8")


def test_split_deterministic(cfg):
    subs = sorted({C.subject_of(r, [["201", "202"]]) for r in MITDB_RECORDS
                   if r not in {"102", "104", "107", "217"}})
    sc = {**cfg["split"], "seed": TEST_ONLY_SEED, "stratification": "none"}
    assert C.split_subjects(subs, sc) == C.split_subjects(list(reversed(subs)), sc)


@requires_data
def test_real_data_cohort_run_is_reproducible(tmp_path, cfg):
    """Two independent runs on the real data produce the identical event list and counts."""
    chk = PROJECT_ROOT / "outputs" / "input_checksums.csv"
    if not chk.exists():
        pytest.skip("run `python -m topovital fetch` first")
    from topovital.stage0 import run_cohort

    sums = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        out.mkdir()
        shutil.copy(chk, out / "input_checksums.csv")
        sums.append(run_cohort(cfg, out, f"test{i}", {"cache_used": False}))
    keys = ("n_cohort_records", "n_cohort_subjects", "n_step1_pvc_candidates", "event_list_sha256")
    assert {k: sums[0][k] for k in keys} == {k: sums[1][k] for k in keys}
    main_summary = PROJECT_ROOT / "outputs" / "cohort_summary.json"
    if main_summary.exists():
        assert json.loads(main_summary.read_text(encoding="utf-8"))["event_list_sha256"] == sums[0]["event_list_sha256"]
