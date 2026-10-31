import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError, split_subjects
from topovital.stage0 import run_split
from topovital.stage0_pipeline import (_quality_gate, intersects_loss, post_is_single_v,
                                       retained_after_overlap, run_pipeline, signal_loss_intervals)
from topovital.util import atomic_write_csv, atomic_write_parquet


def test_stratified_quotas_deterministic(cfg):
    counts = {f"S{i:03d}": 0 if i < 11 else 1 if i < 27 else 10 for i in range(43)}
    a = split_subjects(list(counts), cfg["split"], counts)
    b = split_subjects(list(reversed(counts)), cfg["split"], counts)
    assert a == b
    assert [sum(a[s] == "development" for s in counts if (counts[s] == n)) for n in (0, 1, 10)] == [6, 8, 8]
    assert sum(v == "development" for v in a.values()) == 22
    with pytest.raises(BlockedError):
        split_subjects(list(counts), {**cfg["split"], "stratification": "unknown"}, counts)


def test_post_and_signal_loss_intervals():
    samples = np.array([10, 20, 30, 40, 50])
    assert post_is_single_v(samples, np.array(["N", "V", "N", "~", "N"]), 15, 40, 20)
    assert not post_is_single_v(samples, np.array(["N", "V", "F", "~", "N"]), 15, 40, 20)
    assert not post_is_single_v(samples, np.array(["N", "V", "V", "~", "N"]), 15, 40, 20)
    intervals = signal_loss_intervals([10, 20, 30, 50], ["N", "~", "N", "~"], [0, -1, 0, 0], 100)
    assert intervals == [(20, 50)]
    assert intersects_loss(intervals, 49, 51)
    assert not intersects_loss(intervals, 50, 60)


def test_overlap_keeps_earlier_with_ten_second_separation():
    events = pd.DataFrame([
        dict(event_id="a", record_id="100", start_abs=100, end_abs=110),
        dict(event_id="b", record_id="100", start_abs=119, end_abs=129),
        dict(event_id="c", record_id="100", start_abs=120, end_abs=130),
    ])
    assert retained_after_overlap(events, fs=1, min_separation_s=10) == {"a", "c"}


def test_drop_reason_uses_first_gate(cfg, monkeypatch):
    cfg["cohort"]["fs_expected"] = 10
    d = np.full(100, 1000, dtype=int)
    p = np.sin(np.arange(100))
    samples = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85])
    symbols = np.array(["N", "F", "N", "N", "N", "V", "N", "N", "N"])
    d[10:13] = 2047
    reason, _ = _quality_gate(d, p, 0, 50, 100, samples, symbols, [], cfg, 55)
    assert reason == "pre_beats_not_all_N"
    symbols[1] = "N"
    reason, _ = _quality_gate(d, p, 0, 50, 100, samples, symbols, [], cfg, 55)
    assert reason == "pre_saturation"
    d[10:13] = 1000
    monkeypatch.setattr("topovital.stage0_pipeline.bsqi", lambda *args: (1.0, ""))
    reason, _ = _quality_gate(d, p, 0, 50, 100, samples, symbols, [(70, 90)], cfg, 55)
    assert reason == "post_signal_loss"


def test_split_assignment_immutable(tmp_path, cfg):
    cfg["split"]["n_development_subjects"] = 1
    cfg["split"]["n_test_subjects"] = 2
    man = pd.DataFrame([
        dict(record_id="100", subject_id="S100", in_stage0_cohort=True, n_pvc_candidates_step1=0),
        dict(record_id="101", subject_id="S101", in_stage0_cohort=True, n_pvc_candidates_step1=1),
        dict(record_id="103", subject_id="S103", in_stage0_cohort=True, n_pvc_candidates_step1=2),
    ])
    atomic_write_csv(tmp_path / "cohort_manifest.csv", man)
    first = run_split(cfg, tmp_path, "test")
    assert len(first) == 3 and first.split.value_counts().to_dict() == {"test": 2, "development": 1}
    assert "split_group" in pd.read_csv(tmp_path / "cohort_manifest.csv")
    cfg["split"]["seed"] += 1
    with pytest.raises(BlockedError):
        run_split(cfg, tmp_path, "test2")


def test_development_reader_never_sees_test(tmp_path, cfg, monkeypatch):
    cfg["cohort"]["fs_expected"] = 10
    cfg["split"]["n_development_subjects"] = 1
    cfg["split"]["n_test_subjects"] = 1
    cfg["time"]["g_candidates_ms"] = [200]
    cfg["g_selection"]["fallback_g_ms"] = 200
    manifest = pd.DataFrame([
        dict(record_id="100", subject_id="S100", split="development", in_stage0_cohort=True, mlii_channel_index=0),
        dict(record_id="101", subject_id="S101", split="test", in_stage0_cohort=True, mlii_channel_index=0),
    ])
    atomic_write_csv(tmp_path / "cohort_manifest.csv", manifest)
    atomic_write_csv(tmp_path / "split_assignment.csv", pd.DataFrame({"subject_id": ["S100", "S101"]}))
    events = pd.DataFrame([
        dict(event_id="100_500", subject_id="S100", record_id="100", r_abs=500, ann_index=50,
             beat_symbol="V", fs=10, mlii_channel_index=0),
        dict(event_id="101_500", subject_id="S101", record_id="101", r_abs=500, ann_index=50,
             beat_symbol="V", fs=10, mlii_channel_index=0),
    ])
    atomic_write_parquet(tmp_path / "event_audit.parquet", events)
    beats = np.arange(10, 1990, 10)
    symbols = np.array(["V" if x == 500 else "N" for x in beats])
    digital = np.full(2000, 1000, dtype=int)
    physical = np.sin(np.arange(2000) / 5)
    seen = []

    def reader(path, idx):
        seen.append(path.name)
        assert path.name == "100"
        return digital, physical, beats, symbols, np.zeros(len(beats), dtype=int)

    monkeypatch.setattr("topovital.stage0_pipeline._read_record", reader)
    monkeypatch.setattr("topovital.stage0_pipeline.bsqi", lambda *args: (1.0, ""))
    monkeypatch.setattr("topovital.stage0_pipeline._delineate", lambda *args: (498, 200.0, "valid"))
    monkeypatch.setattr("topovital.stage0_pipeline._figures", lambda *args: None)
    run_pipeline(cfg, tmp_path, "synthetic", "development", [200])
    audit = pd.read_parquet(tmp_path / "development" / "event_audit_dev.parquet")
    assert len(audit) == 1 and audit.record_id.iloc[0] == "100"
    assert audit.in_common_cohort.iloc[0]
    assert set(seen) == {"100"}
    assert (tmp_path / "g_decision.draft.yaml").exists()
