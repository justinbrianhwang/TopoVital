import numpy as np
import pandas as pd
import pytest

from topovital.matching import match_normals
from topovital.noise import NoiseBlockError, assert_segment, segment_in_block, validate_blocks
from topovital.template import branch, template_eligibility


def _ev(rows):
    return pd.DataFrame(rows, columns=["event_id", "record_id", "start_abs", "end_abs", "pre_median_rr_s"])


def _cand(rows):
    return pd.DataFrame(rows, columns=["seg_id", "record_id", "start_abs", "end_abs", "pre_median_rr_s"])


MATCHING_INTERVALS = {"pvc_lookback_s": 60, "normal_lookback_s": 0}


def test_matching_no_reuse_and_rr_tolerance():
    ev = _ev([("e1", "100", 100, 110, 1.0), ("e2", "100", 300, 310, 1.0)])
    ca = _cand([("n1", "100", 500, 510, 1.02), ("n2", "100", 700, 710, 1.2)])
    m = match_normals(ev, ca, 0.15, fs=1, **MATCHING_INTERVALS).set_index("event_id")
    assert m.loc["e1", "seg_id"] == "n1"
    assert m.loc["e2", "match_status"] == "no_admissible_normal_window"
    assert m.seg_id.dropna().is_unique


def test_matching_rr_bound_inclusive_and_exclusive():
    ev = _ev([("e1", "100", 100, 110, 1.0)])
    assert match_normals(ev, _cand([("n", "100", 300, 310, 1.15)]), 0.15, fs=1, **MATCHING_INTERVALS).seg_id[0] == "n"
    assert match_normals(ev, _cand([("n", "100", 300, 310, 1.1501)]), 0.15, fs=1, **MATCHING_INTERVALS).seg_id.isna()[0]


def test_matching_no_time_overlap():
    ev = _ev([("e1", "100", 100, 110, 1.0)])
    ca = _cand([("over_event", "100", 90, 100, 1.0),
                ("a", "100", 500, 510, 1.0), ("b_overlaps_a", "100", 505, 515, 1.0)])
    ev2 = _ev([("e1", "100", 100, 110, 1.0), ("e2", "100", 300, 310, 1.0)])
    m = match_normals(ev2, ca, 0.15, fs=1, **MATCHING_INTERVALS).set_index("event_id")
    assert m.loc["e1", "seg_id"] == "a"
    assert m.loc["e2", "match_status"] == "no_admissible_normal_window"
    assert match_normals(ev, ca[ca.seg_id == "over_event"], 0.15, fs=1, **MATCHING_INTERVALS).seg_id.isna()[0]


def test_matching_rejects_candidate_in_pvc_lookback():
    ev = _ev([("e", "100", 100, 110, 1.0)])
    ca = _cand([("in_lookback", "100", 50, 60, 1.0)])
    assert match_normals(ev, ca, 0.15, fs=1, **MATCHING_INTERVALS).seg_id.isna()[0]


def test_matching_allows_adjacent_nonoverlapping_normals():
    ev = _ev([("e1", "100", 100, 110, 1.0), ("e2", "100", 300, 310, 1.0)])
    ca = _cand([("n1", "100", 500, 510, 1.0), ("n2", "100", 511, 521, 1.0)])
    m = match_normals(ev, ca, 0.15, fs=1, **MATCHING_INTERVALS).set_index("event_id")
    assert m.loc["e1", "seg_id"] == "n1"
    assert m.loc["e2", "seg_id"] == "n2"


def test_matching_tie_break_deterministic():
    ev = _ev([("e1", "100", 200, 210, 1.0)])
    ca = _cand([("n_far", "100", 500, 510, 1.0), ("n_b", "100", 300, 310, 1.0),
                ("n_a", "100", 100, 110, 1.0)])
    assert match_normals(ev, ca, 0.15, fs=1, **MATCHING_INTERVALS).seg_id[0] == "n_a"
    assert match_normals(ev, ca.iloc[::-1], 0.15, fs=1, **MATCHING_INTERVALS).seg_id[0] == "n_a"


def test_matching_prefers_within_record_for_shared_subject():
    ev = _ev([("e1", "201", 200, 210, 1.0)]).assign(subject_id="S201")
    ca = _cand([("other", "202", 100, 110, 1.0), ("same", "201", 500, 510, 1.0)]).assign(subject_id="S201")
    assert match_normals(ev, ca, .15, fs=1, **MATCHING_INTERVALS).seg_id[0] == "same"


def _beats(fs, boundary, n_normals, rr_s=0.8):
    r = boundary - np.round(fs * (0.5 + rr_s * np.arange(n_normals))).astype(int)
    return r[::-1], np.array(["N"] * n_normals)


def test_template_eligibility_and_450ms_rule():
    fs, b = 360, 100_000
    r, s = _beats(fs, b, 12)
    res = template_eligibility(r, s, b, fs)
    assert res["template_status"] == "eligible"
    assert np.all(res["template_beat_samples"] + 162 <= b)
    # a beat whose +450 ms window crosses the boundary is not usable
    res2 = template_eligibility([b - 161], ["N"], b, fs, min_beats=1)
    assert res2["n_template_beats"] == 0
    assert template_eligibility([b - 162], ["N"], b, fs, min_beats=1)["n_template_beats"] == 1


def test_template_lookback_and_non_N():
    fs, b = 360, 100_000
    old = b - 61 * fs
    assert template_eligibility([old], ["N"], b, fs, min_beats=1)["n_template_beats"] == 0
    assert template_eligibility([b - 1000], ["V"], b, fs, min_beats=1)["n_template_beats"] == 0


def test_branching_template_failure_stays_in_common_only():
    fs, b = 360, 100_000
    r_ok, s_ok = _beats(fs, b, 12)
    r_bad, s_bad = _beats(fs, b, 5)
    common = pd.DataFrame([
        {"event_id": "ok", **{k: v for k, v in template_eligibility(r_ok, s_ok, b, fs).items() if k != "template_beat_samples"}},
        {"event_id": "few", **{k: v for k, v in template_eligibility(r_bad, s_bad, b, fs).items() if k != "template_beat_samples"}},
    ])
    r2 = branch(common)
    assert list(common.event_id) == ["ok", "few"]           # common cohort unchanged
    assert list(r2.event_id) == ["ok"]                       # only Regime 2 excludes
    assert common.set_index("event_id").loc["few", "template_status"] == "insufficient_normal_beats"


def test_noise_blocks_disjoint_and_no_crossing():
    n = 650_000
    dev, test = (0, 325_000), (325_000, 650_000)
    validate_blocks(dev, test, n)
    with pytest.raises(NoiseBlockError):
        validate_blocks((0, 325_001), test, n)
    with pytest.raises(NoiseBlockError):
        validate_blocks(None, test, n)
    blocks = {"development": dev, "test": test}
    assert segment_in_block(325_000 - 3600, 3600, dev)
    with pytest.raises(NoiseBlockError):
        assert_segment(325_000 - 3599, 3600, "development", blocks)  # crosses into test
    with pytest.raises(NoiseBlockError):
        assert_segment(1000, 3600, "test", blocks)                   # dev noise used for test
