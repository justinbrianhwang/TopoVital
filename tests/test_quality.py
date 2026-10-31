import math

import numpy as np
import pytest

from topovital.quality import bsqi_f1, is_saturated, match_one_to_one, max_run_at_bounds

LO, HI = 0, 2047  # synthetic bounds for the mechanism test only


def test_saturation_two_vs_three_consecutive():
    two = np.array([500, HI, HI, 500, LO, LO, 500])
    three = np.array([500, HI, HI, HI, 500])
    assert max_run_at_bounds(two, LO, HI) == 2 and not is_saturated(two, LO, HI, 3)
    assert max_run_at_bounds(three, LO, HI) == 3 and is_saturated(three, LO, HI, 3)


def test_saturation_mixed_low_high_run_and_pending_bounds():
    assert max_run_at_bounds(np.array([HI, LO, HI]), LO, HI) == 3
    with pytest.raises(ValueError):
        is_saturated(np.array([1]), None, None)


def test_bsqi_identical():
    a = [100, 500, 900]
    assert bsqi_f1(a, a, 54) == 1.0


def test_bsqi_disjoint_and_empty():
    assert bsqi_f1([100, 500], [10_000, 20_000], 54) == 0.0
    assert bsqi_f1([], [100], 54) == 0.0
    assert bsqi_f1([], [], 54) == 0.0


def test_bsqi_one_to_one_no_double_use():
    # two detections of b both within tolerance of one a: only one may match
    a, b = [1000], [990, 1010]
    assert len(match_one_to_one(a, b, 54)) == 1
    assert bsqi_f1(a, b, 54) == pytest.approx(2 * 1 / 3)
    pairs = match_one_to_one([1000, 1100], [1050], 54)
    assert len(pairs) == 1


def test_bsqi_tolerance_inclusive_and_order_invariant():
    tol = 54  # 150 ms at 360 Hz
    assert len(match_one_to_one([0], [54], tol)) == 1
    assert len(match_one_to_one([0], [55], tol)) == 0
    assert bsqi_f1([900, 100, 500], [505, 95, 905], tol) == bsqi_f1([100, 500, 900], [95, 505, 905], tol) == 1.0


def test_bsqi_maximum_matching_chain():
    # greedy earliest-first must achieve the maximum (2) here
    assert len(match_one_to_one([0, 100], [50, 150], 50)) == 2
