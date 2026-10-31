"""Quality primitives: ADC saturation run detection and bSQI one-to-one matching."""
from __future__ import annotations

import numpy as np


def max_run_at_bounds(d: np.ndarray, low: int, high: int) -> int:
    """Longest run of consecutive samples with d <= low or d >= high."""
    d = np.asarray(d)
    hit = (d <= low) | (d >= high)
    if not hit.any():
        return 0
    padded = np.concatenate(([0], hit.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max())


def is_saturated(d: np.ndarray, low: int | None, high: int | None, consecutive: int = 3) -> bool:
    if low is None or high is None:
        raise ValueError("saturation digital bounds are pending (protocol value required)")
    return max_run_at_bounds(d, low, high) >= consecutive


def match_one_to_one(a, b, tol: int) -> list[tuple[int, int]]:
    """Maximum one-to-one matching of two detection lists within |a_i - b_j| <= tol.

    Both lists are sorted (stable) first; a two-pointer sweep matches each point at
    most once. Ties are resolved deterministically: when a_i can match b_j the pair is
    taken immediately (earliest-first), which yields a maximum-cardinality matching for
    1-D tolerance windows. Returns pairs of indices into the *sorted* arrays.
    """
    a = np.sort(np.asarray(a, dtype=np.int64), kind="stable")
    b = np.sort(np.asarray(b, dtype=np.int64), kind="stable")
    i = j = 0
    pairs = []
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            pairs.append((i, j))
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return pairs


def bsqi_f1(a, b, tol: int) -> float:
    """F1 = 2M / (|a| + |b|); zero detections give zero quality."""
    na, nb = len(a), len(b)
    if na + nb == 0:
        return 0.0
    return 2.0 * len(match_one_to_one(a, b, tol)) / (na + nb)
