"""Normal-template eligibility (Stage 0 flag) and common / Regime 2 branching (brief §8).

This module only decides *eligibility*. The projection itself (RR median, anchors,
linear-interpolated gaps, cosine cross-fade) is a later step and is not validated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from topovital.timeaxis import ms_to_samples


def template_eligibility(beat_samples, beat_symbols, boundary_abs: int, fs: int, *,
                         lookback_s: float = 60, min_beats: int = 10,
                         win_ms=(-250, 450), normal_symbols=("N",)) -> dict:
    """A normal beat r_j qualifies if it lies in the look-back [boundary-60s, boundary),
    its whole extraction window ends before the boundary (r_j + samples(450ms) <=
    boundary_abs) and starts inside the record (r_j - samples(250ms) >= 0).
    """
    r = np.asarray(beat_samples, dtype=np.int64)
    s = np.asarray(beat_symbols)
    pre, post = ms_to_samples(-win_ms[0], fs), ms_to_samples(win_ms[1], fs)
    lb_start = boundary_abs - ms_to_samples(lookback_s * 1000, fs)
    ok = (np.isin(s, list(normal_symbols)) & (r >= lb_start) & (r < boundary_abs)
          & (r + post <= boundary_abs) & (r - pre >= 0))
    n = int(ok.sum())
    return {"n_template_beats": n,
            "template_status": "eligible" if n >= min_beats else "insufficient_normal_beats",
            "template_beat_samples": r[ok]}


def branch(common: pd.DataFrame) -> pd.DataFrame:
    """Regime 2 subset of the common cohort; never removes events from `common`."""
    return common[common["template_status"] == "eligible"].copy()
