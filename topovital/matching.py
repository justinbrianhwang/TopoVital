"""Chronological, subject-grouped normal-window matching for Stage 0."""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd


def _overlaps(s1, e1, s2, e2) -> bool:
    return s1 < e2 and s2 < e1


def match_normals(events: pd.DataFrame, cands: pd.DataFrame, rr_tol: float,
                  allowed: Callable | None = None, *, fs: int = 360,
                  pvc_lookback_s: float = 60,
                  normal_lookback_s: float = 0) -> pd.DataFrame:
    """Match using pre-boundary median RR, asymmetric occupied intervals, and record preference.

    Both tables carry subject_id, record_id, start_abs, end_abs and
    pre_median_rr_s. Candidate ID is seg_id. Missing RR makes a pair inadmissible.
    """
    e = events.copy()
    c = cands.copy()
    for table in (e, c):
        if "subject_id" not in table:
            table["subject_id"] = table.record_id
    if "pre_median_rr_s" not in e or "pre_median_rr_s" not in c:
        raise ValueError("pre_median_rr_s is required for normal matching")
    pvc_lookback = round(pvc_lookback_s * fs)
    normal_lookback = round(normal_lookback_s * fs)
    occupied = {}
    for ev in e.itertuples():
        occupied.setdefault(ev.record_id, []).append((ev.start_abs - pvc_lookback, ev.end_abs))
    records_per_subject = pd.concat([e[["subject_id", "record_id"]], c[["subject_id", "record_id"]]]).groupby("subject_id").record_id.nunique().to_dict()
    used = set()
    result = []
    for ev in e.sort_values(["subject_id", "record_id", "start_abs", "event_id"], kind="stable").itertuples():
        admissible = []
        for n in c.itertuples():
            if n.seg_id in used or n.subject_id != ev.subject_id:
                continue
            if not (math.isfinite(float(ev.pre_median_rr_s)) and math.isfinite(float(n.pre_median_rr_s))):
                continue
            if ev.pre_median_rr_s <= 0 or abs(n.pre_median_rr_s - ev.pre_median_rr_s) > rr_tol * ev.pre_median_rr_s + 1e-12:
                continue
            if n.record_id != ev.record_id and records_per_subject.get(ev.subject_id, 0) < 2:
                continue
            if any(_overlaps(n.start_abs - normal_lookback, n.end_abs, s, stop)
                   for s, stop in occupied.get(n.record_id, ())):
                continue
            if allowed is not None and not allowed(ev, n):
                continue
            key = ((0, abs(n.start_abs - ev.start_abs), n.start_abs, str(n.seg_id))
                   if n.record_id == ev.record_id else (1, n.start_abs, str(n.record_id), str(n.seg_id)))
            admissible.append((key, n))
        if admissible:
            n = min(admissible, key=lambda x: x[0])[1]
            used.add(n.seg_id)
            occupied.setdefault(n.record_id, []).append((n.start_abs - normal_lookback, n.end_abs))
            result.append({"event_id": ev.event_id, "seg_id": n.seg_id,
                           "normal_start_sample": n.start_abs, "normal_record_id": n.record_id,
                           "normal_pre_median_rr_s": n.pre_median_rr_s, "match_status": "matched"})
        else:
            result.append({"event_id": ev.event_id, "seg_id": None,
                           "normal_start_sample": None, "normal_record_id": None,
                           "normal_pre_median_rr_s": None,
                           "match_status": "no_admissible_normal_window"})
    return pd.DataFrame(result)
