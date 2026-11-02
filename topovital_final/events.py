"""Test event membership validation, without outcome columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from topovital.cohort import BlockedError
from topovital_stage1.core import FS, HALF

MEMBERSHIP_COLUMNS = ("event_id", "subject_id", "record_id", "normal_record_id",
                      "start_abs", "boundary_abs", "end_abs", "normal_start_sample",
                      "mlii_channel_index")


def load_test_events(audit_path: Path, split_path: Path, settings: dict,
                     development_events: pd.DataFrame | None = None) -> pd.DataFrame:
    required = set(MEMBERSHIP_COLUMNS) | {"fs", "g_ms", "in_common_cohort", "split"}
    audit = pd.read_parquet(audit_path, columns=sorted(required))
    if required - set(audit.columns):
        raise BlockedError(f"Test audit missing membership columns: {sorted(required - set(audit.columns))}")
    events = audit.loc[audit.g_ms.eq(settings["g_star_ms"]) &
                       audit.in_common_cohort.eq(True)].copy()
    expected = int(settings["expected_test_events"])
    n_subjects = int(settings["expected_test_subjects"])
    if (len(events) != expected or events.event_id.nunique() != expected or
            events.subject_id.nunique() != n_subjects):
        raise BlockedError("Test event or subject count differs from frozen expectation")
    if not events.split.eq("test").all() or events[list(MEMBERSHIP_COLUMNS) + ["fs"]].isna().any().any():
        raise BlockedError("Invalid Test membership or missing geometry")
    splits = pd.read_csv(split_path, dtype={"subject_id": str})
    if not {"subject_id", "split"}.issubset(splits) or splits.subject_id.duplicated().any():
        raise BlockedError("Invalid split assignment")
    assigned = splits.set_index("subject_id")["split"].to_dict()
    if any(assigned.get(str(s)) != "test" for s in events.subject_id):
        raise BlockedError("Test event subject is not assigned to Test")
    if set(events.subject_id) != set(settings["test_subjects"]):
        raise BlockedError("Test subject list mismatch")
    dev_subjects = set(splits.loc[splits.split.eq("development"), "subject_id"])
    if set(events.subject_id) & dev_subjects:
        raise BlockedError("Development/Test subject overlap")
    if development_events is not None:
        if set(events.subject_id) & set(development_events.subject_id):
            raise BlockedError("Development/Test subject overlap")
        if (set(events.record_id.astype(str)) | set(events.normal_record_id.astype(str))) & (
                set(development_events.record_id.astype(str)) |
                set(development_events.normal_record_id.astype(str))):
            raise BlockedError("Development/Test record overlap")
    for ev in events.itertuples(index=False):
        if (int(ev.fs) != FS or int(ev.start_abs) < 0 or
                int(ev.boundary_abs) - int(ev.start_abs) != HALF or
                int(ev.end_abs) - int(ev.boundary_abs) != HALF or
                int(ev.normal_start_sample) < 0 or int(ev.mlii_channel_index) < 0):
            raise BlockedError(f"Invalid Test window geometry: {ev.event_id}")
        rid, normal = str(ev.record_id), str(ev.normal_record_id)
        subject = "S201" if rid == "202" else f"S{rid}"
        if str(ev.subject_id) != subject or (normal != rid and
                not ({rid, normal} == {"201", "202"} and subject == "S201")):
            raise BlockedError(f"Test record/subject mismatch: {ev.event_id}")
    return events.loc[:, MEMBERSHIP_COLUMNS].sort_values("event_id", kind="stable").reset_index(drop=True)
