"""Deterministic Test-block NSTDB slot allocation."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError


def test_allocation_table(events: pd.DataFrame, settings: dict,
                          development: pd.DataFrame | None = None) -> pd.DataFrame:
    spec = settings["test_noise_allocation"]
    if spec.get("rule") != "non_overlapping_grid_seeded_permutation_test_block":
        raise BlockedError("Test noise allocation rule is PENDING or unsupported")
    start, stop = map(int, spec["block"])
    length, step = int(spec["snippet_samples"]), int(spec["grid_step_samples"])
    if start != 325000 or stop != 650000 or length <= 0 or step < length:
        raise BlockedError("Invalid or overlapping Test noise grid")
    slots = np.arange(start, stop - length + 1, step, dtype=int)
    ordered = events.sort_values("event_id", kind="stable")
    if ordered.event_id.duplicated().any() or len(ordered) > len(slots):
        raise BlockedError("Cannot allocate unique Test noise snippets")
    prior = []
    if development is not None:
        if not {"snippet_start", "noise_type"}.issubset(development):
            raise BlockedError("Invalid Development allocation")
        prior = [(int(r.snippet_start), int(r.snippet_start) + length)
                 for r in development.itertuples(index=False)]
    rows = []
    for k, noise_type in enumerate(settings["noise_types"]):
        permutation = np.random.default_rng([int(spec["seed"]), k]).permutation(len(slots))
        for ev, slot in zip(ordered.itertuples(index=False), permutation):
            position = int(slots[slot])
            if position < start or position + length > stop or position < 325000:
                raise BlockedError("Test snippet outside Test block")
            if any(position < b and a < position + length for a, b in prior):
                raise BlockedError("Test snippet overlaps Development")
            rows.append({"event_id": ev.event_id, "noise_type": noise_type,
                         "snippet_start": position, "snippet_id": f"{noise_type}_{position}"})
    counts = Counter(r["snippet_id"] for r in rows)
    for row in rows:
        row["reuse_count"] = counts[row["snippet_id"]]
    if any(n != 1 for n in counts.values()):
        raise BlockedError("Test noise snippet was reused")
    return pd.DataFrame(rows)
