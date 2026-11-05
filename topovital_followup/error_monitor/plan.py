"""Pure dependency planning; this module never opens study data."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Plan:
    basefits: frozenset[frozenset[str]]
    units: frozenset[tuple[frozenset[str], float]]
    monitor_fits: int

    def counts(self) -> dict:
        by_size: dict[int, int] = {}
        for subjects in self.basefits:
            by_size[len(subjects)] = by_size.get(len(subjects), 0) + 1
        return {"basefits_by_size": dict(sorted(by_size.items())),
                "unique_basefits": len(self.basefits), "unique_units": len(self.units),
                "monitor_fits": self.monitor_fits}


def needed_basefits(outer_subjects, all_subjects, c_grid) -> Plan:
    """Return all base fits and fixed-C units needed for requested outer folds."""
    all_subjects = frozenset(all_subjects)
    outer_subjects = tuple(outer_subjects)
    if len(all_subjects) != len(set(all_subjects)) or not set(outer_subjects) <= all_subjects:
        raise ValueError("invalid_subjects")
    basefits: set[frozenset[str]] = set()
    units: set[tuple[frozenset[str], float]] = set()
    for outer in outer_subjects:
        training = all_subjects - {outer}
        basefits.add(training)
        for held in training:
            basefits.add(training - {held})
        for first, second in combinations(sorted(training), 2):
            basefits.add(training - {first, second})
    for subjects in basefits:
        for c in c_grid:
            units.add((subjects, float(c)))
        for held in subjects:
            for c in c_grid:
                units.add((subjects - {held}, float(c)))
    return Plan(frozenset(basefits), frozenset(units),
                len(outer_subjects) * 4 * ((len(all_subjects) - 1) * 3 + 1))
