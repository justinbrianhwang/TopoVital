"""Input hashes and provenance checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from importlib import metadata

from topovital.cohort import BlockedError


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hashes(root, patterns):
    return {p.relative_to(root).as_posix(): sha(p) for pattern in patterns
            for p in sorted(root.glob(pattern)) if p.is_file()}


def assert_unchanged(before, after):
    if before != after:
        raise BlockedError("protected input hashes changed")


def _subjects(value):
    return set(value.split(",")) - {""} if isinstance(value, str) else set(value)


def inner_provenance_check(j, U, table_U, eval_frame_j):
    U = set(U)
    rows = list(table_U.itertuples())
    passed = all(row.subject_id in U and j != row.subject_id and
                 j not in _subjects(row.training_subjects) and
                 _subjects(row.training_subjects) <= U for row in rows)
    passed &= set(table_U.subject_id) <= U
    passed &= all(_subjects(value) <= U and j not in _subjects(value)
                  for value in eval_frame_j.training_subjects)
    passed &= set(eval_frame_j.subject_id) == {j}
    if not passed:
        raise BlockedError(f"inner provenance failed for {j}")
    return {"name": f"inner_provenance:{j}", "passed": bool(passed),
            "n_checked": len(rows) + len(eval_frame_j), "detail": f"training subjects={len(U)}"}


def outer_check(outer, units, basefits, tables, eval_frames, pi_subject_sets, monitor_subject_sets):
    collections = [*[u.subjects for u in units], *[b.subjects for b in basefits],
                   *[set(t.subject_id) | set().union(*(_subjects(v) for v in t.training_subjects)) for t in tables],
                   *[set().union(*(_subjects(v) for v in f.training_subjects)) for f in eval_frames],
                   *pi_subject_sets, *monitor_subject_sets]
    passed = all(outer not in subjects for subjects in collections)
    if not passed:
        raise BlockedError(f"outer subject {outer} leaked into training provenance")
    return {"name": "outer_provenance", "passed": bool(passed), "n_checked": len(collections),
            "detail": f"outer={outer}"}


def provenance_check(outer, basefits, tables, inner=None):
    rows = [row for table in tables for row in table.itertuples()]
    passed = all(row.subject_id not in _subjects(row.training_subjects) and
                 outer not in _subjects(row.training_subjects) for row in rows)
    if not passed:
        raise BlockedError("error row provenance failed")
    return [{"name": "error_row_provenance", "passed": bool(passed),
             "n_checked": len(rows), "detail": "row subject absent from generating fit"}]


def manifest(root, cfg, code_hash):
    inputs = {}
    for value in cfg["inputs"].values():
        for item in (value if isinstance(value, list) else [value]):
            path = root / item
            if path.is_file():
                inputs[item] = sha(path)
    diagrams = [p for item in cfg["inputs"]["diagrams"] for p in (root / item).rglob("*.npz")]
    digests = {p.relative_to(root).as_posix(): sha(p) for p in sorted(diagrams)}
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {"inputs": inputs, "diagrams": digests,
            "diagrams_combined_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": sys.version, "code_sha256": code_hash,
            "packages": {name: metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")}}
