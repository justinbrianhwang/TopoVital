"""Explain the Regime 1 ledger's two configuration hashes."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topovital.util import atomic_write_json, config_hash
from tools.make_public_copies import repo_relative_path

DATA_DIR = ROOT / "outputs" / "stage1_features" / "development"


def flatten_config(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten mapping keys; retain leaf values, including lists, unchanged."""
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        result.update(flatten_config(child, name))
    return result


def differing_keys(left: dict, right: dict) -> list[str]:
    first, second = flatten_config(left), flatten_config(right)
    return sorted(key for key in first.keys() | second.keys()
                  if key not in first or key not in second or first[key] != second[key])


def cached_job_key(root: Path, ledger_path: str) -> str:
    relative = Path(repo_relative_path(ledger_path))
    path = (root / relative).resolve()
    allowed = (root / "outputs" / "stage1_features" / "development" / "regime1_jobs").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"Cached job is outside regime1_jobs: {relative}")
    with np.load(path, allow_pickle=False) as saved:
        return str(json.loads(str(saved["metadata"].item()))["cache_key"])


def compare(root: Path = ROOT) -> dict:
    data_dir = root / "outputs" / "stage1_features" / "development"
    ledger = pd.read_csv(data_dir / "regime1_ledger.csv", dtype=str, keep_default_na=False)
    required = {"config_hash", "cached", "cache_key", "job_path"}
    if not required.issubset(ledger.columns):
        raise ValueError(f"Ledger lacks columns: {sorted(required - set(ledger.columns))}")
    with open(root / "configs" / "stage1_regime1.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(root / "configs" / "stage1.yaml", encoding="utf-8") as handle:
        stage1_cfg = yaml.safe_load(handle)
    current = {"regime1": cfg, "stage1": stage1_cfg}
    validation_cfg = copy.deepcopy(cfg)
    validation_cfg["approved_for_full_run"] = False
    validation = {"regime1": validation_cfg, "stage1": stage1_cfg}
    current_hash = config_hash(current)
    validation_hash = config_hash(validation)
    groups = []
    reproduced = {}
    for digest, group in ledger.groupby("config_hash", sort=True, dropna=False):
        cached = group["cached"].str.casefold().eq("true")
        matches = [label for label, candidate in (("current", current_hash),
                                                   ("validation", validation_hash))
                   if digest == candidate]
        reproduced[digest] = bool(matches)
        groups.append({"config_hash": digest, "rows": len(group),
                       "cached_true": int(cached.sum()),
                       "cached_false": int((~cached).sum()),
                       "reconstruction_matches": matches})
    cached_rows = ledger[ledger["cached"].str.casefold().eq("true")]
    matches = 0
    mismatches = []
    for row in cached_rows.itertuples(index=False):
        relative = ""
        try:
            relative = repo_relative_path(row.job_path)
            actual = cached_job_key(root, row.job_path)
            if actual == row.cache_key:
                matches += 1
            else:
                mismatches.append(relative)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            mismatches.append(f"{relative or 'invalid_job_path'}:{type(exc).__name__}")
    differences = differing_keys(current, validation)
    valid = (differences == ["regime1.approved_for_full_run"]
             and all(reproduced.values()) and matches == len(cached_rows)
             and len(cached_rows) == 108)
    result = {
        "ledger_hash_groups": groups,
        "current_hash": current_hash,
        "reconstructed_validation_hash": validation_hash,
        "reproduced": reproduced,
        "differing_keys": differences,
        "n_cached": len(cached_rows),
        "cache_keys_match": matches,
        "cache_key_mismatches": mismatches,
        "conclusion": (
            "The 108 cached jobs retain their cache keys; the two run config hashes differ "
            "only because approved_for_full_run changed from false to true."
            if valid else "The ledger does not establish the expected config-only difference."
        ),
    }
    atomic_write_json(data_dir / "hashes" / "regime1_cache_config_diff.json", result)
    if not valid:
        raise ValueError("Regime 1 cache/config comparison failed; see the JSON report")
    return result


if __name__ == "__main__":
    try:
        result = compare()
        print(f"Compared {result['n_cached']} cached jobs; report in {DATA_DIR / 'hashes'}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Cache/config comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
