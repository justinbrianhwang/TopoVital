"""Frozen Development-only full run."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import load_yaml, peak_rss_mb
from topovital_stage1.core import load_dev_events
from topovital_analysis.level1 import validate_events
from topovital_analysis.level2 import _load_windows
from topovital_analysis.level3 import KEYS
from topovital_analysis.runtime import verify_blas_threads, blas_thread_report

from .aggregate import aggregate_folds
from .audit import assert_unchanged, file_hashes, manifest, sha
from .core import BaseFitStore
from .pilot import compute_fold, threshold_target, _csv, _json

CONFIG = "configs/followup_error_monitor_v0_1.yaml"
PATHS = ("topovital_followup", CONFIG)


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _identity(root):
    code = file_hashes(root, ["topovital_followup/**/*.py"])
    return {"git_commit": _git(root, "rev-parse", "HEAD"), "code_files": code,
            "code_sha256": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
            "config_sha256": sha(root / CONFIG),
            "packages": {name: metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")}}


def _clean(root):
    if _git(root, "status", "--porcelain", "--", *PATHS):
        raise BlockedError("follow-up code or config git tree is dirty")


def freeze_full(root, cfg):
    root = Path(root)
    _clean(root)
    path = root / cfg["full_run"]["freeze"]["record"]
    if path.exists():
        raise BlockedError("full freeze record already exists")
    record = _identity(root) | {"python": sys.version, "blas_report": blas_thread_report()}
    _json(path, record)
    return record


def check_full(root, cfg):
    root = Path(root)
    if not cfg.get("approved_for_full_run"):
        raise BlockedError("full run requires approved_for_full_run")
    path = root / cfg["full_run"]["freeze"]["record"]
    if not path.is_file():
        raise BlockedError("full freeze record missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    current = _identity(root)
    if any(record.get(key) != value for key, value in current.items()):
        raise BlockedError("full freeze identity mismatch")
    _clean(root)
    blas = verify_blas_threads()
    if not blas:
        raise BlockedError("BLAS must use one thread")
    return current, blas


def _inputs(root, cfg, code_sha):
    inputs = manifest(root, cfg, code_sha)
    expected = cfg["scope"]["expected_counts"]
    if len(inputs["diagrams"]) != expected["npz_jobs"]:
        raise BlockedError("diagram count mismatch")
    events = load_dev_events(root / cfg["inputs"]["events"], root / "outputs/split_assignment.csv")
    validate_events(events, expected["events"])
    windows = _load_windows(events, load_yaml(root / "configs/stage1_analysis.yaml"), root)
    if len(windows) != expected["windows"] or len({w["subject_id"] for w in windows}) != expected["subjects"]:
        raise BlockedError("window or subject count mismatch")
    for condition in ("C0", "C1", "C2", "C3"):
        if sum(w["condition"] == condition for w in windows) != expected[condition]:
            raise BlockedError(f"{condition} count mismatch")
    summary = json.loads((root / cfg["inputs"]["classical_features_summary"]).read_text(encoding="utf-8"))
    feature_sha = sha(root / cfg["inputs"]["classical_features"])
    if not summary.get("stop_rule_passed") or feature_sha != summary["classical_features_sha256"]:
        raise BlockedError("classical feature stop rule or SHA mismatch")
    raw = pd.read_parquet(root / cfg["inputs"]["classical_features"])
    keys = pd.DataFrame([{key: w[key] for key in KEYS} for w in windows])
    if raw.duplicated(list(KEYS)).any() or keys.duplicated(list(KEYS)).any() or len(raw) != len(keys):
        raise BlockedError("classical keys not one-to-one")
    classical = keys.merge(raw, on=list(KEYS), validate="one_to_one", indicator=True)
    if not classical._merge.eq("both").all():
        raise BlockedError("classical key mismatch")
    return windows, classical.drop(columns="_merge"), feature_sha, inputs


def _pilot_compare(pred, pilot):
    keys = list(KEYS)
    joined = pred.merge(pilot, on=keys, validate="one_to_one", suffixes=("", "_pilot"))
    if len(joined) != len(pred) or len(pilot) != len(pred):
        raise BlockedError("S105 pilot row mismatch")
    columns = ["p", "e", "y_hat", *(f"M{m}_score" for m in range(5)),
               *(f"M{m}_operating_threshold" for m in range(5))]
    for column in columns:
        if not np.allclose(joined[column], joined[f"{column}_pilot"], atol=1e-12, rtol=0, equal_nan=True):
            raise BlockedError(f"S105 pilot mismatch: {column}")
    return {"name": "S105_equals_pilot", "passed": True, "n_checked": len(pred), "detail": "all scores, labels and thresholds <=1e-12"}


def run_full(root, cfg):
    root = Path(root)
    identity, blas = check_full(root, cfg)
    out = root / cfg["full_run"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    before = file_hashes(root, cfg["protected"])
    _json(out / "hashes/prerun.json", {"protected": before, "code": identity["code_files"], "config": identity["config_sha256"]})
    started = time.perf_counter()
    windows, classical, feature_sha, inputs = _inputs(root, cfg, identity["code_sha256"])
    _json(out / "input_manifest.json", inputs)
    subjects = sorted({w["subject_id"] for w in windows})
    factorial = load_yaml(root / "configs/stage1_factorial.yaml")
    target = threshold_target(factorial)
    reference = pd.read_parquet(root / cfg["inputs"]["level3_reference"])
    pilot = pd.read_parquet(root / cfg["output_dir"] / "pilot_predictions.parquet")
    store = BaseFitStore(windows, classical, cfg["base_model"]["c_grid"], target,
                         feature_sha, identity["config_sha256"], identity["code_sha256"])
    table_cache, folds, audits, times = {}, {}, [], {}
    for subject in subjects:
        fold_started = time.perf_counter()
        fold_cfg = json.loads(json.dumps(cfg))
        fold_cfg["base_model"]["regression_check"]["reference_fold"] = f"outer_{subject}"
        result = compute_fold(windows, classical, subject, fold_cfg, threshold_target=target,
                              hashes=(feature_sha, identity["config_sha256"], identity["code_sha256"]),
                              reference=reference, log=lambda _: None, store=store, table_cache=table_cache)
        fold_dir = out / "folds" / f"outer_{subject}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in result["csv"].items():
            _csv(fold_dir / f"{name}.csv", rows)
        result["error_label_manifest"].to_parquet(fold_dir / "error_label_manifest.parquet", index=False)
        result["predictions"].to_parquet(fold_dir / "predictions.parquet", index=False)
        audit = result["audit"][:]
        if subject == "S105":
            audit.append(_pilot_compare(result["predictions"], pilot))
        _json(fold_dir / "leakage_audit.json", audit)
        audits.extend({"subject_id": subject, **entry} for entry in audit)
        folds[subject] = {key: result[key] for key in ("outer_frame", "predictions", "csv", "summary")}
        times[subject] = time.perf_counter() - fold_started
        print(f"outer_{subject}: elapsed_s={times[subject]:.3f}", flush=True)
    if set(folds) != set(subjects):
        raise BlockedError("not every subject held out once")
    aggregate_folds(folds, cfg, out, pilot=pilot)
    audits.extend([{"name": "freeze_record_matched", "passed": True, "n_checked": 1},
                   {"name": "every_subject_held_out_once", "passed": True, "n_checked": len(folds)},
                   {"name": "no_subject_aggregated_twice", "passed": True, "n_checked": len(folds)},
                   {"name": "blas", "passed": bool(blas), "n_checked": 1, "detail": str(blas)}])
    _json(out / "leakage_audit.json", audits)
    after = file_hashes(root, cfg["protected"])
    _json(out / "hashes/postrun.json", {"protected": after, "code": file_hashes(root, ["topovital_followup/**/*.py"]),
                                         "config": sha(root / CONFIG)})
    assert_unchanged(before, after)
    report = {"fold_wall_seconds": times, "total_wall_s": time.perf_counter() - started,
              "folds_completed": len(folds), "fits": len(store.units), "basefits": len(store.basefits),
              "cache_hits": store.cache_hits, "error_table_cache_entries": len(table_cache),
              "monitor_fits": sum(r["summary"]["monitor_fits"] for r in folds.values()),
              "fallbacks": [f for r in folds.values() for f in r["summary"]["fallbacks"]],
              "warnings": sum(u.fit_record["convergence_warnings"] for u in store.units.values()) +
                          sum(r["summary"]["monitor_convergence_warnings"] for r in folds.values()),
              "peak_rss_mb": peak_rss_mb(), "blas": blas}
    _json(out / "run_summary.json", report)
    return report
