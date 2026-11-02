"""One-shot final evaluation command line entry point."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_info

from topovital.cohort import BlockedError
from topovital.util import (PROJECT_ROOT, atomic_write_csv, atomic_write_json,
                            atomic_write_parquet, load_yaml, new_run_id, profiled)
from topovital_analysis.level1 import analysis_code_hash
from topovital_analysis.runtime import ANALYSIS_CODE_VERSION, verify_blas_threads
from topovital_stage1.core import load_dev_events

from . import final_code_hash
from .allocation import test_allocation_table
from .apply import apply_models, load_models
from .evaluate import evaluate
from .events import load_test_events
from .features import failure_log, load_windows, run_features
from .gate import build_freeze_manifest, require_test_approval
from .verify_dev import verify_dev
from .waveforms import make_waveforms


def _provenance() -> dict:
    return {"threadpool_info": threadpool_info(),
            "analysis_code_version": ANALYSIS_CODE_VERSION,
            "final_code_hash": final_code_hash(),
            "analysis_code_hash": analysis_code_hash()}


def _membership(root: Path, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = load_dev_events(root / "outputs/development/event_audit_dev.parquet",
                          root / "outputs/split_assignment.csv")
    events = load_test_events(root / settings["test_events_source"],
                              root / settings["split"], settings, dev)
    return events, dev


def _write_freeze_inputs(root: Path, settings: dict) -> dict:
    events, _ = _membership(root, settings)
    output = root / "outputs/final_test"
    atomic_write_csv(output / "test_event_manifest.csv", events)
    development = pd.read_csv(root / "outputs/stage1_features/development/snippet_allocation.csv")
    allocation = test_allocation_table(events, settings, development)
    atomic_write_csv(output / "test_noise_allocation.csv", allocation)
    return {"test_events": len(events), "test_subjects": events.subject_id.nunique(),
            "test_snippets": len(allocation)}


def _write_test_membership(root: Path, settings: dict) -> dict:
    events, _ = _membership(root, settings)
    atomic_write_csv(root / "outputs/final_test/test_event_manifest.csv", events)
    return {"test_events": len(events), "test_subjects": events.subject_id.nunique()}


def _hash_directory(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file() and p.name not in
            ("preflight.json", "postrun.json", "resource_profile.csv")}


def _run_test(root: Path, settings: dict, manifest: dict, workers: int) -> dict:
    output = root / "outputs/final_test"
    if workers != int(settings["workers"]):
        raise BlockedError("Test workers must equal frozen settings.workers")
    output.joinpath("hashes").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "hashes/preflight.json", {"freeze_manifest": manifest,
                                                        "files": _hash_directory(output), **_provenance()})
    cfg_r1 = load_yaml(root / "configs/stage1_regime1.yaml")
    cfg_s1 = load_yaml(root / "configs/stage1.yaml")
    cfg_s0 = load_yaml(root / "configs/stage0.yaml")
    cfg_a = load_yaml(root / "configs/stage1_analysis.yaml")
    cfg_l3 = load_yaml(root / "configs/stage1_level3.yaml")
    cfg_f = load_yaml(root / "configs/stage1_factorial.yaml")
    events, _ = _membership(root, settings)
    frozen_events = pd.read_csv(output / "test_event_manifest.csv", dtype={"event_id": str})
    if not events.astype(str).equals(frozen_events[events.columns].astype(str)):
        raise BlockedError("Test event membership differs from freeze")
    atomic_write_csv(output / "test_event_manifest.csv", events)
    allocation = test_allocation_table(events, settings,
                                       pd.read_csv(root / "outputs/stage1_features/development/snippet_allocation.csv"))
    frozen_allocation = pd.read_csv(output / "test_noise_allocation.csv", dtype={"event_id": str})
    if not allocation.astype(str).equals(frozen_allocation[allocation.columns].astype(str)):
        raise BlockedError("Test noise allocation differs from freeze")
    atomic_write_csv(output / "test_noise_allocation.csv", allocation)
    waveforms, snr_audit, channel_audit = make_waveforms(events, allocation, settings,
                                                         cfg_r1, root, split="test")
    atomic_write_csv(output / "waveform_and_snr_audit.csv", snr_audit)
    atomic_write_csv(output / "noise_channel_audit.csv", channel_audit)
    classical, failures, ledger = run_features(waveforms, output, cfg_s1, cfg_s0,
                                               workers=workers, run_id="final_test")
    atomic_write_csv(output / "failure_log.csv", failure_log(waveforms, ledger))
    diagrams = load_windows(waveforms, output, cfg_a["inputs"]["representations"])
    if len(classical) != len(diagrams):
        # A failed injection is kept in predictions with an undefined classical feature row.
        keys = ["event_id", "subject_id", "condition", "noise_type", "snr_target"]
        classical = pd.DataFrame([{**{k: w[k] for k in keys}} for w in diagrams]).merge(
            classical, on=keys, how="left", validate="one_to_one")
    models, selection = load_models(root, settings, manifest)
    predictions = apply_models(diagrams, classical, models, selection, split="test")
    if len(predictions) != 5 * len(events) * (2 + 2 * len(settings["noise_types"]) * len(settings["snr_db"])):
        raise BlockedError("Test prediction row count mismatch")
    atomic_write_parquet(output / "test_predictions.parquet", predictions)
    metrics = evaluate(predictions, settings, cfg_l3, cfg_f, cfg_a, output)
    summary = {"events": len(events), "subjects": events.subject_id.nunique(),
               "windows": len(diagrams), "predictions": len(predictions),
               "ph_jobs": len(ledger), "ph_failures": int(ledger.validation_status.ne("ok").sum()),
               "feature_failures": len(failures), "metrics": metrics,
               "common_evaluable_windows": metrics["common_evaluable_windows"],
               "excluded_windows": metrics["excluded_windows"],
               "excluded_by_subject": metrics["excluded_by_subject"],
               "metric_specific_pair_exclusions": metrics["metric_specific_pair_exclusions"],
               **_provenance()}
    atomic_write_json(output / "run_summary.json", summary)
    atomic_write_json(output / "hashes/postrun.json", {"files": _hash_directory(output), **_provenance()})
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test-membership")
    sub.add_parser("freeze-manifest")
    verify = sub.add_parser("verify-dev")
    verify.add_argument("--output-dir", default="outputs/final_test/dev_verification")
    run = sub.add_parser("run-test")
    run.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    manifest = None
    try:
        # The one-shot gate is the first operation of the Test command.
        if args.command == "run-test":
            manifest = require_test_approval(root)
        verify_blas_threads()
        settings = load_yaml(root / "configs/final_test_evaluation.yaml")["settings"]
        if args.command == "test-membership":
            print(json.dumps(_write_test_membership(root, settings), indent=2))
            return 0
        workers = int(settings["workers"]) if args.command != "run-test" or args.workers is None else args.workers
        if args.command == "verify-dev":
            relative = Path(args.output_dir)
            if (relative.is_absolute() or ".." in relative.parts or relative == Path(".") or
                    not (root / relative).resolve().is_relative_to(root.resolve())):
                raise BlockedError("verify-dev output directory must be repo-relative")
            if (root / relative / "dev_verification_report.json").exists():
                raise BlockedError("Development verification report already exists")
        run_id = new_run_id(final_code_hash())
        profile_dir = (root / args.output_dir if args.command == "verify-dev"
                       else root / "outputs/final_test")
        with profiled(profile_dir, run_id, args.command, n_jobs=workers):
            if args.command == "freeze-manifest":
                counts = _write_freeze_inputs(root, settings)
                result = build_freeze_manifest(root)
                print(json.dumps({"membership": counts, "manifest": result}, indent=2))
            elif args.command == "verify-dev":
                result = verify_dev(root, settings, workers=workers, output_dir=args.output_dir)
                print(json.dumps(result, indent=2, default=str))
            else:
                result = _run_test(root, settings, manifest, workers)
                print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        if manifest is not None:
            atomic_write_json(root / "outputs/final_test/incident_report.json", {
                "error": f"{type(exc).__name__}: {exc}", **_provenance()})
        raise
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BlockedError as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        sys.exit(3)
