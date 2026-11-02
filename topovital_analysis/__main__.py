"""Command-line entry point for Development PH analysis."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys

from topovital.cohort import BlockedError
from topovital.util import PROJECT_ROOT, config_hash, load_yaml, new_run_id, profiled
from topovital_stage1.core import load_dev_events
from topovital_stage1.features import representative_events

from .level1 import run_level1
from .level2 import run_level2
from .factorial import run_factorial
from .classical_features import run_level3_features
from .level3 import run_level3
from .g06_ablation import run_g06_ablation
from .runtime import ANALYSIS_CODE_VERSION, verify_blas_threads


_PROFILE_NOTE_SUFFIX = f"; analysis_code_version={ANALYSIS_CODE_VERSION}; blas_threads_verified=1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m topovital_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    level = sub.add_parser("level1")
    level.add_argument("--workers", type=int, default=1)
    level.add_argument("--events", type=int)
    level2 = sub.add_parser("level2")
    level2.add_argument("--pilot", action="store_true")
    factorial = sub.add_parser("factorial")
    factorial.add_argument("--pilot", action="store_true")
    level3_features = sub.add_parser("level3-features")
    level3_features.add_argument("--workers", type=int, default=1)
    level3 = sub.add_parser("level3")
    level3.add_argument("--pilot", action="store_true")
    sub.add_parser("g06-ablation")
    args = parser.parse_args(argv)
    try:
        verify_blas_threads()
        if args.command == "g06-ablation":
            root = PROJECT_ROOT
            cfg_g = load_yaml(root / "configs" / "stage1_g06_ablation.yaml")
            if not cfg_g["approved_for_full_run"]:
                raise BlockedError("G-06 full run requires approved_for_full_run")
            cfg_l3 = load_yaml(root / "configs" / "stage1_level3.yaml")
            cfg_f = load_yaml(root / "configs" / "stage1_factorial.yaml")
            cfg_a = load_yaml(root / "configs" / "stage1_analysis.yaml")
            events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                     root / "outputs" / "split_assignment.csv")
            out = root / cfg_g["output"]
            run_id = new_run_id(config_hash({"g06": cfg_g, "level3": cfg_l3, "factorial": cfg_f, "analysis": cfg_a}))
            with profiled(root / "outputs", run_id, "stage1 analysis g06 ablation", n_jobs=1) as record:
                summary = run_g06_ablation(events, cfg_g, cfg_l3, cfg_f, cfg_a, root, out)
                record["note"] = f"fits={summary['fits']}; outer_folds={summary['outer_folds']}{_PROFILE_NOTE_SUFFIX}"
            print(f"G-06 ablation: {summary['fits']} fits; output={out.relative_to(root).as_posix()}")
            return 0
        if args.command in ("level3-features", "level3"):
            root = PROJECT_ROOT
            cfg_l3 = load_yaml(root / "configs" / "stage1_level3.yaml")
            cfg_a = load_yaml(root / "configs" / "stage1_analysis.yaml")
            if args.command == "level3" and not args.pilot and not cfg_l3["approved_for_full_run"]:
                raise BlockedError("Level 3 full run requires approved_for_full_run")
            if args.command == "level3-features" and args.workers < 1:
                raise BlockedError("--workers must be positive")
            events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                     root / "outputs" / "split_assignment.csv")
            if args.command == "level3-features":
                cfg_r1 = load_yaml(root / "configs" / "stage1_regime1.yaml")
                cfg_s0 = load_yaml(root / "configs" / "stage0.yaml")
                out = root / "outputs" / "stage1_analysis" / "development" / "level3_features"
                run_id = new_run_id(config_hash({"level3": cfg_l3, "analysis": cfg_a, "regime1": cfg_r1, "stage0": cfg_s0}))
                with profiled(root / "outputs", run_id, "stage1 analysis level3 features", n_jobs=args.workers) as record:
                    summary = run_level3_features(events, cfg_l3, cfg_a, cfg_r1, cfg_s0, root, out, workers=args.workers)
                    record["note"] = f"windows={summary['windows']}; stop_rule_passed={summary['stop_rule_passed']}{_PROFILE_NOTE_SUFFIX}"
                print(f"Level 3 features: {summary['windows']} windows; output={out.relative_to(root).as_posix()}")
            else:
                cfg_f = load_yaml(root / "configs" / "stage1_factorial.yaml")
                out = root / "outputs" / "stage1_analysis" / "development" / ("level3_pilot" if args.pilot else "level3")
                run_id = new_run_id(config_hash({"level3": cfg_l3, "factorial": cfg_f, "analysis": cfg_a}))
                with profiled(root / "outputs", run_id, "stage1 analysis level3", n_jobs=1) as record:
                    summary = run_level3(events, cfg_l3, cfg_f, cfg_a, root, out, pilot=args.pilot)
                    record["note"] = f"fits={summary['fits']}; outer_folds={summary['outer_folds']}{_PROFILE_NOTE_SUFFIX}"
                print(f"Level 3: {summary['fits']} fits; output={out.relative_to(root).as_posix()}")
            return 0
        if args.command == "factorial":
            root = PROJECT_ROOT
            cfg_f = load_yaml(root / "configs" / "stage1_factorial.yaml")
            cfg_a = load_yaml(root / "configs" / "stage1_analysis.yaml")
            if not args.pilot and not cfg_f["approved_for_full_run"]:
                raise BlockedError("Factorial full run requires approved_for_full_run")
            events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                     root / "outputs" / "split_assignment.csv")
            out = root / "outputs" / "stage1_analysis" / "development" / ("factorial_pilot" if args.pilot else "factorial")
            run_id = new_run_id(config_hash({"factorial": cfg_f, "analysis": cfg_a}))
            with profiled(root / "outputs", run_id, "stage1 analysis factorial", n_jobs=1) as record:
                summary = run_factorial(events, cfg_f, cfg_a, root, out, pilot=args.pilot)
                record["note"] = f"fits={summary['fits']}; outer_folds={summary['outer_folds']}{_PROFILE_NOTE_SUFFIX}"
            print(f"Factorial: {summary['fits']} fits; output={out.relative_to(root).as_posix()}")
            return 0
        if args.command == "level2":
            root = PROJECT_ROOT
            cfg = load_yaml(root / "configs" / "stage1_analysis.yaml")
            if not args.pilot and not cfg["level2"]["approved_for_full_run"]:
                raise BlockedError("Level 2 full run requires approved_for_full_run")
            events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                     root / "outputs" / "split_assignment.csv")
            out = root / "outputs" / "stage1_analysis" / "development" / ("level2_pilot" if args.pilot else "level2")
            run_id = new_run_id(config_hash(cfg))
            with profiled(root / "outputs", run_id, "stage1 analysis level2", n_jobs=1) as record:
                summary = run_level2(events, cfg, root, out, pilot=args.pilot)
                record["note"] = f"fits={summary['fits']}; outer_folds={summary['outer_folds']}{_PROFILE_NOTE_SUFFIX}"
            print(f"Level 2: {summary['fits']} fits; output={out.relative_to(root).as_posix()}")
            return 0
        if args.workers < 1:
            raise BlockedError("--workers must be positive")
        root = PROJECT_ROOT
        cfg = load_yaml(root / "configs" / "stage1_analysis.yaml")
        if cfg["level1"]["status"] != "approved_for_full_development_run":
            raise BlockedError("Level 1 Development analysis is not approved")
        events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                 root / "outputs" / "split_assignment.csv")
        if args.events is not None:
            events = representative_events(events, args.events)
        out = root / "outputs" / "stage1_analysis" / "development" / ("level1_pilot" if args.events is not None else "level1")
        run_id = new_run_id(config_hash(cfg))
        with profiled(root / "outputs", run_id, f"stage1 analysis level1 workers={args.workers}", n_jobs=args.workers) as record:
            summary = run_level1(events, cfg, root, out, args.workers, pilot=args.events is not None)
            record["note"] = f"windows={summary['windows']}; failures={summary['failures']}{_PROFILE_NOTE_SUFFIX}"
        print(f"Level 1: {summary['windows']} windows, {summary['failures']} failures; output={out.relative_to(root).as_posix()}")
        return 0
    except BlockedError as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
