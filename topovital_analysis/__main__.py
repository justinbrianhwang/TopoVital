"""Command-line entry point for Development PH analysis."""
from __future__ import annotations

import argparse
import os
import sys

from topovital.cohort import BlockedError
from topovital.util import PROJECT_ROOT, config_hash, load_yaml, new_run_id, profiled
from topovital_stage1.core import load_dev_events
from topovital_stage1.features import representative_events

from .level1 import run_level1
from .level2 import run_level2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m topovital_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    level = sub.add_parser("level1")
    level.add_argument("--workers", type=int, default=1)
    level.add_argument("--events", type=int)
    level2 = sub.add_parser("level2")
    level2.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "level2":
            root = PROJECT_ROOT
            cfg = load_yaml(root / "configs" / "stage1_analysis.yaml")
            if not args.pilot and not cfg["level2"]["approved_for_full_run"]:
                raise BlockedError("Level 2 full run requires approved_for_full_run")
            events = load_dev_events(root / "outputs" / "development" / "event_audit_dev.parquet",
                                     root / "outputs" / "split_assignment.csv")
            out = root / "outputs" / "stage1_analysis" / "development" / ("level2_pilot" if args.pilot else "level2")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                os.environ[name] = "1"
            run_id = new_run_id(config_hash(cfg))
            with profiled(root / "outputs", run_id, "stage1 analysis level2", n_jobs=1) as record:
                summary = run_level2(events, cfg, root, out, pilot=args.pilot)
                record["note"] = f"fits={summary['fits']}; outer_folds={summary['outer_folds']}"
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
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = "1"
        run_id = new_run_id(config_hash(cfg))
        with profiled(root / "outputs", run_id, f"stage1 analysis level1 workers={args.workers}", n_jobs=args.workers) as record:
            summary = run_level1(events, cfg, root, out, args.workers, pilot=args.events is not None)
            record["note"] = f"windows={summary['windows']}; failures={summary['failures']}"
        print(f"Level 1: {summary['windows']} windows, {summary['failures']} failures; output={out.relative_to(root).as_posix()}")
        return 0
    except BlockedError as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
