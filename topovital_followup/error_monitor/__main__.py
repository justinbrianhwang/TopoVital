"""Command line for the Development-only follow-up."""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json

from topovital.cohort import BlockedError
from topovital.util import PROJECT_ROOT, load_yaml
from topovital_analysis.runtime import blas_thread_report, verify_blas_threads
from topovital_stage1.core import load_dev_events

from .plan import needed_basefits


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m topovital_followup.error_monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("pilot")
    commands.add_parser("full")
    commands.add_parser("freeze-full")
    commands.add_parser("blas-check")
    args = parser.parse_args(argv)
    if args.command == "blas-check":
        report = verify_blas_threads()
        print(json.dumps(report if report is not None else blas_thread_report(), sort_keys=True))
        return 0
    root = PROJECT_ROOT
    cfg = load_yaml(root / "configs/followup_error_monitor_v0_1.yaml")
    if args.command == "freeze-full":
        from .full import freeze_full
        print(json.dumps(freeze_full(root, cfg), indent=2, sort_keys=True))
        return 0
    if args.command == "full":
        from .full import run_full
        print(json.dumps(run_full(root, cfg), indent=2, sort_keys=True))
        return 0
    if args.command == "pilot":
        if not cfg["approved_for_pilot"]:
            raise BlockedError("pilot requires approved_for_pilot")
        from .pilot import run_pilot
        print(json.dumps(run_pilot(root, cfg), indent=2, sort_keys=True))
        return 0
    # The plan uses the Development subject IDs, discovered from the audit input.
    events = load_dev_events(root / cfg["inputs"]["events"],
                             root / "outputs/split_assignment.csv")
    actual = sorted(events.subject_id.unique())
    subjects = tuple(actual)
    outer = cfg["pilot"]["outer_subject"]
    grid = cfg["base_model"]["c_grid"]
    result = {"pilot": needed_basefits([outer], subjects, grid).counts(),
              "full": needed_basefits(subjects, subjects, grid).counts()}
    output = root / cfg["output_dir"] / "plan.json"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
