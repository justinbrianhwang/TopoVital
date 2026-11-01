"""CLI for the Development-only Stage 1 pilot."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import (PROJECT_ROOT, atomic_write_csv, atomic_write_text,
                            config_hash, load_yaml, new_run_id, profiled)

from .core import embedding_summary, diagnose_segment, iter_segments, load_dev_events, require_frozen
from .ph import pilot_half
from .features import representative_events, require_approved_parameters, run_features, check_memory
from .regime1 import require_run_scope, run_regime1

OUT = PROJECT_ROOT / "outputs"
PILOT = OUT / "stage1_pilot"
DATA = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"

PENDING_TEXT = """# Embedding selection — PENDING researcher decisions

These diagnostics and the PH cost pilot do not select downstream settings.

- Decide how per-segment tau_ami and m_fnn become one Development value (for example, median or mode), including treatment of failures.
- Decide the AMI histogram binning and any other NeuroKit2 estimation settings; this pilot calls `complexity_delay(..., delay_max=100, method=\"fraser1986\")` with library defaults.
- Approve or revise the FNN criteria used in this diagnostic pilot: Rtol=15, Atol=2, <5%, candidate m in [3,5,7].
- Approve or revise the 300-point subsampling rule. The resource pilot uses `np.linspace(0,n-1,300).round()` only for cost measurement.
- Decide the fixed r_max selection rule on Development. The resource pilot uses no maximum edge length to measure upper-bound cost.

The PH pilot uses fallback tau=10 and m=5 only for resource measurement.
"""


def run_embedding(events: pd.DataFrame, run_id: str, cfg_hash: str, stage0_code_hash: str) -> None:
    rows = []
    for info, half in iter_segments(events, DATA):
        row = {**info, "run_id": run_id, "config_hash": cfg_hash, "code_hash": stage0_code_hash}
        if half is None:
            row.update(tau_ami=np.nan, tau_failure_reason=info["normalization_failure_reason"],
                       m_fnn=np.nan, m_failure_reason=info["normalization_failure_reason"],
                       fnn_rtol=15, fnn_atol=2, fnn_threshold=0.05)
        else:
            row.update(diagnose_segment(half))
        rows.append(row)
    if len(rows) != 504:
        raise RuntimeError(f"Expected 504 embedding rows; got {len(rows)}")
    frame = pd.DataFrame(rows).sort_values(["event_id", "condition", "half"], kind="stable")
    atomic_write_csv(PILOT / "embedding_diagnostics_dev.csv", frame)
    summary = embedding_summary(frame)
    summary["run_id"] = run_id
    summary["config_hash"] = cfg_hash
    summary["code_hash"] = stage0_code_hash
    atomic_write_csv(PILOT / "embedding_diagnostics_summary_dev.csv", summary)
    atomic_write_text(PILOT / "embedding_selection_PENDING.md", PENDING_TEXT)
    print(f"Embedding outputs: {PILOT}")


def run_ph(events: pd.DataFrame, run_id: str, cfg_hash: str, stage0_code_hash: str) -> None:
    subset = events.head(5)
    rows = []
    for info, half in iter_segments(subset, DATA):
        row = {**info, "run_id": run_id, "config_hash": cfg_hash, "code_hash": stage0_code_hash,
               "pilot_tau": 10, "pilot_m": 5, "pilot_n_points": 300,
               "subsample_rule": "np.linspace(0,n-1,300).round() (pilot pending approval)",
               "r_max_rule": "unset (full Rips complex; upper-bound cost)",
               "vr_max_dimension": 2}
        if half is None:
            row.update(vr_status="failed_normalization", vr_failure_reason=info["normalization_failure_reason"],
                       cubical_status="failed_normalization",
                       cubical_failure_reason=info["normalization_failure_reason"])
        else:
            row.update(pilot_half(half))
        rows.append(row)
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 PH jobs; got {len(rows)}")
    frame = pd.DataFrame(rows).sort_values(["event_id", "condition", "half"], kind="stable")
    atomic_write_csv(PILOT / "ph_resource_pilot_dev.csv", frame)
    atomic_write_text(PILOT / "embedding_selection_PENDING.md", PENDING_TEXT)
    print(f"PH resource output: {PILOT / 'ph_resource_pilot_dev.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m topovital_stage1")
    parser.add_argument("command", choices=["pilot", "features", "regime1"])
    parser.add_argument("--only", choices=["embedding", "ph"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--events", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "regime1":
            if args.only is not None:
                parser.error("--only applies only to pilot")
            if args.events is not None and args.events < 1:
                parser.error("--events must be positive")
            regime_cfg = load_yaml(PROJECT_ROOT / "configs" / "stage1_regime1.yaml")
            require_run_scope(regime_cfg, args.events)
            stage1_cfg = load_yaml(PROJECT_ROOT / regime_cfg["features"]["parameters_from"])
            require_approved_parameters(stage1_cfg)
            check_memory(args.workers)
            require_frozen(OUT / "locked_config.yaml")
            all_events = load_dev_events(OUT / "development" / "event_audit_dev.parquet",
                                         OUT / "split_assignment.csv")
            selected = representative_events(all_events, args.events) if args.events is not None else all_events
            cfg_hash = config_hash({"regime1": regime_cfg, "stage1": stage1_cfg})
            run_id = new_run_id(cfg_hash)
            out_dir = OUT / "stage1_features" / "development"
            command = f"stage1 regime1 --workers {args.workers}" + (f" --events {args.events}" if args.events else "")
            with profiled(OUT, run_id, command, n_jobs=args.workers) as rec:
                summary = run_regime1(all_events, selected, DATA,
                                      PROJECT_ROOT / "data" / "raw" / "mit-bih-noise-stress-test-database-1.0.0",
                                      out_dir, regime_cfg, stage1_cfg, args.workers, run_id)
                rec["cache_used"] = summary["cached"] > 0
                rec["note"] = (f"jobs={summary['total_jobs']}; success={summary['success']}; "
                               f"failed={summary['failed']}; cached={summary['cached']}")
            print(f"Regime 1: {summary['success']}/{summary['total_jobs']} success, "
                  f"{summary['failed']} failed, {summary['cached']} cached; "
                  f"output={out_dir}")
            return 0
        if args.command == "features":
            if args.only is not None:
                parser.error("--only applies only to pilot")
            if args.events is not None and args.events < 1:
                parser.error("--events must be positive")
            cfg = load_yaml(PROJECT_ROOT / "configs" / "stage1.yaml")
            require_approved_parameters(cfg)
            check_memory(args.workers)
            require_frozen(OUT / "locked_config.yaml")
            events = load_dev_events(OUT / "development" / "event_audit_dev.parquet",
                                     OUT / "split_assignment.csv")
            if args.events is not None:
                events = representative_events(events, args.events)
            cfg_hash = config_hash(cfg)
            run_id = new_run_id(cfg_hash)
            command = f"stage1 features --workers {args.workers}" + (f" --events {args.events}" if args.events else "")
            with profiled(OUT, run_id, command, n_jobs=args.workers) as rec:
                summary = run_features(events, DATA, OUT / "stage1_features" / "development",
                                       cfg, args.workers, run_id)
                rec["cache_used"] = summary["cached"] > 0
                rec["note"] = f"jobs={summary['total_jobs']}; success={summary['success']}; failed={summary['failed']}; cached={summary['cached']}"
            print(f"Features: {summary['success']}/{summary['total_jobs']} success, "
                  f"{summary['failed']} failed, {summary['cached']} cached; "
                  f"workers={args.workers}; output={OUT / 'stage1_features' / 'development'}")
            return 0
        if args.events is not None or args.workers != 1:
            parser.error("--events and --workers apply only to features")
        locked = require_frozen(OUT / "locked_config.yaml")
        events = load_dev_events(OUT / "development" / "event_audit_dev.parquet",
                                 OUT / "split_assignment.csv")
        cfg_hash = config_hash(locked)
        run_id = new_run_id(cfg_hash)
        with profiled(OUT, run_id, "stage1 pilot" + (f" --only {args.only}" if args.only else ""), n_jobs=1) as rec:
            if args.only != "ph":
                run_embedding(events, run_id, cfg_hash, locked["code_hash"])
            if args.only != "embedding":
                run_ph(events, run_id, cfg_hash, locked["code_hash"])
            rec["note"] = f"stage1_pilot; n_events={len(events)}; ph_jobs={20 if args.only != 'embedding' else 0}"
        print(f"Resource profile: {OUT / 'resource_profile.csv'}")
        return 0
    except BlockedError as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
