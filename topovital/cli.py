"""Command line interface: python -m topovital <command>."""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

from topovital.cohort import BlockedError
from topovital.util import (PROJECT_ROOT, atomic_write_csv, atomic_write_json, atomic_write_text,
                            atomic_write_yaml, config_hash, load_yaml, new_run_id, profiled, resolve)

# Stage 0 runs record-by-record in one process; n_jobs is kept at 1 (brief §3).

EXIT_BLOCKED = 3
DEFAULT_CONFIG = "configs/stage0.yaml"


def _cmd_env_check(args, cfg, out, rec):
    from topovital.envcheck import collect, missing_required

    env = collect()
    atomic_write_json(out / "environment.json", env)
    lock = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    atomic_write_text(PROJECT_ROOT / "requirements.lock.txt",
                      "# Exact versions installed and import-verified in this environment (pip freeze equivalent)\n"
                      + "\n".join(lock) + "\n")
    miss = missing_required(env)
    print(f"environment.json written; logical cores={env['cpu']['logical_cores']}, "
          f"RAM total={env['memory_bytes']['total'] / 2**30:.1f} GiB, GPU probe={env['gpu_optional_probe']}")
    if miss:
        rec["status"] = "failed"
        rec["note"] = f"missing required packages: {miss}"
        print(rec["note"], file=sys.stderr)
        return 1
    return 0


def _cmd_fetch(args, cfg, out, rec):
    from topovital.fetch import fetch

    df, info = fetch(cfg)
    atomic_write_csv(out / "input_checksums.csv", df)
    atomic_write_json(out / "data_provenance.json", info)
    rec["cache_used"] = all(v["newly_extracted"] == 0 for v in info.values())
    rec["note"] = "; ".join(f"{k}: {v['files_listed']} files verified, {v['newly_extracted']} extracted"
                            for k, v in info.items())
    print(rec["note"])
    return 0


def _cmd_cohort(args, cfg, out, rec, run_id):
    from topovital.stage0 import run_cohort

    s = run_cohort(cfg, out, run_id, rec)
    print(f"cohort: {s['n_source_records']}/{s['n_source_subjects']} -> "
          f"{s['n_cohort_records']}/{s['n_cohort_subjects']} (records/subjects); "
          f"step-1 'V' candidates={s['n_step1_pvc_candidates']}; cache_used={rec['cache_used']}")
    return 0


def _cmd_stage0(args, cfg, out, rec, run_id):
    from topovital.gate import require_test_allowed
    from topovital.stage0 import run_development_sweep

    if args.split == "test":
        locked = require_test_allowed(resolve(args.config or "outputs/locked_config.yaml"), args.g_sweep)
        from topovital.stage0 import run_test
        run_test(locked["stage0"], out, run_id, int(locked["g_star"]))
        return 0
    if not args.g_sweep:
        raise BlockedError("Development run requires --g-sweep (candidate g values from config).")
    run_development_sweep(cfg, out, run_id)
    return 0


def _cmd_split(args, cfg, out, rec, run_id):
    from topovital.stage0 import run_split
    assignment = run_split(cfg, out, run_id)
    print(f"split: {(assignment.split == 'development').sum()} Development / "
          f"{(assignment.split == 'test').sum()} Test subjects")
    return 0


def _cmd_freeze(args, cfg, out, rec):
    from topovital.gate import freeze

    locked = freeze(cfg, resolve(args.decision), out)
    print(f"Stage 0 frozen with g_star={locked['g_star']} ms")
    return 0


def _cmd_report(args, cfg, out, rec):
    from topovital.gate import build_locked_config, pending_settings
    from topovital.report import write_report

    locked_path = out / "locked_config.yaml"
    existing = load_yaml(locked_path) if locked_path.exists() else None
    if not (existing and existing.get("stage0_freeze_status") == "frozen"):
        atomic_write_yaml(locked_path, build_locked_config(cfg, "pending", None, pending_settings(cfg)))
    write_report(cfg, out)
    print(f"wrote {out / 'stage0_report.md'} and {locked_path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m topovital")
    p.add_argument("--out", default="outputs")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("env-check", "fetch", "cohort", "split", "report-stage0"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=DEFAULT_CONFIG)
    sp = sub.add_parser("stage0")
    sp.add_argument("--split", choices=["development", "test"], required=True)
    sp.add_argument("--g-sweep", action="store_true")
    sp.add_argument("--config", default=None,
                    help="development: stage0 config (default configs/stage0.yaml); test: locked config")
    sp = sub.add_parser("freeze-stage0")
    sp.add_argument("--decision", required=True)
    sp.add_argument("--config", default=DEFAULT_CONFIG)
    args = p.parse_args(argv)

    # For `stage0 --split test`, --config names the locked config; the base config is fixed.
    test_run = args.cmd == "stage0" and args.split == "test"
    cfg = load_yaml(resolve(DEFAULT_CONFIG if test_run else (args.config or DEFAULT_CONFIG)))
    out = resolve(args.out)
    run_id = new_run_id(config_hash(cfg))
    command = " ".join(["python -m topovital", *(argv if argv is not None else sys.argv[1:])])
    handlers = {
        "env-check": lambda rec: _cmd_env_check(args, cfg, out, rec),
        "fetch": lambda rec: _cmd_fetch(args, cfg, out, rec),
        "cohort": lambda rec: _cmd_cohort(args, cfg, out, rec, run_id),
        "split": lambda rec: _cmd_split(args, cfg, out, rec, run_id),
        "stage0": lambda rec: _cmd_stage0(args, cfg, out, rec, run_id),
        "freeze-stage0": lambda rec: _cmd_freeze(args, cfg, out, rec),
        "report-stage0": lambda rec: _cmd_report(args, cfg, out, rec),
    }
    with profiled(out, run_id, command, n_jobs=cfg.get("execution", {}).get("n_jobs", 1)) as rec:
        try:
            rc = handlers[args.cmd](rec)
        except BlockedError as e:
            rec["status"] = "blocked"
            rec["note"] = str(e).splitlines()[0]
            print(f"BLOCKED: {e}", file=sys.stderr)
            rc = EXIT_BLOCKED
        rec["exit_code"] = rc
        if rc and rec["status"] == "success":
            rec["status"] = "failed"
    return rc
