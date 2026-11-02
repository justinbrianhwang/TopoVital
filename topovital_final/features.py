"""Final PH jobs and classical features, using unchanged Stage 1 and Level 3 workers."""
from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from zipfile import BadZipFile

import gudhi
import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import atomic_write_csv, atomic_write_parquet, config_hash
from topovital_analysis.classical_features import KEYS, _feature_job, _worker_init
from topovital_analysis.distances import finite_diagram
from topovital_stage1 import features as stage_features


def _snr_label(value) -> str:
    value = int(value)
    return f"m{-value}" if value < 0 else str(value)


def _job_info(window: dict, half: str) -> dict:
    return ({key: window[key] for key in ("event_id", "subject_id", "record_id", "condition")}
            | {"half": half} | {key: window[key] for key in ("snippet_id", "snippet_start",
               "noise_type", "channel", "reuse_count", "snr_target", "snr_actual", "scale_a") if key in window})


def job_paths(output: Path, window: dict) -> tuple[Path, Path]:
    eid, condition = str(window["event_id"]), window["condition"]
    if condition in ("C0", "C1"):
        return (output / "jobs" / f"{eid}__{condition}__pre.npz",
                output / "jobs" / f"{eid}__{condition}__post.npz")
    source = "C0" if condition == "C2" else "C1"
    return (output / "jobs" / f"{eid}__{source}__pre.npz",
            output / "regime1_jobs" /
            f"{eid}__{condition}__{window['noise_type']}__{_snr_label(window['snr_target'])}__post.npz")


def run_features(windows: list[dict], output: Path, stage1_cfg: dict, stage0_cfg: dict,
                 *, workers: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output)
    scientific = stage_features.require_approved_parameters(stage1_cfg)
    stage_features.check_memory(workers)
    code_hash = stage_features.stage1_code_hash()
    cfg_hash = config_hash(stage1_cfg)
    ledger, pending, futures = [], set(), {}
    seen = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def finish(done):
            for future in done:
                info = futures.pop(future)
                try:
                    ledger.append(future.result())
                except Exception as exc:
                    ledger.append({**info, "validation_status": "failed",
                                   "validation_messages": f"job_exception:{type(exc).__name__}:{exc}"})

        for window in windows:
            pre_path, post_path = job_paths(output, window)
            for half, path in (("pre", pre_path), ("post", post_path)):
                if window["condition"] in ("C2", "C3") and half == "pre":
                    continue
                if path in seen:
                    continue
                seen.add(path)
                info = _job_info(window, half)
                signal = window[half]
                sha = stage_features.input_hash(signal) if signal is not None else ""
                key = stage_features.make_cache_key(sha, scientific, gudhi.__version__, code_hash)
                cached = stage_features.cached_metadata(path, key) if signal is not None else None
                if cached is not None:
                    ledger.append({**cached, "cached": True, "run_id": run_id})
                    continue
                future = pool.submit(stage_features.compute_job, info, signal, path, key, sha,
                                     cfg_hash, code_hash, run_id, scientific)
                pending.add(future)
                futures[future] = info
                if len(pending) >= workers:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    finish(done)
        if pending:
            done, _ = wait(pending)
            finish(done)
    job_frame = pd.DataFrame(ledger)
    atomic_write_csv(output / "jobs_ledger.csv", job_frame)
    jobs = []
    for window in windows:
        if window["physical_post"] is None:
            continue
        keys = {key: window[key] for key in KEYS}
        jobs.append((keys, window["physical_pre"], window["physical_post"],
                     window["pre"], window["post"], stage0_cfg))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        results = list(pool.map(_feature_job, jobs, chunksize=8))
    frame = pd.DataFrame([r[0] for r in results])
    pre_columns = [c for c in frame if c.endswith("__pre") or c.endswith("_pre_missing")]
    for target, source in (("C2", "C0"), ("C3", "C1")):
        mask = frame.condition.eq(target)
        reference = frame.loc[frame.condition.eq(source)].set_index("event_id")
        for col in pre_columns:
            frame.loc[mask, col] = reference.loc[frame.loc[mask, "event_id"], col].to_numpy()
    failures = pd.DataFrame([f for r in results for f in r[1]],
                            columns=[*KEYS, "half", "group", "feature", "reason"])
    def audit_key(row, half):
        snr = row["snr_target"]
        return (str(row["event_id"]), row["condition"], row["noise_type"],
                None if pd.isna(snr) else float(snr), half)

    audit_rows = [{**row, "passed": False} for row in failures.to_dict("records")]
    failed_halves = {audit_key(r, r["half"]) for r in failures.to_dict("records")}
    for row in frame.to_dict("records"):
        for half in ("pre", "post", "diff", "window"):
            key = audit_key(row, half)
            if key not in failed_halves:
                audit_rows.append({**{k: row[k] for k in KEYS}, "half": half,
                                   "group": "", "feature": "", "reason": "", "passed": True})
    for window in windows:
        if window["physical_post"] is not None:
            continue
        keys = {k: window[k] for k in KEYS}
        audit_rows.append({**keys, "half": "pre", "group": "", "feature": "",
                           "reason": "reused_clean_pre", "passed": True})
        for half in ("post", "diff", "window"):
            audit_rows.append({**keys, "half": half, "group": "waveform", "feature": "",
                               "reason": window.get("failure_reason", "injection_failed"),
                               "passed": False})
    audit = pd.DataFrame(audit_rows, columns=[*KEYS, "half", "group", "feature", "reason", "passed"])
    atomic_write_parquet(output / "classical_features.parquet", frame)
    atomic_write_csv(output / "test_feature_audit.csv", audit)
    return frame, failures, job_frame


def load_windows(windows: list[dict], output: Path, representations: dict) -> list[dict]:
    """Read final diagrams and retain failed windows for classical-only scoring."""
    loaded = {}
    def read(path):
        if path not in loaded:
            try:
                diagrams, meta = stage_features.load_job(path)
                if meta.get("validation_status") != "ok":
                    raise ValueError("invalid_job_validation_status")
                pairs = {rep: finite_diagram(diagrams[key], vr_h0=key == "vr_h0", vr_h1=key == "vr_h1")
                         for rep, key in representations.items()}
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                loaded[path] = (None, f"{type(exc).__name__}:{exc}")
            else:
                loaded[path] = (pairs, "")
        return loaded[path]
    result = []
    for window in windows:
        pre_path, post_path = job_paths(output, window)
        pre, pre_reason = read(pre_path)
        post, post_reason = read(post_path)
        if pre is None or post is None:
            result.append({**{k: window[k] for k in KEYS}, "record_id": window["record_id"],
                           "failure_reason": window.get("failure_reason", ""), "pre": None, "post": None,
                           "pi_failure_reason": pre_reason or post_reason})
        else:
            result.append({**{k: window[k] for k in KEYS}, "record_id": window["record_id"],
                           "failure_reason": window.get("failure_reason", ""), "pre": pre, "post": post,
                           "pi_failure_reason": ""})
    return result


FAILURE_COLUMNS = ["event_id", "subject_id", "record_id", "condition", "noise_type",
                   "snr_target", "half", "reason"]


def failure_log(windows: list[dict], ledger: pd.DataFrame) -> pd.DataFrame:
    """One row for each failed PH job and failed waveform injection."""
    rows = []
    for row in ledger.to_dict("records"):
        if row.get("validation_status") != "ok":
            rows.append({key: row.get(key) for key in FAILURE_COLUMNS[:-1]} |
                        {"reason": str(row.get("validation_messages") or "ph_job_failed")})
    for window in windows:
        if window.get("physical_post") is None:
            rows.append({key: window.get(key) for key in FAILURE_COLUMNS[:-2]} |
                        {"half": "post", "reason": window.get("failure_reason", "injection_failed")})
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)
