"""Development-only C2/C3 noise injection and resumable post-half PH jobs."""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import gudhi
import numpy as np
import pandas as pd
import wfdb

from topovital.cohort import BlockedError
from topovital.noise import assert_segment
from topovital.util import atomic_write_csv, atomic_write_json, config_hash, resolve

from . import features
from .core import FS, HALF, WINDOW, normalize_window, read_record_signal


def validate_config(cfg: dict) -> None:
    """Reject unsupported scientific variants rather than silently applying this protocol."""
    noise, injection, feat = (cfg.get(name, {}) for name in ("noise", "injection", "features"))
    required = {
        "noise.database": (noise.get("database"), "nstdb 1.0.0"),
        "noise.types": (noise.get("types"), ["bw", "ma", "em"]),
        "noise.development_block": (noise.get("development_block"), [0, 325000]),
        "noise.snippet_samples": (noise.get("snippet_samples"), HALF),
        "noise.allocation": (noise.get("allocation"), "non_overlapping_grid_seeded_permutation"),
        "noise.grid_step_samples": (noise.get("grid_step_samples"), HALF),
        "noise.same_snippet_for_c2_c3_and_all_snr": (noise.get("same_snippet_for_c2_c3_and_all_snr"), True),
        "noise.mean_removal": (noise.get("mean_removal"), True),
        "noise.onset_taper.kind": (noise.get("onset_taper", {}).get("kind"), "raised_cosine_ramp"),
        "noise.onset_taper.samples": (noise.get("onset_taper", {}).get("samples"), 36),
        "noise.onset_taper.duration_ms": (noise.get("onset_taper", {}).get("duration_ms"), 100),
        "noise.onset_taper.position": (noise.get("onset_taper", {}).get("position"), "post_window_start"),
        "noise.hard_onset_variant": (noise.get("hard_onset_variant"), "not_computed"),
        "injection.target": (injection.get("target"), "post_window_only"),
        "features.compute_post_half_only": (feat.get("compute_post_half_only"), True),
    }
    bad = [name for name, (actual, expected) in required.items() if actual != expected]
    if not isinstance(noise.get("channel"), int) or noise["channel"] < 0:
        bad.append("noise.channel")
    if not isinstance(noise.get("allocation_seed"), int) or noise["allocation_seed"] < 0:
        bad.append("noise.allocation_seed")
    snrs = injection.get("snr_db")
    if not isinstance(snrs, list) or not snrs or any(
        not isinstance(x, (int, float)) or not np.isfinite(x) for x in snrs
    ) or len(set(snrs)) != len(snrs) or any(not float(x).is_integer() for x in snrs):
        bad.append("injection.snr_db")
    threshold = injection.get("reject_if_noise_norm_below")
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        bad.append("injection.reject_if_noise_norm_below")
    for name in ("abs_snr_db", "rel_norm"):
        tolerance = injection.get("tolerance", {}).get(name)
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            bad.append(f"injection.tolerance.{name}")
    if feat.get("parameters_from") != "configs/stage1.yaml":
        bad.append("features.parameters_from")
    if feat.get("reuse_pre_half_from") != "outputs/stage1_features/development":
        bad.append("features.reuse_pre_half_from")
    if bad:
        raise BlockedError("Unsupported Regime 1 configuration: " + ", ".join(bad))


def require_run_scope(cfg: dict, events_count: int | None) -> None:
    validate_config(cfg)
    if events_count is None and cfg.get("approved_for_full_run") is not True:
        raise BlockedError("Regime 1 full run requires approved_for_full_run: true")


def allocation_table(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Allocate all Development events before selecting any representative subset."""
    validate_config(cfg)
    noise = cfg["noise"]
    start, stop = (int(x) for x in noise["development_block"])
    length = int(noise["snippet_samples"])
    step = int(noise["grid_step_samples"])
    slots = np.arange(start, stop - length + 1, step, dtype=int)
    ordered = events.sort_values("event_id", kind="stable")
    if ordered.event_id.duplicated().any() or len(ordered) > len(slots):
        raise BlockedError("Cannot allocate unique noise snippets to Development events")
    blocks = {"development": (start, stop), "test": (stop, 2 * stop)}
    rows = []
    for k, noise_type in enumerate(noise["types"]):
        permutation = np.random.default_rng([noise["allocation_seed"], k]).permutation(len(slots))
        for ev, slot_idx in zip(ordered.itertuples(index=False), permutation):
            snippet_start = int(slots[slot_idx])
            assert_segment(snippet_start, length, "development", blocks)
            rows.append({"event_id": ev.event_id, "noise_type": noise_type,
                         "snippet_start": snippet_start,
                         "snippet_id": f"{noise_type}_{snippet_start}"})
    counts = Counter(row["snippet_id"] for row in rows)
    for row in rows:
        row["reuse_count"] = counts[row["snippet_id"]]
    return pd.DataFrame(rows)


def taper(length: int, ramp_samples: int) -> np.ndarray:
    weights = np.ones(length, dtype=float)
    indices = np.arange(min(length, ramp_samples))
    weights[:len(indices)] = 0.5 * (1 - np.cos(np.pi * indices / ramp_samples))
    return weights


def inject(pre: np.ndarray, post: np.ndarray, snippet: np.ndarray, snr: float,
           cfg: dict) -> tuple[np.ndarray | None, dict]:
    """Inject physical noise and report both independent SNR checks."""
    noise_cfg, inj = cfg["noise"], cfg["injection"]
    result = {"snr_target": snr, "snr_actual": np.nan, "scale_a": np.nan,
              "norm_rel_error": np.nan, "signal_norm": np.nan, "noise_norm": np.nan,
              "scaled_noise_norm": np.nan, "target_noise_norm": np.nan,
              "pre_median": np.nan, "snippet_mean": np.nan,
              "snr_check_passed": False, "injection_failure_reason": ""}
    if len(pre) != len(post) or len(post) != len(snippet) or not all(
        np.isfinite(x).all() for x in (pre, post, snippet)
    ):
        result["injection_failure_reason"] = "invalid_input_signal"
        return None, result
    median = float(np.median(pre))
    mean = float(np.mean(snippet))
    signal_norm = float(np.linalg.norm(post - median))
    centered = snippet - mean if noise_cfg["mean_removal"] else snippet
    noise_eta = centered * taper(len(snippet), noise_cfg["onset_taper"]["samples"])
    noise_norm = float(np.linalg.norm(noise_eta))
    result.update(pre_median=median, snippet_mean=mean, signal_norm=signal_norm,
                  noise_norm=noise_norm)
    if not np.isfinite(noise_norm) or noise_norm < inj["reject_if_noise_norm_below"]:
        result["injection_failure_reason"] = "noise_norm_below_threshold"
        return None, result
    if not np.isfinite(signal_norm) or signal_norm <= 0:
        result["injection_failure_reason"] = "zero_signal_reference_norm"
        return None, result
    target_noise_norm = signal_norm * 10 ** (-snr / 20)
    scale_a = target_noise_norm / noise_norm
    scaled_noise = scale_a * noise_eta
    scaled_noise_norm = float(np.linalg.norm(scaled_noise))
    snr_actual = float(20 * np.log10(signal_norm / scaled_noise_norm))
    rel_error = abs(scaled_noise_norm - target_noise_norm) / target_noise_norm
    passed = (np.isfinite(snr_actual) and np.isfinite(rel_error)
              and abs(snr_actual - snr) < inj["tolerance"]["abs_snr_db"]
              and rel_error < inj["tolerance"]["rel_norm"])
    result.update(snr_actual=snr_actual, scale_a=float(scale_a),
                  norm_rel_error=float(rel_error), scaled_noise_norm=scaled_noise_norm,
                  target_noise_norm=float(target_noise_norm), snr_check_passed=bool(passed),
                  injection_failure_reason="" if passed else "snr_verification_failed")
    return (post + scaled_noise if passed else None), result


def pre_reuse(pre_normalized: np.ndarray, path: Path) -> tuple[str, str]:
    """Verify identical pre input without recomputing its topological features."""
    try:
        _, metadata = features.load_job(path)
    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        return "mismatch", f"pre_reference_unreadable:{type(exc).__name__}"
    if metadata.get("input_sha256") != features.input_hash(pre_normalized):
        return "mismatch", "pre_input_sha256_mismatch"
    return "verified", ""


def _physical_windows(events: pd.DataFrame, data_root: Path):
    jobs = []
    for ev in events.itertuples(index=False):
        jobs.extend(((str(ev.normal_record_id), int(ev.normal_start_sample), "C0", ev),
                     (str(ev.record_id), int(ev.start_abs), "C1", ev)))
    jobs.sort(key=lambda x: (x[0], x[3].event_id, x[2]))
    current_id, signal = None, None
    for rid, start, condition, ev in jobs:
        if rid != current_id:
            current_id = rid
            signal = read_record_signal(data_root / rid, int(ev.mlii_channel_index))
        end = int(ev.end_abs) if condition == "C1" else start + WINDOW
        if end - start != WINDOW or start < 0 or end > len(signal):
            raise BlockedError(f"Window outside Development record for {ev.event_id} {condition}")
        yield ev, rid, condition, signal[start:end]


def _noise_signals(cfg: dict, noise_root: Path) -> dict[str, np.ndarray]:
    noise = cfg["noise"]
    start, stop = noise["development_block"]
    signals = {}
    for noise_type in noise["types"]:
        record = wfdb.rdrecord(str(noise_root / noise_type), sampfrom=start, sampto=stop,
                               channels=[noise["channel"]], physical=True)
        if int(record.fs) != FS or record.p_signal.shape != (stop - start, 1):
            raise BlockedError(f"Unexpected NSTDB Development signal shape for {noise_type}")
        signals[noise_type] = np.asarray(record.p_signal[:, 0], dtype=float)
    return signals


def development_snippet(signal: np.ndarray, start: int, cfg: dict) -> np.ndarray:
    noise = cfg["noise"]
    dev_start, dev_stop = noise["development_block"]
    length = noise["snippet_samples"]
    assert_segment(start, length, "development",
                   {"development": (dev_start, dev_stop), "test": (dev_stop, 2 * dev_stop)})
    offset = start - dev_start
    return signal[offset:offset + length]


def _snr_label(snr: int | float) -> str:
    value = float(snr)
    if not value.is_integer():
        raise BlockedError("Regime 1 filename requires integer SNR targets")
    return f"m{int(-value)}" if value < 0 else str(int(value))


def _failed_row(info: dict, reason: str, run_id: str) -> dict:
    return {**info, "validation_status": "failed", "validation_messages": [reason],
            "vr_status": "skipped", "vr_failure_reason": reason, "cubical_status": "skipped",
            "vr_child_peak_rss_mb": 0.0, "cached": False, "run_id": run_id}


def run_regime1(all_events: pd.DataFrame, selected_events: pd.DataFrame, ecg_root: Path,
                noise_root: Path, out_dir: Path, cfg: dict, stage1_cfg: dict,
                workers: int, run_id: str) -> dict:
    validate_config(cfg)
    scientific = features.require_approved_parameters(stage1_cfg)
    features.check_memory(workers)
    if not set(selected_events.event_id).issubset(set(all_events.event_id)):
        raise BlockedError("Regime 1 subset contains an event outside the Development cohort")
    allocation = allocation_table(all_events, cfg)
    allocation_lookup = {(r.event_id, r.noise_type): r for r in allocation.itertuples(index=False)}
    selected = set(selected_events.event_id)
    source_dir = resolve(cfg["features"]["reuse_pre_half_from"]) / "jobs"
    jobs_dir = out_dir / "regime1_jobs"
    cfg_hash = config_hash({"regime1": cfg, "stage1": stage1_cfg})
    code_sha = features.stage1_code_hash()
    started = time.perf_counter()
    log_rows, ledger_rows = [], []
    signals = _noise_signals(cfg, noise_root)
    pending = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def finish(done):
            for future in done:
                try:
                    ledger_rows.append({**pending_info[future], **future.result()})
                except Exception as exc:
                    ledger_rows.append(_failed_row(pending_info.pop(future),
                                                   f"job_exception:{type(exc).__name__}:{exc}", run_id))
                else:
                    pending_info.pop(future)

        pending_info = {}
        for ev, rid, source_condition, window in _physical_windows(selected_events, ecg_root):
            target_condition = "C2" if source_condition == "C0" else "C3"
            pre, post = window[:HALF], window[HALF:]
            try:
                pre_norm, _ = normalize_window(window)
                norm_reason = ""
            except ValueError as exc:
                pre_norm, norm_reason = None, str(exc)
            pre_path = features.job_path(source_dir, {"event_id": ev.event_id,
                                                     "condition": source_condition, "half": "pre"})
            pre_status, pre_reason = (pre_reuse(pre_norm, pre_path) if pre_norm is not None
                                      else ("mismatch", f"normalization_failed:{norm_reason}"))
            pre_info = {"pre_reference": str(pre_path), "pre_reuse_status": pre_status,
                        "pre_reuse_failure_reason": pre_reason, "source_condition": source_condition}
            for noise_type in cfg["noise"]["types"]:
                slot = allocation_lookup[(ev.event_id, noise_type)]
                snippet = development_snippet(signals[noise_type], slot.snippet_start, cfg)
                for snr in cfg["injection"]["snr_db"]:
                    injected, numbers = inject(pre, post, snippet, snr, cfg)
                    common = {"event_id": ev.event_id, "subject_id": ev.subject_id,
                              "record_id": rid, "condition": target_condition, "half": "post",
                              "snippet_id": slot.snippet_id, "snippet_start": slot.snippet_start,
                              "noise_type": noise_type, "channel": cfg["noise"]["channel"],
                              "reuse_count": slot.reuse_count, **numbers, **pre_info}
                    path = features.job_path(jobs_dir, {
                        "event_id": ev.event_id,
                        "condition": f"{target_condition}__{noise_type}__{_snr_label(snr)}",
                        "half": "post"})
                    common["job_path"] = str(path)
                    log_rows.append(common.copy())
                    if not numbers["snr_check_passed"]:
                        ledger_rows.append(_failed_row(common, numbers["injection_failure_reason"], run_id))
                        continue
                    if pre_status != "verified":
                        ledger_rows.append(_failed_row(common, pre_reason, run_id))
                        continue
                    if pre_norm is None:
                        ledger_rows.append(_failed_row(common, norm_reason, run_id))
                        continue
                    # The original pre median and IQR are used for the injected post.
                    median = float(np.median(pre))
                    iqr = float(np.percentile(pre, 75, method="linear") -
                                np.percentile(pre, 25, method="linear"))
                    post_norm = (injected - median) / iqr
                    sha = features.input_hash(post_norm)
                    key = features.make_cache_key(sha, scientific, gudhi.__version__, code_sha)
                    cached = features.cached_metadata(path, key)
                    if cached is not None:
                        ledger_rows.append({**common, **cached, "cached": True, "run_id": run_id})
                        continue
                    future = pool.submit(features.compute_job, common, post_norm, path, key, sha,
                                         cfg_hash, code_sha, run_id, scientific)
                    pending.add(future)
                    pending_info[future] = common
                    if len(pending) >= workers:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        finish(done)
        if pending:
            done, _ = wait(pending)
            finish(done)
    expected = len(selected) * len(cfg["noise"]["types"]) * len(cfg["injection"]["snr_db"]) * 2
    if len(log_rows) != expected or len(ledger_rows) != expected:
        raise RuntimeError(f"Expected {expected} Regime 1 rows; got {len(log_rows)} logs and {len(ledger_rows)} jobs")
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["event_id", "condition", "noise_type", "snr_target"], kind="stable")
    log = pd.DataFrame(log_rows).sort_values(
        ["event_id", "condition", "noise_type", "snr_target"], kind="stable")
    ledger["cached"] = ledger["cached"].fillna(False).astype(bool) if "cached" in ledger else False
    summary = {"total_jobs": expected, "success": int(ledger.validation_status.eq("ok").sum()),
               "failed": int(ledger.validation_status.ne("ok").sum()),
               "cached": int(ledger.cached.sum()), "total_wall_time_s": time.perf_counter() - started,
               "max_child_peak_rss_mb": float(ledger.vr_child_peak_rss_mb.fillna(0).max()),
               "workers_used": workers, "snr_checks_passed": int(log.snr_check_passed.sum()),
               "snr_checks_total": expected, "run_id": run_id}
    atomic_write_csv(out_dir / "snippet_allocation.csv", allocation)
    atomic_write_csv(out_dir / "regime1_injection_log.csv", log)
    atomic_write_csv(out_dir / "regime1_ledger.csv", ledger)
    atomic_write_json(out_dir / "regime1_summary.json", summary)
    return summary
