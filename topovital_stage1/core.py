"""Validated Development input, half-window signals, and embedding diagnostics."""
from __future__ import annotations

import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

from topovital.cohort import BlockedError
from topovital.util import code_hash, config_hash, load_yaml

FS = 360
WINDOW = 3600
HALF = 1800
RTOL = 15.0
ATOL = 2.0
FNN_LIMIT = 0.05
FNN_CANDIDATES = (3, 5, 7)


def require_frozen(locked_path: Path) -> dict:
    if not locked_path.is_file():
        raise BlockedError(f"Stage 1 blocked: {locked_path} is missing")
    locked = load_yaml(locked_path)
    if locked.get("stage0_freeze_status") != "frozen" or locked.get("g_star") != 250:
        raise BlockedError("Stage 1 requires frozen locked_config.yaml with g_star=250")
    if locked.get("stage0_config_hash") != config_hash(locked.get("stage0")):
        raise BlockedError("Stage 1 blocked: frozen Stage 0 configuration hash mismatch")
    if locked.get("code_hash") != code_hash():
        raise BlockedError("Stage 1 blocked: frozen Stage 0 code hash mismatch")
    return locked


def load_dev_events(audit_path: Path, split_path: Path) -> pd.DataFrame:
    """Validate the whole requested cohort before any signal can be read."""
    audit = pd.read_parquet(audit_path)
    required = {"event_id", "subject_id", "record_id", "normal_record_id",
                "mlii_channel_index", "fs", "g_ms", "in_common_cohort", "split",
                "start_abs", "boundary_abs", "end_abs", "normal_start_sample"}
    missing = required - set(audit.columns)
    if missing:
        raise BlockedError(f"Development audit missing columns: {sorted(missing)}")
    events = audit.loc[audit.g_ms.eq(250) & audit.in_common_cohort.eq(True)].copy()
    if len(events) != 126 or events.event_id.nunique() != 126:
        raise BlockedError(f"Expected 126 unique Development events at g=250; found {len(events)}")
    splits = pd.read_csv(split_path, dtype={"subject_id": str})
    if not {"subject_id", "split"}.issubset(splits.columns) or splits.subject_id.duplicated().any():
        raise BlockedError("Invalid subject split assignment")
    assigned = splits.set_index("subject_id")["split"].to_dict()
    if not events.split.eq("development").all() or any(
        assigned.get(str(sid)) != "development" for sid in events.subject_id
    ):
        raise BlockedError("Stage 1 audit contains a non-Development or Test subject")
    if events[list(required - {"split", "in_common_cohort"})].isna().any().any():
        raise BlockedError("Development event has a missing required field")
    if not events.fs.eq(FS).all():
        raise BlockedError("Stage 1 requires fs=360 for every event")
    for ev in events.itertuples():
        if not (int(ev.start_abs) >= 0 and int(ev.boundary_abs) - int(ev.start_abs) == HALF
                and int(ev.end_abs) - int(ev.boundary_abs) == HALF
                and int(ev.normal_start_sample) >= 0):
            raise BlockedError(f"Invalid window for event {ev.event_id}")
        if not str(ev.record_id).isdigit() or not str(ev.normal_record_id).isdigit():
            raise BlockedError(f"Invalid record identifier for event {ev.event_id}")
        expected_subject = "S201" if str(ev.record_id) == "202" else f"S{ev.record_id}"
        if str(ev.subject_id) != expected_subject:
            raise BlockedError(f"Record/subject mismatch for event {ev.event_id}")
        if str(ev.normal_record_id) != str(ev.record_id) and not (
            {str(ev.record_id), str(ev.normal_record_id)} == {"201", "202"}
            and str(ev.subject_id) == "S201"
        ):
            raise BlockedError(f"Normal record is outside Development subject for {ev.event_id}")
    return events.sort_values("event_id", kind="stable").reset_index(drop=True)


def normalize_window(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(window, dtype=float)
    if x.ndim != 1 or len(x) != WINDOW or not np.isfinite(x).all():
        raise ValueError("invalid_window_signal")
    pre = x[:HALF]
    median = float(np.median(pre))
    iqr = float(np.percentile(pre, 75, method="linear") -
                np.percentile(pre, 25, method="linear"))
    if iqr == 0:
        raise ValueError("zero_pre_iqr")
    return (pre - median) / iqr, (x[HALF:] - median) / iqr


def delay_vectors(signal: np.ndarray, tau: int, m: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or tau < 1 or m < 1 or len(x) - (m - 1) * tau < 1:
        raise ValueError("insufficient_half_length_for_embedding")
    starts = np.arange(len(x) - (m - 1) * tau)
    return x[starts[:, None] + tau * np.arange(m)[None, :]]


def uniform_indices(n: int, n_points: int = 300) -> np.ndarray:
    if n < n_points:
        raise ValueError("insufficient_embedding_points")
    idx = np.linspace(0, n - 1, n_points).round().astype(int)
    if len(np.unique(idx)) != n_points or np.any(np.diff(idx) <= 0):
        raise ValueError("non_unique_uniform_indices")
    return idx


def ami_delay(signal: np.ndarray) -> tuple[float, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau, _ = nk.complexity_delay(signal, delay_max=100, method="fraser1986")
        if not np.isfinite(tau) or not 1 <= int(tau) <= 100:
            return np.nan, "no_first_local_minimum"
        return float(int(tau)), ""
    except Exception as exc:
        return np.nan, f"ami_error:{type(exc).__name__}:{exc}"


def fnn_fraction(signal: np.ndarray, tau: int, m: int,
                 rtol: float = RTOL, atol: float = ATOL) -> float:
    """Kennel's two criteria, with neighbors found in m dimensions in NumPy blocks.

    Only starts that have an (m+1)th coordinate are eligible. Self matches are
    excluded; no temporal exclusion window is imposed by the brief.
    """
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(x).all() or np.std(x) == 0:
        raise ValueError("invalid_or_constant_half")
    extended = delay_vectors(x, tau, m + 1)
    base = extended[:, :m]
    extra = extended[:, m]
    n = len(base)
    if n < 2:
        raise ValueError("insufficient_fnn_vectors")
    norms = np.sum(base * base, axis=1)
    false_count = 0
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        distance2 = norms[start:stop, None] + norms[None, :] - 2 * (base[start:stop] @ base.T)
        np.maximum(distance2, 0, out=distance2)
        distance2[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argmin(distance2, axis=1)
        d2 = distance2[np.arange(stop - start), nearest]
        delta = np.abs(extra[start:stop] - extra[nearest])
        ratio_false = np.where(d2 == 0, delta > 0, delta > rtol * np.sqrt(d2))
        scale_false = np.sqrt(d2 + delta * delta) / np.std(x) > atol
        false_count += int(np.count_nonzero(ratio_false | scale_false))
    return false_count / n


def diagnose_segment(signal: np.ndarray) -> dict:
    tau, reason = ami_delay(signal)
    used = int(tau) if np.isfinite(tau) else 10
    result = {"tau_ami": tau, "tau_failure_reason": reason,
              "fnn_tau_used": used, "fnn_tau_source": "ami" if not reason else "fallback_10",
              "fnn_rtol": RTOL, "fnn_atol": ATOL, "fnn_threshold": FNN_LIMIT,
              "m_fnn": np.nan, "m_failure_reason": ""}
    for m in FNN_CANDIDATES:
        try:
            fraction = fnn_fraction(signal, used, m)
        except Exception as exc:
            result["m_failure_reason"] = f"fnn_error:{type(exc).__name__}:{exc}"
            break
        result[f"fnn_fraction_m{m}"] = fraction
        if fraction < FNN_LIMIT:
            result["m_fnn"] = m
            break
    if not np.isfinite(result["m_fnn"]) and not result["m_failure_reason"]:
        result["m_failure_reason"] = "no_candidate_below_5_percent"
    return result


def read_record_signal(path: Path, channel: int) -> np.ndarray:
    rec = wfdb.rdrecord(str(path), channels=[channel], physical=True)
    if int(rec.fs) != FS or rec.sig_name[0] != "MLII":
        raise BlockedError(f"Record {path.name} channel {channel} is not MLII at 360 Hz")
    return np.asarray(rec.p_signal[:, 0], dtype=float)


def iter_segments(events: pd.DataFrame, data_root: Path):
    """Load one record at a time, yielding its C1 and C0 normalized halves."""
    jobs = []
    for ev in events.itertuples(index=False):
        jobs.extend([(str(ev.record_id), int(ev.start_abs), "C1", ev),
                     (str(ev.normal_record_id), int(ev.normal_start_sample), "C0", ev)])
    jobs.sort(key=lambda item: (item[0], item[3].event_id, item[2]))
    current_id = None
    signal = None
    for rid, start, condition, ev in jobs:
        if rid != current_id:
            signal = None
            current_id = rid
            signal = read_record_signal(data_root / rid, int(ev.mlii_channel_index))
        end = int(ev.end_abs) if condition == "C1" else start + WINDOW
        if end - start != WINDOW or end > len(signal):
            raise BlockedError(f"Window outside record for {ev.event_id} {condition}")
        try:
            halves = normalize_window(signal[start:end])
            failure = ""
        except ValueError as exc:
            halves = (None, None)
            failure = str(exc)
        for half, values in zip(("pre", "post"), halves):
            yield {"event_id": ev.event_id, "subject_id": ev.subject_id,
                   "record_id": rid, "condition": condition, "half": half,
                   "normalization_failure_reason": failure}, values


def embedding_summary(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (condition, half), group in rows.groupby(["condition", "half"], sort=True):
        row = {"condition": condition, "half": half, "n_segments": len(group)}
        for name in ("tau_ami", "m_fnn"):
            values = pd.to_numeric(group[name], errors="coerce").dropna().to_numpy()
            row[f"{name}_count"] = len(values)
            row[f"{name}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{name}_iqr"] = (float(np.percentile(values, 75, method="linear") -
                                          np.percentile(values, 25, method="linear")) if len(values) else np.nan)
            row[f"{name}_failure_pct"] = 100 * (len(group) - len(values)) / len(group)
        result.append(row)
    return pd.DataFrame(result)
