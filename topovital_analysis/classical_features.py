"""Development waveform regeneration and fold-independent classical features."""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
from threadpoolctl import threadpool_limits
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, skew
from wfdb.processing import xqrs_detect

from topovital.cohort import BlockedError
from topovital.stage0_pipeline import bsqi
from topovital.util import atomic_write_csv, atomic_write_json, atomic_write_parquet, peak_rss_mb
from topovital_stage1.core import normalize_window
from topovital_stage1.features import input_hash, load_job
from topovital_stage1.regime1 import (_noise_signals, _physical_windows, allocation_table,
                                       development_snippet, inject)

from .level1 import analysis_code_hash, validate_events, window_specs

FS = 360
KEYS = ("event_id", "subject_id", "condition", "noise_type", "snr_target")
GROUPS = {
    "rhythm": ("n_beats", "median_rr_s", "rr_cv"),
    "qrs": ("median_qrs_width_ms", "max_qrs_width_ms", "median_abs_r_amplitude_mv"),
    "template": ("min_template_correlation", "max_normalized_rms_difference"),
    "dwt": tuple(f"relative_energy_{x}" for x in ("d1", "d2", "d3", "d4", "d5", "a5")),
    "psd": ("relative_power_0_1", "relative_power_1_5", "relative_power_5_15",
            "relative_power_15_40", "relative_power_40_180", "spectral_centroid_hz"),
    "spectral_entropy": ("normalized_entropy",),
    "sqi": ("bsqi", "ksqi", "ssqi", "psqi", "bassqi"),
    "amplitude": ("min", "max", "range"),
    "boundary": ("rr_cross_ratio", "min_rr_ratio_post", "compensatory_ratio"),
}


def _result(names, reason):
    return {name: np.nan for name in names}, {name: reason for name in names}


def _clean(values, reasons):
    for name, value in values.items():
        if not np.isfinite(value):
            values[name] = np.nan
            reasons[name] = reasons.get(name) or "nonfinite"
    return values, reasons


def filtered_peaks(signal):
    sos = butter(4, [0.5, 40], btype="bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=float))
    try:
        detected = xqrs_detect(filtered, FS, verbose=False)
    except Exception:
        detected = []
    radius = round(.05 * FS)
    peaks = sorted({int(np.argmax(np.abs(filtered[max(0, p-radius):min(len(filtered), p+radius+1)])) + max(0, p-radius))
                    for p in detected})
    return filtered, np.asarray(peaks, dtype=int)


def rhythm(signal, peaks=None):
    if peaks is None:
        _, peaks = filtered_peaks(signal)
    values = {"n_beats": float(len(peaks)), "median_rr_s": np.nan, "rr_cv": np.nan}
    reasons = {}
    if len(peaks) < 2:
        reasons["median_rr_s"] = "too_few_beats"
    else:
        rr = np.diff(peaks) / FS
        values["median_rr_s"] = float(np.median(rr))
        if len(peaks) < 3:
            reasons["rr_cv"] = "too_few_beats"
        else:
            values["rr_cv"] = _log_ratio(float(np.std(rr) / np.median(rr)), reasons, "rr_cv")
    return _clean(values, reasons)


def _log_ratio(value, reasons, name):
    if not np.isfinite(value) or value <= 0:
        reasons[name] = "nonpositive_ratio"
        return np.nan
    reasons.pop(name, None)
    return float(np.log(value))


def qrs(signal, filtered=None, peaks=None):
    if filtered is None or peaks is None:
        filtered, peaks = filtered_peaks(signal)
    values, reasons = _result(GROUPS["qrs"], "too_few_beats")
    if len(peaks):
        values["median_abs_r_amplitude_mv"] = float(np.median(np.abs(filtered[peaks])))
        reasons.pop("median_abs_r_amplitude_mv")
    try:
        _, marks = nk.ecg_delineate(filtered, rpeaks=peaks, sampling_rate=FS, method="dwt")
        onset = np.asarray(marks["ECG_R_Onsets"], dtype=float)
        offset = np.asarray(marks["ECG_R_Offsets"], dtype=float)
        width = (offset - onset) * 1000 / FS
        width = width[np.isfinite(width) & (width > 0)]
        if len(width):
            values["median_qrs_width_ms"] = float(np.median(width))
            values["max_qrs_width_ms"] = float(np.max(width))
            reasons.pop("median_qrs_width_ms")
            reasons.pop("max_qrs_width_ms")
        else:
            reasons.update({name: "delineation_failed:no_valid_width" for name in GROUPS["qrs"][:2]})
    except Exception as exc:
        reasons.update({name: f"delineation_failed:{type(exc).__name__}" for name in GROUPS["qrs"][:2]})
    return _clean(values, reasons)


def beat_segments(filtered, peaks):
    left, right = round(.25 * FS), round(.45 * FS)
    return np.asarray([filtered[p-left:p+right+1] for p in peaks if p-left >= 0 and p+right < len(filtered)])


def _correlation(a, b):
    x, y = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def _template_with_reference(signal, pre_template=None, filtered=None, peaks=None):
    if filtered is None or peaks is None:
        filtered, peaks = filtered_peaks(signal)
    segments = beat_segments(filtered, peaks)
    if pre_template is None:
        if len(segments) < 2:
            return *_result(GROUPS["template"], "no_template"), None
        pre_template = np.median(segments, axis=0)
    values, reasons = _result(GROUPS["template"], "too_few_beats")
    if len(segments):
        denom = float(np.sqrt(np.mean(pre_template ** 2)))
        corr = [_correlation(segment, pre_template) for segment in segments]
        values["min_template_correlation"] = float(np.min(corr))
        values["max_normalized_rms_difference"] = _log_ratio(
            float(np.max([np.sqrt(np.mean((s-pre_template)**2)) / denom for s in segments])) if denom > 0 else np.nan,
            reasons, "max_normalized_rms_difference")
        reasons.pop("min_template_correlation", None)
    return *_clean(values, reasons), pre_template


def template(signal, pre_template=None, filtered=None, peaks=None):
    values, reasons, _ = _template_with_reference(signal, pre_template, filtered, peaks)
    return values, reasons


def dwt(signal):
    coeffs = pywt.wavedec(np.asarray(signal, dtype=float), "db4", mode="symmetric", level=5)
    energies = np.asarray([np.sum(c*c) for c in coeffs[::-1]], dtype=float)
    total = energies.sum()
    if total <= 0 or not np.isfinite(total):
        return _result(GROUPS["dwt"], "nonpositive_power")
    return _clean(dict(zip(GROUPS["dwt"], energies / total)), {})


def _spectrum(signal):
    x = np.asarray(signal, dtype=float)
    return welch(x - x.mean(), fs=FS, window="hann", nperseg=720, noverlap=360,
                 detrend="constant", scaling="density")


def _bands(freq, power):
    bounds = ((0, 1), (1, 5), (5, 15), (15, 40), (40, 180))
    # Bin sums assign each frequency to exactly one band, including Nyquist.
    return np.asarray([power[(freq >= lo) & ((freq < hi) if hi < 180 else (freq <= hi))].sum()
                       for lo, hi in bounds])


def psd(signal):
    freq, power = _spectrum(signal)
    bands = _bands(freq, power)
    total = bands.sum()
    if total <= 0 or not np.isfinite(total):
        return _result(GROUPS["psd"], "nonpositive_power")
    values = dict(zip(GROUPS["psd"][:5], bands / total))
    values["spectral_centroid_hz"] = float(np.sum(freq * power) / np.sum(power))
    return _clean(values, {})


def spectral_entropy(signal):
    freq, power = _spectrum(signal)
    selected = power[(freq >= .5) & (freq <= 40)]
    total = selected.sum()
    if total <= 0 or not np.isfinite(total):
        return _result(GROUPS["spectral_entropy"], "nonpositive_power")
    p = selected / total
    value = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p))
    return _clean({"normalized_entropy": float(value)}, {})


def sqi(signal, stage0_cfg):
    values, reasons = {}, {}
    values["bsqi"], error = bsqi(np.asarray(signal), FS, stage0_cfg["quality"]["bsqi"])
    if error:
        reasons["bsqi"] = error
    values["ksqi"] = float(kurtosis(signal, fisher=False))
    values["ssqi"] = float(skew(signal))
    freq, power = _spectrum(signal)
    bands = _bands(freq, power)
    values["psqi"] = float(bands[2] / (bands[2] + bands[3])) if bands[2] + bands[3] > 0 else np.nan
    values["bassqi"] = float(1 - bands[0] / bands[:4].sum()) if bands[:4].sum() > 0 else np.nan
    return _clean(values, reasons)


def _boundary_with_latency(window, *, filtered=None, peaks=None):
    if filtered is None or peaks is None:
        filtered, peaks = filtered_peaks(window)
    values, reasons = _result(GROUPS["boundary"], "too_few_beats")
    post = np.flatnonzero(peaks >= len(window) // 2)
    latency = float((peaks[post[0]] - len(window) // 2) / FS) if len(post) else np.nan
    before = peaks[peaks < len(window) // 2]
    if len(before) < 2 or not len(post):
        return values, reasons, latency, "" if len(post) else "no_post_beat"
    reference = float(np.median(np.diff(before)))
    first = int(post[0])
    values["rr_cross_ratio"] = _log_ratio((peaks[first] - peaks[first-1]) / reference, reasons, "rr_cross_ratio")
    eligible = post[post > 0]
    ratios = np.diff(peaks)[eligible-1] / reference
    chosen = int(eligible[np.argmin(ratios)])
    values["min_rr_ratio_post"] = _log_ratio(float(np.min(ratios)), reasons, "min_rr_ratio_post")
    if chosen + 1 < len(peaks):
        value = (peaks[chosen+1] - peaks[chosen-1]) / (2 * reference)
        values["compensatory_ratio"] = _log_ratio(value, reasons, "compensatory_ratio")
    else:
        reasons["compensatory_ratio"] = "no_following_beat"
    return *_clean(values, reasons), latency, ""


def boundary(window, *, filtered=None, peaks=None):
    values, reasons, _, _ = _boundary_with_latency(window, filtered=filtered, peaks=peaks)
    return values, reasons


def amplitude(signal):
    signal = np.asarray(signal, dtype=float)
    return _clean({"min": float(np.min(signal)), "max": float(np.max(signal)),
                   "range": float(np.ptp(signal))}, {})


def _half(signal, stage0_cfg, pre_template=None, *, require_pre_template=False):
    filtered, peaks = filtered_peaks(signal)
    groups = {"rhythm": rhythm(signal, peaks), "qrs": qrs(signal, filtered, peaks),
              "dwt": dwt(signal), "psd": psd(signal),
              "spectral_entropy": spectral_entropy(signal), "sqi": sqi(signal, stage0_cfg),
              "amplitude": amplitude(signal)}
    if require_pre_template and pre_template is None:
        tv, tr = _result(GROUPS["template"], "no_template")
    else:
        tv, tr, pre_template = _template_with_reference(signal, pre_template, filtered, peaks)
    groups["template"] = (tv, tr)
    return groups, pre_template


def _worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    threadpool_limits(limits=1)


def _feature_job(job):
    keys, pre, post, pre_norm, post_norm, cfg_s0 = job
    pre_groups, pre_template = _half(pre, cfg_s0)
    post_groups, _ = _half(post, cfg_s0, pre_template, require_pre_template=True)
    for groups, normalized in ((pre_groups, pre_norm), (post_groups, post_norm)):
        groups["amplitude"] = amplitude(normalized)
    bvalues, breasons, latency, latency_reason = _boundary_with_latency(np.concatenate((pre, post)))
    row, failures = dict(keys), []
    for group in GROUPS:
        if group == "boundary":
            for name, value in bvalues.items():
                row[f"boundary__{name}"] = value
                if np.isnan(value):
                    failures.append({**keys, "half": "window", "group": group, "feature": name, "reason": breasons[name]})
            continue
        before, before_reason = pre_groups[group]
        after, after_reason = post_groups[group]
        for name in GROUPS[group]:
            a, b = before[name], after[name]
            for half, value, reasons in (("pre", a, before_reason), ("post", b, after_reason)):
                row[f"{group}__{name}__{half}"] = value
                if np.isnan(value) or reasons.get(name):
                    failures.append({**keys, "half": half, "group": group, "feature": name, "reason": reasons[name]})
            row[f"{group}__{name}__diff"] = b-a if np.isfinite(a) and np.isfinite(b) else np.nan
            if np.isnan(row[f"{group}__{name}__diff"]):
                failures.append({**keys, "half": "diff", "group": group, "feature": name,
                                 "reason": after_reason.get(name) or before_reason.get(name) or "missing_operand"})
    for group in ("rhythm", "qrs", "template"):
        for half in ("pre", "post"):
            row[f"{group}_{half}_missing"] = int(any(np.isnan(row[f"{group}__{n}__{half}"]) for n in GROUPS[group]))
    row["boundary_missing"] = int(any(np.isnan(row[f"boundary__{n}"]) for n in GROUPS["boundary"]))
    return row, failures, {**keys, "first_beat_latency_s": latency, "reason": latency_reason}, peak_rss_mb()


def _verify(path, half, keys, checks, *, snippet_start=None, numbers=None, tolerance=None):
    try:
        diagrams, meta = load_job(path)
    except (OSError, ValueError, KeyError) as exc:
        checks.append({**keys, "half": "post" if snippet_start is not None else path.stem.rsplit("__", 1)[-1],
                       "job_path": str(path), "passed": False, "reason": f"job_unreadable:{type(exc).__name__}"})
        return
    sha_match = meta.get("input_sha256") == input_hash(half)
    passed = sha_match and meta.get("validation_status") == "ok"
    passed &= meta.get("event_id") == keys["event_id"] and meta.get("condition") == keys["condition"]
    if keys["condition"] in ("C2", "C3"):
        passed &= meta.get("noise_type") == keys["noise_type"] and meta.get("snr_target") == keys["snr_target"]
    amplitude_min_match = (len(diagrams["cubical_sub_essential_births"]) == 1 and
                           bool(np.isclose(np.min(half), diagrams["cubical_sub_essential_births"][0], atol=1e-12, rtol=0)))
    amplitude_max_match = (len(diagrams["cubical_super_essential_births"]) == 1 and
                           bool(np.isclose(np.max(half), -diagrams["cubical_super_essential_births"][0], atol=1e-12, rtol=0)))
    passed &= amplitude_min_match and amplitude_max_match
    scale_diff, snr_diff = np.nan, np.nan
    if snippet_start is not None:
        passed &= meta.get("snippet_start") == snippet_start
        scale_meta = meta.get("scale_a", np.nan)
        snr_meta = meta.get("snr_actual", np.nan)
        scale_diff = abs(scale_meta - numbers["scale_a"]) / abs(scale_meta) if isinstance(scale_meta, (int, float)) and scale_meta != 0 else np.inf
        snr_diff = abs(snr_meta - numbers["snr_actual"]) if isinstance(snr_meta, (int, float)) else np.inf
        passed &= scale_diff <= tolerance["scale_a_rel_tol"] and snr_diff <= tolerance["snr_actual_abs_tol_db"]
    checks.append({**keys, "half": meta.get("half"), "job_path": str(path), "passed": bool(passed),
                   "input_sha256_match": sha_match, "amplitude_min_match": amplitude_min_match,
                   "amplitude_max_match": amplitude_max_match,
                   "snippet_start_match": snippet_start is None or meta.get("snippet_start") == snippet_start,
                   "scale_a_rel_diff": scale_diff, "snr_actual_abs_diff_db": snr_diff})


def run_level3_features(events, cfg_l3, cfg_a, cfg_r1, cfg_s0, root, output, *, workers, log=print):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    expected = root / "outputs/stage1_analysis/development/level3_features"
    if output.resolve() != expected.resolve():
        raise BlockedError("Level 3 feature output directory must be level3_features under Development")
    if workers < 1:
        raise BlockedError("workers must be positive")
    validate_events(events, int(cfg_a["scope"]["events"]))
    noise = _noise_signals(cfg_r1, root / "data/raw/mit-bih-noise-stress-test-database-1.0.0")
    allocation = {(r.event_id, r.noise_type): r.snippet_start for r in allocation_table(events, cfg_r1).itertuples()}
    physical = {(ev.event_id, condition): window for ev, _, condition, window in
                _physical_windows(events, root / "data/raw/mit-bih-arrhythmia-database-1.0.0")}
    checks, jobs, clean_pre = [], [], {}
    tolerance = cfg_l3["waveforms"]["verification"]
    for spec in window_specs(events, cfg_a, root):
        keys = {key: spec[key] for key in KEYS}
        source = spec["condition"] if spec["condition"] in ("C0", "C1") else cfg_a["inputs"]["pre_half_of"][spec["condition"]]
        window = physical[(spec["event_id"], source)]
        pre, post = normalize_window(window)
        post_physical = window[1800:]
        if spec["condition"] in ("C0", "C1"):
            clean_pre[(spec["event_id"], source)] = pre
            _verify(spec["pre_path"], pre, keys, checks)
            _verify(spec["post_path"], post, keys, checks)
        else:
            if not np.array_equal(pre, clean_pre[(spec["event_id"], source)]):
                raise BlockedError("Injected pre half differs from clean pre half")
            start = allocation[(spec["event_id"], spec["noise_type"])]
            snippet = development_snippet(noise[spec["noise_type"]], start, cfg_r1)
            injected, numbers = inject(window[:1800], window[1800:], snippet, spec["snr_target"], cfg_r1)
            if injected is None:
                raise BlockedError("Regime 1 injection failed")
            median = float(np.median(window[:1800]))
            iqr = float(np.percentile(window[:1800], 75, method="linear") -
                        np.percentile(window[:1800], 25, method="linear"))
            post = (injected - median) / iqr
            post_physical = injected
            _verify(spec["post_path"], post, keys, checks, snippet_start=start,
                    numbers=numbers, tolerance=tolerance)
        jobs.append((keys, window[:1800], post_physical, pre, post, cfg_s0))
    if len(checks) != 5040 or len(jobs) != 4788:
        raise BlockedError("Unexpected Level 3 Development job count")
    if not all(row["passed"] for row in checks):
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(output / "waveform_regeneration_check.csv", pd.DataFrame(checks))
        raise BlockedError("Waveform regeneration mismatch; see waveform_regeneration_check.csv")
    del physical, noise
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        results = list(pool.map(_feature_job, jobs, chunksize=8))
    results.sort(key=lambda result: (str(result[0]["event_id"]), str(result[0]["condition"]),
                                     str(result[0]["noise_type"]),
                                     -np.inf if result[0]["snr_target"] is None else float(result[0]["snr_target"])))
    frame = pd.DataFrame([r[0] for r in results])
    pre_columns = [c for c in frame if c.endswith("__pre") or c.endswith("_pre_missing")]
    clean_rows = {(r.event_id, r.condition): r for r in frame.loc[frame.condition.isin(("C0", "C1"))].itertuples(index=False)}
    for row in frame.loc[frame.condition.isin(("C2", "C3"))].itertuples(index=False):
        clean = clean_rows[(row.event_id, "C0" if row.condition == "C2" else "C1")]
        for col in pre_columns:
            a, b = getattr(row, col), getattr(clean, col)
            if not (a == b or (pd.isna(a) and pd.isna(b))):
                raise BlockedError(f"Injected pre feature differs from clean: {row.event_id} {col}")
    for target, source in (("C2", "C0"), ("C3", "C1")):
        mask = frame.condition.eq(target)
        clean = frame.loc[frame.condition.eq(source)].set_index("event_id")
        ids = frame.loc[mask, "event_id"]
        for col in pre_columns:
            frame.loc[mask, col] = clean.loc[ids, col].to_numpy()
    failures = pd.DataFrame([f for r in results for f in r[1]],
                            columns=[*KEYS, "half", "group", "feature", "reason"])
    latency = pd.DataFrame([r[2] for r in results])
    summary_rows = []
    feature_cols = [c for c in frame if c not in KEYS]
    for condition, noise_type, snr in [("overall", "all", "all"), *[(c, n, s) for c, n, s in
             frame[["condition", "noise_type", "snr_target"]].fillna({"snr_target": "clean"}).drop_duplicates().itertuples(index=False, name=None)]]:
        part = frame if condition == "overall" else frame.loc[(frame.condition == condition) & (frame.noise_type == noise_type) &
                  (frame.snr_target.fillna("clean") == snr)]
        for col in feature_cols:
            rate = float(part[col].mean()) if col.endswith("_missing") else float(part[col].isna().mean())
            summary_rows.append({"condition": condition, "noise_type": noise_type or "clean", "snr_target": snr,
                                 "feature": col, "nan_rate": rate})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output / "classical_features.parquet", frame)
    atomic_write_parquet(output / "feature_failures.parquet", failures)
    atomic_write_csv(output / "feature_failure_summary.csv", pd.DataFrame(summary_rows))
    atomic_write_csv(output / "first_beat_latency.csv", latency)
    atomic_write_csv(output / "waveform_regeneration_check.csv", pd.DataFrame(checks))
    clean = frame.loc[frame.condition.isin(("C0", "C1"))]
    rates = {name: float(clean[name].mean()) for name in ("rhythm_pre_missing", "template_pre_missing")}
    summary = {"windows": len(frame), "verification_total": len(checks), "verification_passed": len(checks),
               "clean_pre_missing_rates": rates, "stop_rule_threshold": .05,
               "stop_rule_passed": all(rate <= .05 for rate in rates.values()),
               "wall_time_s": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
               "max_worker_rss_mb": max(r[3] for r in results), "code_hash": analysis_code_hash(),
               "config_sha256": {name: hashlib.sha256((root / "configs" / name).read_bytes()).hexdigest()
                                 for name in ("stage1_level3.yaml", "stage1_analysis.yaml", "stage1_regime1.yaml", "stage0.yaml")},
               "classical_features_sha256": hashlib.sha256((output / "classical_features.parquet").read_bytes()).hexdigest()}
    atomic_write_json(output / "extraction_summary.json", summary)
    return summary
