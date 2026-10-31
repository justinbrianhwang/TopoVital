"""Record-at-a-time Stage 0 eligibility, pairing, and Development g selection."""
from __future__ import annotations

import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BEAT_SYMBOLS, BlockedError
from topovital.matching import match_normals
from topovital.quality import bsqi_f1, max_run_at_bounds
from topovital.template import template_eligibility
from topovital.timeaxis import event_window, ms_to_samples
from topovital.util import (atomic_write_bytes, atomic_write_csv, atomic_write_json,
                            atomic_write_parquet, atomic_write_yaml, code_hash,
                            config_hash, sha256_file)

GATES = [
    ("within_record", "pass_within_record"),
    ("pre_beats_not_all_N", "pass_pre_all_beats_N"),
    ("pre_saturation", "pass_pre_saturation"),
    ("pre_iqr", "pass_iqr"),
    ("pre_bsqi", "pass_bsqi"),
    ("post_not_single_isolated_V", "pass_post_single_isolated_pvc"),
    ("post_saturation", "pass_post_saturation"),
    ("post_signal_loss", "pass_post_no_quality_annotation"),
    ("overlap", "pass_no_overlap"),
    ("no_admissible_normal_window", "in_common_cohort"),
]


def signal_loss_intervals(samples, symbols, subtypes, sig_len):
    """A U state starts at '~' subtype -1 and ends at the next '~'."""
    markers = [(int(s), int(t)) for s, sym, t in zip(samples, symbols, subtypes) if sym == "~"]
    return [(s, markers[i + 1][0] if i + 1 < len(markers) else sig_len)
            for i, (s, t) in enumerate(markers) if t == -1]


def intersects_loss(intervals, start, end):
    return any(start < stop and s < end for s, stop in intervals)


def beat_slice(samples, symbols, start, end):
    a, b = np.searchsorted(samples, [start, end])
    r, sym = samples[a:b], symbols[a:b]
    mask = np.isin(sym, list(BEAT_SYMBOLS))
    return r[mask], sym[mask]


def median_rr(samples, symbols, start, end, fs):
    beats, labels = beat_slice(samples, symbols, start, end)
    normal = beats[labels == "N"]
    return float(np.median(np.diff(normal)) / fs) if len(normal) >= 2 else math.nan


def post_is_single_v(samples, symbols, boundary, end, target):
    beats, labels = beat_slice(samples, symbols, boundary, end)
    return bool((labels == "V").sum() == 1 and (beats[labels == "V"] == target).all()
                and np.isin(labels, ["N", "V"]).all())


def retained_after_overlap(events, fs, min_separation_s):
    """Keep the earlier eligible event in each record at every conflict."""
    kept = set()
    last_end = {}
    for ev in events.sort_values(["record_id", "start_abs", "event_id"], kind="stable").itertuples():
        if ev.start_abs >= last_end.get(ev.record_id, -math.inf) + min_separation_s * fs:
            kept.add(ev.event_id)
            last_end[ev.record_id] = ev.end_abs
    return kept


def bsqi(signal, fs, cfg):
    """Both detectors receive the same zero-phase, fourth-order SOS bandpass."""
    from scipy.signal import butter, sosfiltfilt
    from wfdb.processing import xqrs_detect
    import neurokit2 as nk

    if cfg["filter_phase"] != "zero_phase_sosfiltfilt" or cfg["detectors"] != [
        "wfdb.processing.xqrs_detect", "neurokit2.ecg_peaks(method='pantompkins1985')"
    ]:
        raise BlockedError("unsupported configured bSQI filter phase or detectors")
    order = re.search(r"order\s+(\d+)", cfg["filter"])
    if order is None or "SOS" not in cfg["filter"]:
        raise BlockedError("unsupported configured bSQI filter")
    errs = []
    try:
        sos = butter(int(order.group(1)), cfg["bandpass_hz"], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, signal)
    except Exception as exc:
        return 0.0, f"bandpass: {type(exc).__name__}: {exc}"
    try:
        first = xqrs_detect(filtered, fs, verbose=False)
    except Exception as exc:
        first = []
        errs.append(f"xqrs: {type(exc).__name__}: {exc}")
    try:
        _, info = nk.ecg_peaks(filtered, sampling_rate=fs, method="pantompkins1985")
        second = info["ECG_R_Peaks"]
    except Exception as exc:
        second = []
        errs.append(f"pantompkins1985: {type(exc).__name__}: {exc}")
    return bsqi_f1(first, second, ms_to_samples(cfg["match_tolerance_ms"], fs)), "; ".join(errs)


def _read_record(path, idx):
    import wfdb
    rec = wfdb.rdrecord(str(path), channels=[idx], physical=False)
    ann = wfdb.rdann(str(path), "atr")
    digital = rec.d_signal[:, 0]
    physical = (digital - rec.baseline[0]) / rec.adc_gain[0]
    return digital, physical, np.asarray(ann.sample), np.asarray(ann.symbol), np.asarray(ann.subtype)


def _quality_gate(d, p, start, boundary, end, samples, symbols, losses, cfg, target=None):
    """Return first failed quality gate and measured values for a window."""
    qc, sat = cfg["quality"], cfg["quality"]["saturation"]
    row = {}
    pre_beats, pre_symbols = beat_slice(samples, symbols, start, boundary)
    row["pass_pre_all_beats_N"] = bool((pre_symbols == "N").all())
    if not row["pass_pre_all_beats_N"]:
        return "pre_beats_not_all_N", row
    row["pre_saturation_max_run"] = max_run_at_bounds(d[start:boundary], sat["digital_low"], sat["digital_high"])
    row["pass_pre_saturation"] = row["pre_saturation_max_run"] < sat["consecutive_samples"]
    if not row["pass_pre_saturation"]:
        return "pre_saturation", row
    row["iqr_pre_mV"] = float(np.percentile(p[start:boundary], 75, method=qc["iqr"]["percentile_method"])
                               - np.percentile(p[start:boundary], 25, method=qc["iqr"]["percentile_method"]))
    row["iqr_value"] = row["iqr_pre_mV"]
    row["pass_iqr"] = row["iqr_pre_mV"] >= qc["iqr"]["threshold"]
    if not row["pass_iqr"]:
        return "pre_iqr", row
    row["bsqi_pre"], row["bsqi_detector_error"] = bsqi(p[start:boundary], cfg["cohort"]["fs_expected"], qc["bsqi"])
    row["bsqi_f1"] = row["bsqi_pre"]
    row["pass_bsqi"] = row["bsqi_pre"] >= qc["bsqi"]["threshold"]
    if not row["pass_bsqi"]:
        return "pre_bsqi", row
    post_beats, post_symbols = beat_slice(samples, symbols, boundary, end)
    if target is None:
        row["pass_post_single_isolated_pvc"] = bool((post_symbols == "N").all())
    else:
        row["pass_post_single_isolated_pvc"] = post_is_single_v(samples, symbols, boundary, end, target)
    row["pass_post_no_other_abnormal"] = row["pass_post_single_isolated_pvc"]
    if not row["pass_post_single_isolated_pvc"]:
        return "post_not_single_isolated_V", row
    row["post_saturation_max_run"] = max_run_at_bounds(d[boundary:end], sat["digital_low"], sat["digital_high"])
    row["pass_post_saturation"] = row["post_saturation_max_run"] < sat["consecutive_samples"]
    if not row["pass_post_saturation"]:
        return "post_saturation", row
    row["pass_post_no_quality_annotation"] = not intersects_loss(losses, start if target is None else boundary, end)
    if not row["pass_post_no_quality_annotation"]:
        return "post_signal_loss", row
    return None, row


def _normal_candidates(rid, sid, d, p, samples, symbols, losses, cfg):
    fs = cfg["cohort"]["fs_expected"]
    half = ms_to_samples(cfg["time"]["half_window_s"] * 1000, fs)
    width = ms_to_samples(cfg["time"]["window_s"] * 1000, fs)
    grid = ms_to_samples(cfg["normal_matching"]["candidate_grid_step_s"] * 1000, fs)
    rows, failures = [], []
    for start in range(0, len(d) - width + 1, grid):
        end, boundary = start + width, start + half
        reason, measured = _quality_gate(d, p, start, boundary, end, samples, symbols, losses, cfg)
        if measured.get("bsqi_detector_error"):
            failures.append(dict(record_id=rid, subject_id=sid, start_abs=start,
                                 reason="bsqi_detector_error", message=measured["bsqi_detector_error"]))
        if reason is None:
            rr = median_rr(samples, symbols, start, boundary, fs)
            if math.isfinite(rr):
                rows.append(dict(seg_id=f"{rid}_{start}", subject_id=sid, record_id=rid,
                                 start_abs=start, end_abs=end, pre_median_rr_s=rr))
    return rows, failures


def _delineate(p, samples, symbols, r_abs, fs):
    import neurokit2 as nk
    start, end = r_abs - 5 * fs, r_abs + 5 * fs
    if start < 0 or end > len(p):
        return None, None, "context_out_of_record"
    peaks, _ = beat_slice(samples, symbols, start, end)
    rel = peaks - start
    try:
        _, waves = nk.ecg_delineate(p[start:end], rpeaks=rel, sampling_rate=fs, method="dwt")
        ix = np.flatnonzero(peaks == r_abs)
        if len(ix) != 1:
            return None, None, "target_annotation_missing"
        onset = waves["ECG_R_Onsets"][int(ix[0])]
        if onset is None or not np.isfinite(onset):
            return None, None, "onset_missing"
        absolute = start + int(onset)
        return absolute, 1000 * (r_abs - absolute) / fs, "valid"
    except Exception as exc:
        return None, None, f"delineation_error: {type(exc).__name__}: {exc}"


def _figures(cfg, out, audit, onset_by_event):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = cfg["cohort"]["fs_expected"]
    common = audit[audit.in_common_cohort == True].drop_duplicates("event_id")
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    sample_ids = rng.choice(common.event_id.to_numpy(), size=min(30, len(common)), replace=False).tolist()
    extra = [eid for eid, (_, delta, status) in onset_by_event.items() if status == "valid" and delta > 200][:30]
    chosen = list(dict.fromkeys(sample_ids + extra))
    index = []
    by_record = {rid: group for rid, group in common[common.event_id.isin(chosen)].groupby("record_id")}
    panels = {}
    mit = Path(cfg["data"]["extract_dir"])
    if not mit.is_absolute():
        from topovital.util import resolve
        mit = resolve(mit)
    mit = mit / cfg["data"]["mitdb"]["root"]
    for rid, group in by_record.items():
        idx = int(group.mlii_channel_index.iloc[0])
        _, p, _, _, _ = _read_record(mit / rid, idx)
        for row in group.itertuples():
            w = event_window(int(row.r_abs), max(cfg["time"]["g_candidates_ms"]), fs,
                             cfg["time"]["half_window_s"], cfg["time"]["window_s"])
            if not w.within_record(len(p)):
                continue
            panels[row.event_id] = (p[w.start_abs:w.end_abs], w, row)
    for page, offset in enumerate(range(0, len(chosen), 6), 1):
        fig, axes = plt.subplots(6, 1, figsize=(12, 15))
        for ax, eid in zip(axes, chosen[offset:offset + 6]):
            if eid not in panels:
                ax.set_visible(False)
                index.append(dict(event_id=eid, record_id="", page=f"page_{page:02d}.png",
                                  status="largest_window_out_of_record"))
                continue
            y, w, row = panels[eid]
            ax.plot(np.arange(len(y)) / fs, y, lw=.5)
            ax.axvline((int(row.r_abs)-w.start_abs)/fs, color="red", label="R")
            onset = onset_by_event.get(eid, (None, None, "missing"))[0]
            if onset is not None:
                ax.axvline((onset-w.start_abs)/fs, color="green", label="onset")
            for g in cfg["time"]["g_candidates_ms"]:
                ax.axvline((int(row.r_abs)-ms_to_samples(g, fs)-w.start_abs)/fs, ls="--", label=f"{g} ms")
            ax.set_title(eid)
            ax.legend(loc="upper right", ncol=5, fontsize=6)
            index.append(dict(event_id=eid, record_id=row.record_id, page=f"page_{page:02d}.png",
                              status="plotted"))
        for ax in axes[len(chosen[offset:offset + 6]):]:
            ax.set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        atomic_write_bytes(out / "waveform_check" / f"page_{page:02d}.png", buf.getvalue())
    atomic_write_csv(out / "waveform_check" / "index.csv", pd.DataFrame(index, columns=["event_id", "record_id", "page", "status"]))


def run_pipeline(cfg, out: Path, run_id: str, split: str, g_values):
    from topovital.stage0 import AUDIT_SCHEMA
    manifest_path, step1_path = out / "cohort_manifest.csv", out / "event_audit.parquet"
    if not manifest_path.exists() or not step1_path.exists() or not (out / "split_assignment.csv").exists():
        raise BlockedError("run cohort and split before Stage 0")
    man = pd.read_csv(manifest_path, dtype={"record_id": str, "subject_id": str})
    selected = man[(man.split == split) & man.in_stage0_cohort.astype(str).str.lower().eq("true")]
    expected = cfg["split"]["n_development_subjects" if split == "development" else "n_test_subjects"]
    if selected.subject_id.nunique() != expected:
        raise BlockedError(f"{split} split has {selected.subject_id.nunique()} subjects, expected {expected}")
    step1 = pd.read_parquet(step1_path)
    step1 = step1[step1.subject_id.isin(selected.subject_id)].copy()  # before any signal read
    subjects_with_candidates = set(step1.subject_id)
    fs = cfg["cohort"]["fs_expected"]
    cfg_hash, src_hash = config_hash(cfg), code_hash()
    rows, normal_rows, normal_failures, candidate_cache = [], [], [], {}
    from topovital.util import resolve
    mit = resolve(cfg["data"]["extract_dir"]) / cfg["data"]["mitdb"]["root"]
    for rec in selected.itertuples():
        rid, sid = rec.record_id, rec.subject_id
        if sid not in subjects_with_candidates:
            continue
        d, p, samples, symbols, subtypes = _read_record(mit / rid, int(rec.mlii_channel_index))
        losses = signal_loss_intervals(samples, symbols, subtypes, len(d))
        rec_events = step1[step1.record_id == rid]
        for g in g_values:
            for ev in rec_events.itertuples():
                w = event_window(int(ev.r_abs), g, fs, cfg["time"]["half_window_s"], cfg["time"]["window_s"])
                row = {k: getattr(ev, k) for k in ("event_id", "subject_id", "record_id", "r_abs", "ann_index", "beat_symbol", "fs", "mlii_channel_index")}
                row.update(split=split, g_ms=g, g_samples=w.g_samples, realized_g_ms=w.realized_g_ms,
                           boundary_abs=w.boundary_abs, start_abs=w.start_abs, end_abs=w.end_abs,
                           r_rel_samples=w.r_abs-w.start_abs, boundary_rel_samples=w.boundary_abs-w.start_abs,
                           pass_step1_candidate=True, pass_within_record=w.within_record(len(d)),
                           config_hash=cfg_hash, code_hash=src_hash, run_id=run_id)
                if not row["pass_within_record"]:
                    row["primary_drop_reason"] = "within_record"
                else:
                    reason, measured = _quality_gate(d, p, w.start_abs, w.boundary_abs, w.end_abs,
                                                     samples, symbols, losses, cfg, int(ev.r_abs))
                    row.update(measured)
                    row["primary_drop_reason"] = reason
                    if reason is None:
                        row["pvc_pre_median_rr_s"] = median_rr(samples, symbols, w.start_abs, w.boundary_abs, fs)
                rows.append(row)
        # One evaluation per record; aliases include g and both hashes.
        key = (rid, cfg_hash, src_hash)
        candidate_cache[key], failures = _normal_candidates(rid, sid, d, p, samples, symbols, losses, cfg)
        normal_failures.extend(failures)
        for g in g_values:
            candidate_cache[(rid, g, cfg_hash, src_hash)] = candidate_cache[key]
        normal_rows.extend(candidate_cache[key])
    audit = pd.DataFrame(rows)
    if audit.empty:
        audit = pd.DataFrame(columns=list(AUDIT_SCHEMA) + ["primary_drop_reason"])
    normals = pd.DataFrame(normal_rows, columns=["seg_id", "subject_id", "record_id", "start_abs", "end_abs", "pre_median_rr_s"])
    for g in g_values:
        eligible = audit[(audit.g_ms == g) & audit.primary_drop_reason.isna()]
        retained = retained_after_overlap(eligible, fs, cfg["events"]["min_separation_s"])
        for ix, ev in eligible.iterrows():
            keep = ev.event_id in retained
            audit.at[ix, "pass_no_overlap"] = bool(keep)
            if not keep:
                audit.at[ix, "primary_drop_reason"] = "overlap"
        pairable = audit[(audit.g_ms == g) & audit.primary_drop_reason.isna()]
        if len(pairable):
            pairs = match_normals(pairable.rename(columns={"pvc_pre_median_rr_s": "pre_median_rr_s"}),
                                  normals, cfg["normal_matching"]["rr_tolerance_frac"],
                                  fs=fs,
                                  pvc_lookback_s=cfg["normal_matching"]["lookback_s"],
                                  normal_lookback_s=cfg["normal_matching"]["normal_lookback_s"])
            pairs = pairs.set_index("event_id")
            for ix, ev in pairable.iterrows():
                m = pairs.loc[ev.event_id]
                audit.at[ix, "normal_match_status"] = m.match_status
                if m.match_status == "matched":
                    audit.at[ix, "normal_segment_id"] = m.seg_id
                    for col in ("normal_start_sample", "normal_record_id", "normal_pre_median_rr_s"):
                        audit.at[ix, col] = m[col]
                    audit.at[ix, "in_common_cohort"] = True
                else:
                    audit.at[ix, "in_common_cohort"] = False
                    audit.at[ix, "primary_drop_reason"] = "no_admissible_normal_window"
    audit["in_common_cohort"] = audit.in_common_cohort.eq(True).astype(bool)
    audit["in_regime2"] = False
    for rid, group in audit[audit.in_common_cohort].groupby("record_id"):
        idx = int(group.mlii_channel_index.iloc[0])
        _, _, samples, symbols, _ = _read_record(mit / rid, idx)
        for ix, ev in group.iterrows():
            t = template_eligibility(samples, symbols, int(ev.boundary_abs), fs,
                                     lookback_s=cfg["template"]["lookback_s"],
                                     min_beats=cfg["template"]["min_normal_beats"],
                                     win_ms=cfg["template"]["extract_window_ms"],
                                     normal_symbols=cfg["template"]["normal_beat_symbols"])
            audit.at[ix, "template_status"] = t["template_status"]
            audit.at[ix, "n_template_beats"] = t["n_template_beats"]
            audit.at[ix, "template_eligible"] = t["template_status"] == "eligible"
            audit.at[ix, "in_regime2"] = t["template_status"] == "eligible"
    # Full-window bSQI is a sensitivity count only.
    for rid, group in audit[audit.in_common_cohort].groupby("record_id"):
        _, p, _, _, _ = _read_record(mit / rid, int(group.mlii_channel_index.iloc[0]))
        for ix, ev in group.iterrows():
            score, err = bsqi(p[int(ev.start_abs):int(ev.end_abs)], fs, cfg["quality"]["bsqi"])
            audit.at[ix, "bsqi_full_window"] = score
            if err:
                prev = audit.at[ix, "bsqi_detector_error"]
                audit.at[ix, "bsqi_detector_error"] = (str(prev) + "; " if isinstance(prev, str) and prev else "") + "full_window: " + err
    onset_by_event = {}
    if split == "development":
        for rid, group in audit[audit.in_common_cohort].drop_duplicates("event_id").groupby("record_id"):
            _, p, samples, symbols, _ = _read_record(mit / rid, int(group.mlii_channel_index.iloc[0]))
            for ev in group.itertuples():
                onset_by_event[ev.event_id] = _delineate(p, samples, symbols, int(ev.r_abs), fs)
        for ix, ev in audit.iterrows():
            if ev.event_id in onset_by_event:
                onset, delta, status = onset_by_event[ev.event_id]
                audit.at[ix, "qrs_onset_abs"] = onset
                audit.at[ix, "r_minus_onset_ms"] = delta
                audit.at[ix, "onset_status"] = status
    audit["first_failed_step"] = audit.primary_drop_reason
    audit["stage0_status"] = np.where(audit.in_common_cohort, "common", "excluded")
    for col, dtype in AUDIT_SCHEMA.items():
        if col not in audit:
            audit[col] = pd.NA
        audit[col] = audit[col].astype(dtype)
    audit = audit[list(AUDIT_SCHEMA) + [c for c in audit if c not in AUDIT_SCHEMA]]
    folder = out / split
    suffix = "dev" if split == "development" else "test"
    atomic_write_csv(folder / f"normal_candidate_failures_{suffix}.csv",
                     pd.DataFrame([{**failure, "config_hash": cfg_hash, "code_hash": src_hash, "run_id": run_id}
                                   for failure in normal_failures],
                                  columns=["record_id", "subject_id", "start_abs", "reason", "message",
                                           "config_hash", "code_hash", "run_id"]))
    audit_path = folder / f"event_audit_{suffix}.parquet"
    atomic_write_parquet(audit_path, audit)
    attr_rows, count_rows, g_rows = [], [], []
    for g in g_values:
        a = audit[audit.g_ms == g]
        for sid in selected.subject_id.unique():
            sub = a[a.subject_id == sid]
            count_rows.append(dict(g_ms=g, subject_id=sid, E_s_common=int(sub.in_common_cohort.sum()),
                                   E_s_regime2=int(sub.in_regime2.sum()), split=split,
                                   config_hash=cfg_hash, code_hash=src_hash, run_id=run_id))
        attr_rows.append(dict(g_ms=g, step="step1_candidates", branch="common", split=split,
                              n_events=len(a), n_subjects=a.subject_id.nunique(), n_records=a.record_id.nunique(),
                              S_eff=a.subject_id.nunique(), status="executed", config_hash=cfg_hash,
                              code_hash=src_hash, run_id=run_id))
        for step, _ in GATES:
            rank = [name for name, _ in GATES].index(step)
            survivors = a[a.primary_drop_reason.isna() | ~a.primary_drop_reason.isin([x for x, _ in GATES[:rank+1]])]
            attr_rows.append(dict(g_ms=g, step=step, branch="common", split=split, n_events=len(survivors),
                                  n_subjects=survivors.subject_id.nunique(), n_records=survivors.record_id.nunique(),
                                  S_eff=survivors.subject_id.nunique(), status="executed", config_hash=cfg_hash,
                                  code_hash=src_hash, run_id=run_id))
        r2 = a[a.in_regime2]
        attr_rows.append(dict(g_ms=g, step="template_eligible", branch="regime2", split=split,
                              n_events=len(r2), n_subjects=r2.subject_id.nunique(), n_records=r2.record_id.nunique(),
                              S_eff=r2.subject_id.nunique(), status="executed", config_hash=cfg_hash,
                              code_hash=src_hash, run_id=run_id))
        common = a[a.in_common_cohort]
        excluded = common[common.bsqi_full_window.astype(float) < cfg["quality"]["bsqi"]["threshold"]]
        valid = common[common.onset_status == "valid"] if "onset_status" in common else common.iloc[0:0]
        covered = valid[valid.r_minus_onset_ms.astype(float) <= 1000 * ms_to_samples(g, fs) / fs]
        coverage = len(covered) / len(valid) if len(valid) else None
        g_rows.append(dict(g_ms=g, common_events=len(common), n_valid_onsets=len(valid),
                           n_onset_failures=len(common)-len(valid), coverage=coverage,
                           sensitivity_excluded_events=len(excluded),
                           sensitivity_excluded_subjects=excluded.subject_id.nunique(),
                           config_hash=cfg_hash, code_hash=src_hash, run_id=run_id))
    attr = pd.DataFrame(attr_rows)
    atomic_write_csv(folder / f"attrition_{suffix}.csv", attr)
    atomic_write_csv(folder / f"per_subject_counts_{suffix}.csv", pd.DataFrame(count_rows))
    if split == "development":
        summary = pd.DataFrame(g_rows)
        atomic_write_csv(folder / "g_selection_summary.csv", summary)
        passing = summary[(summary.coverage.notna()) & (summary.coverage >= cfg["g_selection"]["coverage_threshold"])]
        rule = int(passing.g_ms.min()) if len(passing) else int(cfg["g_selection"]["fallback_g_ms"])
        atomic_write_json(folder / "g_rule_result.json", dict(rule_g_star=rule, coverage_threshold=cfg["g_selection"]["coverage_threshold"],
                                                           per_g=g_rows, config_hash=cfg_hash,
                                                           code_hash=src_hash, run_id=run_id))
        draft_path = resolve("configs/g_decision.draft.yaml") if out.resolve() == resolve("outputs").resolve() else out / "g_decision.draft.yaml"
        atomic_write_yaml(draft_path, dict(g_star=rule, waveform_inspection_performed=None,
                                           dev_audit_sha256=sha256_file(audit_path),
                                           config_hash=cfg_hash, code_hash=src_hash, run_id=run_id))
        _figures(cfg, folder, audit, onset_by_event)
    main_attr = out / "attrition_summary.csv"
    if main_attr.exists():
        old = pd.read_csv(main_attr)
        old = old[old.split != split]
        atomic_write_csv(main_attr, pd.concat([old, attr], ignore_index=True))
