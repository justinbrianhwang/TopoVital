"""Regenerate clean and injected windows with the frozen Stage 1 operations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from topovital.cohort import BlockedError
from topovital_stage1.core import FS, HALF, iter_segments, normalize_window
from topovital_stage1.regime1 import _physical_windows, inject


def noise_channel(rule: str, mlii_channel_index: int) -> int:
    if rule == "fixed_channel_0":
        return 0
    if rule == "match_ecg_signal_index":
        return int(mlii_channel_index)
    raise BlockedError("Test noise channel rule is PENDING or unsupported")


def make_waveforms(events: pd.DataFrame, allocation: pd.DataFrame, settings: dict,
                   regime_cfg: dict, root: Path, *, split: str) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Read only the selected split's NSTDB block, then reuse clean pre halves."""
    root = Path(root)
    if split not in ("development", "test"):
        raise BlockedError("Invalid waveform split")
    channel_rule = settings["noise_channel_rule"] if split == "test" else "fixed_channel_0"
    noise_channel(channel_rule, 0)
    block = (settings["test_noise_allocation"]["block"] if split == "test" else
             regime_cfg["noise"]["development_block"])
    lo, hi = map(int, block)
    length = int(regime_cfg["noise"]["snippet_samples"])
    ecg = root / "data/raw/mit-bih-arrhythmia-database-1.0.0"
    noise_dir = root / "data/raw/mit-bih-noise-stress-test-database-1.0.0"
    clean = {(str(info["event_id"]), info["condition"], info["half"]): half
             for info, half in iter_segments(events, ecg)}
    physical = {(str(ev.event_id), condition): (ev, rid, window)
                for ev, rid, condition, window in _physical_windows(events, ecg)}
    alloc = {(str(r.event_id), r.noise_type): r for r in allocation.itertuples(index=False)}
    snippets = {}
    windows, audit, channel_audit = [], [], []
    for ev in events.itertuples(index=False):
        eid = str(ev.event_id)
        for source in ("C0", "C1"):
            _, rid, window = physical[(eid, source)]
            pre, post = clean[(eid, source, "pre")], clean[(eid, source, "post")]
            if pre is None or post is None:
                raise BlockedError(f"Clean normalization failed: {eid} {source}")
            keys = {"event_id": eid, "subject_id": str(ev.subject_id),
                    "record_id": rid, "condition": source, "noise_type": "", "snr_target": None}
            windows.append({**keys, "pre": pre, "post": post,
                            "physical_pre": window[:HALF], "physical_post": window[HALF:]})
            for half in ("pre", "post"):
                audit.append({**keys, "half": half, "snr_actual": None,
                              "passed": True, "reason": ""})
            target = "C2" if source == "C0" else "C3"
            for noise_type in settings["noise_types"]:
                slot = alloc[(eid, noise_type)]
                start = int(slot.snippet_start)
                channel = noise_channel(channel_rule, int(ev.mlii_channel_index))
                if not lo <= start or start + length > hi:
                    raise BlockedError("NSTDB snippet crosses split block")
                cache_key = (noise_type, start, channel)
                if cache_key not in snippets:
                    record = wfdb.rdrecord(str(noise_dir / noise_type), sampfrom=start,
                                           sampto=start + length, channels=[channel], physical=True)
                    if int(record.fs) != FS or record.p_signal.shape != (length, 1):
                        raise BlockedError("Invalid NSTDB snippet shape or rate")
                    snippets[cache_key] = np.asarray(record.p_signal[:, 0], dtype=float)
                if split == "test" and int(ev.mlii_channel_index) != 0 and source == "C0":
                    other_channel = 1 - channel
                    other_key = (noise_type, start, other_channel)
                    if other_key not in snippets:
                        other = wfdb.rdrecord(str(noise_dir / noise_type), sampfrom=start,
                                              sampto=start + length, channels=[other_channel],
                                              physical=True)
                        if int(other.fs) != FS or other.p_signal.shape != (length, 1):
                            raise BlockedError("Invalid NSTDB audit snippet shape or rate")
                        snippets[other_key] = np.asarray(other.p_signal[:, 0], dtype=float)
                    first, second = snippets[(noise_type, start, 0)], snippets[(noise_type, start, 1)]
                    channel_audit.append({"record_id": str(ev.record_id), "noise_type": noise_type,
                                          "snippet_id": slot.snippet_id, "channel_used": channel,
                                          "channel_0_rms": float(np.sqrt(np.mean(first ** 2))),
                                          "channel_0_std": float(np.std(first)),
                                          "channel_1_rms": float(np.sqrt(np.mean(second ** 2))),
                                          "channel_1_std": float(np.std(second)),
                                          "pearson_correlation": float(np.corrcoef(first, second)[0, 1])})
                snippet = snippets[cache_key]
                for snr in settings["snr_db"]:
                    injected, numbers = inject(window[:HALF], window[HALF:], snippet, snr, regime_cfg)
                    key = {**keys, "condition": target, "noise_type": noise_type, "snr_target": snr,
                           "snippet_id": slot.snippet_id, "snippet_start": start,
                           "channel": channel, "reuse_count": int(slot.reuse_count)}
                    audit.append({**key, "half": "pre", "snr_actual": None,
                                  "passed": True, "reason": "reused_clean_pre"})
                    passed = injected is not None and abs(numbers["snr_actual"] - snr) < 1e-3
                    audit.append({**key, "half": "post", **numbers, "passed": bool(passed),
                                  "reason": "" if passed else numbers["injection_failure_reason"]})
                    if not passed:
                        windows.append({**key, "pre": pre, "post": None,
                                        "physical_pre": window[:HALF], "physical_post": None,
                                        "failure_reason": numbers["injection_failure_reason"]})
                        continue
                    median = float(np.median(window[:HALF]))
                    iqr = float(np.percentile(window[:HALF], 75, method="linear") -
                                np.percentile(window[:HALF], 25, method="linear"))
                    if iqr <= 0:
                        raise BlockedError("Invalid pre IQR for injected waveform")
                    windows.append({**key, "pre": pre, "post": (injected - median) / iqr,
                                    "physical_pre": window[:HALF], "physical_post": injected,
                                    "snr_actual": numbers["snr_actual"], "scale_a": numbers["scale_a"]})
    return windows, pd.DataFrame(audit), pd.DataFrame(channel_audit, columns=[
        "record_id", "noise_type", "snippet_id", "channel_used", "channel_0_rms",
        "channel_0_std", "channel_1_rms", "channel_1_std", "pearson_correlation"])
