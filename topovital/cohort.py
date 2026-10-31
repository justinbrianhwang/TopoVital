"""Cohort construction: records, subject groups, exclusions, MLII lead, split."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# PhysioNet / WFDB standard beat annotation codes (physiobank annotations page,
# "Beat annotations"). Everything else (e.g. + ~ | ! [ ] x " ) is a non-beat
# (rhythm / signal-quality / comment) annotation and must not be treated as a beat.
BEAT_SYMBOLS = frozenset("N L R B A a J S V r F e j n E / f Q ?".split())
NOISE_SYMBOL = "~"   # signal quality change; subtype = per-signal bitmask, -1 = unreadable (U)
ARTIFACT_SYMBOL = "|"  # isolated QRS-like artifact


class CohortError(AssertionError):
    pass


class BlockedError(RuntimeError):
    """A step cannot run because a protocol-defined setting is still pending."""


def subject_of(record: str, same_subject_groups: list[list[str]]) -> str:
    for grp in same_subject_groups:
        if record in grp:
            return "S" + min(grp)
    return "S" + record


def select_lead(sig_names: list[str], lead_name: str) -> int:
    """Channel index by header name; never assume index 0."""
    hits = [i for i, n in enumerate(sig_names) if n == lead_name]
    if len(hits) != 1:
        raise CohortError(f"lead {lead_name!r} found {len(hits)} times in {sig_names}")
    return hits[0]


def build_manifest(record_ids: list[str], cohort_cfg: dict) -> pd.DataFrame:
    groups = cohort_cfg["same_subject_groups"]
    excl = set(cohort_cfg["excluded_records"])
    df = pd.DataFrame({"record_id": sorted(record_ids)})
    df["subject_id"] = [subject_of(r, groups) for r in df.record_id]
    df["excluded"] = df.record_id.isin(excl)
    df["exclusion_reason"] = np.where(df.excluded, "protocol_excluded_record(102/104/107/217)", "")
    assert_cohort_counts(df, cohort_cfg)
    return df


def assert_cohort_counts(df: pd.DataFrame, cohort_cfg: dict) -> None:
    n_rec, n_sub = len(df), df.subject_id.nunique()
    if (n_rec, n_sub) != (cohort_cfg["expected_records"], cohort_cfg["expected_subjects"]):
        raise CohortError(f"source cohort {n_rec} records/{n_sub} subjects, expected "
                          f"{cohort_cfg['expected_records']}/{cohort_cfg['expected_subjects']}")
    kept = df[~df.excluded]
    exp = cohort_cfg["expected_after_exclusion"]
    if (len(kept), kept.subject_id.nunique()) != (exp["records"], exp["subjects"]):
        raise CohortError(f"after exclusion {len(kept)}/{kept.subject_id.nunique()}, "
                          f"expected {exp['records']}/{exp['subjects']}")


def split_subjects(subject_ids: list[str], split_cfg: dict,
                   subject_pvc_candidates: dict[str, int] | None = None) -> dict[str, str]:
    """Deterministic subject split, with largest-remainder PVC-count strata."""
    if split_cfg.get("seed") is None:
        raise BlockedError("split.seed is pending (v1.3 protocol source not available)")
    strat = split_cfg.get("stratification")
    if strat is None:
        raise BlockedError("split.stratification is pending; set 'none' only if the protocol "
                           "specifies an unstratified split")
    if strat not in ("none", "pvc_candidate_count_3strata"):
        raise BlockedError(f"unsupported stratification {strat!r}")
    subs = sorted(set(subject_ids))
    n_dev, n_test = split_cfg["n_development_subjects"], split_cfg["n_test_subjects"]
    if len(subs) != n_dev + n_test:
        raise CohortError(f"{len(subs)} subjects != {n_dev}+{n_test}")
    rng = np.random.default_rng(int(split_cfg["seed"]))
    if strat == "none":
        perm = rng.permutation(subs)
        return {s: ("development" if k < n_dev else "test") for k, s in enumerate(perm)}
    if subject_pvc_candidates is None or set(subject_pvc_candidates) != set(subs):
        raise BlockedError("stratified split requires a PVC candidate count for every subject")
    counts = {s: int(subject_pvc_candidates[s]) for s in subs}
    if any(v < 0 for v in counts.values()):
        raise BlockedError("PVC candidate counts must be nonnegative")
    positive = [v for v in counts.values() if v >= 1]
    if not positive:
        raise BlockedError("median threshold undefined: no subject has a PVC candidate")
    threshold = float(np.median(positive))
    strata = {"zero": [], "low": [], "high": []}
    for s in subs:
        strata["zero" if counts[s] == 0 else "low" if counts[s] <= threshold else "high"].append(s)
    exact = {k: len(v) * n_dev / len(subs) for k, v in strata.items()}
    quotas = {k: int(np.floor(v)) for k, v in exact.items()}
    order = ("zero", "low", "high")
    for k in sorted(order, key=lambda k: (-(exact[k] - quotas[k]), order.index(k)))[:n_dev-sum(quotas.values())]:
        quotas[k] += 1
    assignment = {}
    for k in order:
        for j, s in enumerate(rng.permutation(strata[k])):
            assignment[str(s)] = "development" if j < quotas[k] else "test"
    return assignment


def inventory_record(rec_path: Path, cohort_cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Header/annotation facts for one record (record-by-record, no bulk loading).

    Returns (header_row, annotation_symbol_counts, noise_annotations).
    """
    import wfdb

    hdr = wfdb.rdheader(str(rec_path))
    rid = rec_path.name
    row = {"record_id": rid, "fs_header": hdr.fs, "sig_len": hdr.sig_len,
           "sig_names": "|".join(hdr.sig_name)}
    try:
        idx = select_lead(hdr.sig_name, cohort_cfg["lead_name"])
    except CohortError:
        idx = None
    row["mlii_channel_index"] = idx
    row["fs_matches_expected"] = hdr.fs == cohort_cfg["fs_expected"]
    if idx is not None:
        row.update(fmt=hdr.fmt[idx], adc_gain=hdr.adc_gain[idx], baseline=hdr.baseline[idx],
                   adc_res=hdr.adc_res[idx], adc_zero=hdr.adc_zero[idx])
        rec = wfdb.rdrecord(str(rec_path), channels=[idx], physical=False)
        d = rec.d_signal[:, 0]
        # Diagnostics only: the protocol's digital saturation bounds are pending.
        lo, hi = 0, 2 ** int(hdr.adc_res[idx]) - 1
        row.update(mlii_digital_min=int(d.min()), mlii_digital_max=int(d.max()),
                   adc_range_from_header=f"[{lo},{hi}]",
                   n_samples_at_adc_min=int((d == lo).sum()), n_samples_at_adc_max=int((d == hi).sum()),
                   n_samples_outside_adc_range=int(((d < lo) | (d > hi)).sum()))
        del rec, d
    ann = wfdb.rdann(str(rec_path), "atr")
    sym = np.asarray(ann.symbol)
    cnt = Counter(sym.tolist())
    row.update(n_annotations=len(sym), n_beat_annotations=int(np.isin(sym, list(BEAT_SYMBOLS)).sum()),
               n_V=cnt.get("V", 0), n_N=cnt.get("N", 0), n_noise_ann=cnt.get(NOISE_SYMBOL, 0),
               n_artifact_ann=cnt.get(ARTIFACT_SYMBOL, 0),
               first_ann_sample=int(ann.sample[0]), last_ann_sample=int(ann.sample[-1]))
    counts = pd.DataFrame([{"record_id": rid, "symbol": s, "is_beat": s in BEAT_SYMBOLS, "count": c}
                           for s, c in sorted(cnt.items())])
    nm = sym == NOISE_SYMBOL
    sub = np.asarray(ann.subtype)[nm]
    noise = pd.DataFrame({"record_id": rid, "sample": np.asarray(ann.sample)[nm], "subtype": sub,
                          "aux_note": np.asarray(ann.aux_note, dtype=object)[nm]})
    if idx is not None:
        noise["mlii_noisy_or_unreadable"] = (sub == -1) | ((sub >= 0) & ((sub >> idx) & 1).astype(bool))
    return row, counts, noise


def pvc_candidates(rec_path: Path, symbol: str = "V") -> pd.DataFrame:
    """Step 1: list of target-beat annotations (absolute sample index r_abs)."""
    import wfdb

    ann = wfdb.rdann(str(rec_path), "atr")
    sym = np.asarray(ann.symbol)
    m = sym == symbol
    return pd.DataFrame({"record_id": rec_path.name, "r_abs": np.asarray(ann.sample)[m].astype(np.int64),
                         "beat_symbol": symbol, "ann_index": np.flatnonzero(m).astype(np.int64)})
