"""Stage 0 orchestration on real data.

Implemented and executable now (protocol-independent):
  * cohort assertions 48/47 -> 44/43, subject grouping, MLII-by-name lead check,
    360 Hz check, ADC range diagnostics, annotation / signal-quality inventory,
    NSTDB noise record inventory,
  * step 1 of the event pipeline (target 'V' candidate list), g-independent.

Blocked until the protocol settings are filled in (see gate.pending_settings):
  * Development/Test split, steps 2-6, the g sweep, freeze and Test aggregation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from topovital import cohort as C
from topovital.gate import pending_settings, require_dev_sweep_allowed
from topovital.util import (atomic_write_csv, atomic_write_json, atomic_write_parquet,
                            atomic_write_yaml, code_hash, config_hash, resolve, sha256_bytes, sha256_file)

PENDING_STATUS = "pending:protocol_settings_missing"

# Columns of event_audit.parquet. Step-2..6 columns exist from the start so the schema
# is stable; they stay null until the corresponding step actually runs.
AUDIT_SCHEMA = {
    "event_id": "string", "subject_id": "string", "record_id": "string", "split": "string",
    "r_abs": "Int64", "ann_index": "Int64", "beat_symbol": "string", "fs": "Int64",
    "mlii_channel_index": "Int64",
    "g_ms": "Float64", "g_samples": "Int64", "realized_g_ms": "Float64",
    "boundary_abs": "Int64", "start_abs": "Int64", "end_abs": "Int64",
    "r_rel_samples": "Int64", "boundary_rel_samples": "Int64",
    "pass_step1_candidate": "boolean", "pass_within_record": "boolean",
    "pass_pre_all_beats_N": "boolean", "pre_saturation_max_run": "Int64", "pass_pre_saturation": "boolean",
    "iqr_value": "Float64", "pass_iqr": "boolean", "bsqi_f1": "Float64", "pass_bsqi": "boolean",
    "pass_post_single_isolated_pvc": "boolean", "pass_post_no_other_abnormal": "boolean",
    "pass_post_no_quality_annotation": "boolean", "post_saturation_max_run": "Int64",
    "pass_post_saturation": "boolean", "pass_no_overlap": "boolean",
    "normal_segment_id": "string", "normal_match_status": "string",
    "in_common_cohort": "boolean", "template_status": "string", "n_template_beats": "Int64",
    "in_regime2": "boolean", "first_failed_step": "string", "stage0_status": "string",
    "config_hash": "string", "code_hash": "string", "run_id": "string",
}


def _paths(cfg):
    root = resolve(cfg["data"]["extract_dir"])
    return root / cfg["data"]["mitdb"]["root"], root / cfg["data"]["nstdb"]["root"]


def _input_key(cfg: dict, out: Path) -> str:
    chk = out / "input_checksums.csv"
    if not chk.exists():
        raise C.BlockedError("input_checksums.csv missing; run `python -m topovital fetch` first.")
    return sha256_bytes((sha256_file(chk) + config_hash(cfg) + code_hash()).encode())


def run_cohort(cfg: dict, out: Path, run_id: str, rec: dict) -> dict:
    """Cohort manifest + step-1 event audit + attrition. Reuses verified cache."""
    key = _input_key(cfg, out)
    stamp = out / ".cache" / "cohort_stamp.json"
    products = ["cohort_manifest.csv", "event_audit.parquet", "attrition_summary.csv",
                "record_inventory.csv", "annotation_symbol_counts.csv",
                "signal_quality_annotations.csv", "nstdb_noise_inventory.csv"]
    if stamp.exists():
        st = json.loads(stamp.read_text(encoding="utf-8"))
        if st.get("key") == key and all((out / p).exists() and sha256_file(out / p) == st["sha256"].get(p)
                                        for p in products):
            rec["cache_used"] = True
            rec["note"] = "verified cache reuse (same inputs/config/code)"
            return st["summary"]

    mit, nst = _paths(cfg)
    ccfg = cfg["cohort"]
    records = (mit / "RECORDS").read_text().split()
    manifest = C.build_manifest(records, ccfg)  # asserts 48/47 -> 44/43

    inv_rows, sym_tabs, noise_tabs, cand_tabs = [], [], [], []
    for rid in manifest.record_id:  # record-by-record; nothing kept but small tables
        row, sym, noise = C.inventory_record(mit / rid, ccfg)
        inv_rows.append(row); sym_tabs.append(sym); noise_tabs.append(noise)
        cand_tabs.append(C.pvc_candidates(mit / rid, cfg["events"]["target_beat_symbol"]))
    inv = pd.DataFrame(inv_rows)

    # Assertions on real data
    assert (inv.fs_header == ccfg["fs_expected"]).all(), "fs mismatch with 360 Hz"
    man = manifest.merge(inv[["record_id", "fs_header", "sig_len", "sig_names", "mlii_channel_index", "n_V"]],
                         on="record_id", how="left")
    man["in_stage0_cohort"] = ~man.excluded
    missing_lead = man[man.in_stage0_cohort & man.mlii_channel_index.isna()]
    if len(missing_lead):
        raise C.CohortError(f"cohort records without MLII: {missing_lead.record_id.tolist()}")
    man["mlii_status"] = np.where(man.mlii_channel_index.isna(), "MLII_absent", "MLII_found")
    man["mlii_channel_index"] = man.mlii_channel_index.astype("Int64")
    man["mlii_ch_idx"] = man.mlii_channel_index
    man["included"] = man.in_stage0_cohort
    man = man.rename(columns={"n_V": "n_pvc_candidates_step1"})
    for col in ("split", "split_group", "n_valid_events_common", "n_valid_events_regime2"):
        man[col] = pd.NA
    man["split_status"] = np.where(man.in_stage0_cohort, "pending:split.seed/stratification", "not_applicable")
    man["split_seed"] = cfg["split"]["seed"]
    man["split_ordering"] = cfg["split"]["ordering"]
    man["config_hash"] = config_hash(cfg)
    man["code_hash"] = code_hash()
    man["run_id"] = run_id

    # NSTDB raw noise records only
    import wfdb
    nrows = []
    for r in cfg["data"]["nstdb"]["noise_records"]:
        h = wfdb.rdheader(str(nst / r))
        nrows.append({"noise_record": r, "fs": h.fs, "sig_len": h.sig_len, "sig_names": "|".join(h.sig_name),
                      "fmt": "|".join(h.fmt), "adc_gain": "|".join(map(str, h.adc_gain)),
                      "duration_s": h.sig_len / h.fs, "fs_matches_expected": h.fs == ccfg["fs_expected"],
                      "dev_block": None, "test_block": None, "block_status": "pending:noise blocks"})
    ninv = pd.DataFrame(nrows)

    # Step 1 event audit (g-independent; all later columns stay null)
    cands = pd.concat(cand_tabs, ignore_index=True)
    cands = cands.merge(man[["record_id", "subject_id", "in_stage0_cohort", "mlii_channel_index"]], on="record_id")
    cands = cands[cands.in_stage0_cohort].drop(columns="in_stage0_cohort")
    cands = cands.sort_values(["record_id", "r_abs"], kind="stable").reset_index(drop=True)
    audit = pd.DataFrame({c: pd.Series([pd.NA] * len(cands), dtype=t) for c, t in AUDIT_SCHEMA.items()})
    audit["event_id"] = (cands.record_id + "_" + cands.r_abs.astype(str)).astype("string")
    for c in ("subject_id", "record_id", "beat_symbol"):
        audit[c] = cands[c].astype("string")
    for c in ("r_abs", "ann_index", "mlii_channel_index"):
        audit[c] = cands[c].astype("Int64")
    audit["fs"] = pd.Series([ccfg["fs_expected"]] * len(cands), dtype="Int64")
    audit["pass_step1_candidate"] = pd.Series([True] * len(cands), dtype="boolean")
    audit["stage0_status"] = PENDING_STATUS
    audit["config_hash"] = config_hash(cfg)
    audit["code_hash"] = code_hash()
    audit["run_id"] = run_id
    audit = audit.astype(AUDIT_SCHEMA)
    event_list_sha = sha256_bytes(audit[["event_id", "subject_id", "record_id", "r_abs"]]
                                  .to_csv(index=False, lineterminator="\n").encode())

    kept = man[man.in_stage0_cohort]
    attr = [
        dict(step="0_source_records", branch="cohort", split="all", g_ms=None, n_records=len(man),
             n_subjects=man.subject_id.nunique(), n_events=None, status="executed", note="MIT-BIH RECORDS file"),
        dict(step="0_after_record_exclusion", branch="cohort", split="all", g_ms=None, n_records=len(kept),
             n_subjects=kept.subject_id.nunique(), n_events=None, status="executed",
             note="excluded 102/104/107/217"),
        dict(step="1_pvc_candidates_V", branch="cohort", split="all(pre-split)", g_ms=None,
             n_records=int((kept.n_pvc_candidates_step1 > 0).sum()), n_subjects=cands.subject_id.nunique(),
             n_events=len(cands), status="executed",
             note="g-independent beat-symbol 'V' annotations; n_records/n_subjects count those with >=1"),
        dict(step="split_development_test", branch="cohort", split="development|test", g_ms=None,
             n_records=None, n_subjects=None, n_events=None, status="blocked", note="split.seed/stratification pending"),
    ]
    blk = "blocked: " + ", ".join(pending_settings(cfg)[:3]) + " ..."
    for g in cfg["time"]["g_candidates_ms"]:
        for step in ("1b_within_record", "2_pre_half_criteria", "3_post_half_criteria",
                     "4_overlap_and_normal_matching", "5_common_pairs"):
            attr.append(dict(step=step, branch="common", split="development", g_ms=g, n_records=None,
                             n_subjects=None, n_events=None, status="blocked", note=blk))
        attr.append(dict(step="6_regime2_template_eligible", branch="regime2", split="development", g_ms=g,
                         n_records=None, n_subjects=None, n_events=None, status="blocked", note=blk))
    for branch in ("common", "regime2"):
        attr.append(dict(step="test_aggregation_g_star", branch=branch, split="test", g_ms=None, n_records=None,
                         n_subjects=None, n_events=None, status="blocked_gate",
                         note="g_star not frozen; Test candidate sweep forbidden"))
    attr = pd.DataFrame(attr)
    for c in ("n_records", "n_subjects", "n_events"):
        attr[c] = attr[c].astype("Int64")

    sym_all = pd.concat(sym_tabs, ignore_index=True)
    noise_all = pd.concat(noise_tabs, ignore_index=True)
    outputs = {
        "cohort_manifest.csv": man, "attrition_summary.csv": attr, "record_inventory.csv": inv,
        "annotation_symbol_counts.csv": sym_all, "signal_quality_annotations.csv": noise_all,
        "nstdb_noise_inventory.csv": ninv,
    }
    for name, df in outputs.items():
        atomic_write_csv(out / name, df)
    atomic_write_parquet(out / "event_audit.parquet", audit)

    summary = {
        "n_source_records": len(man), "n_source_subjects": int(man.subject_id.nunique()),
        "n_cohort_records": len(kept), "n_cohort_subjects": int(kept.subject_id.nunique()),
        "n_step1_pvc_candidates": len(cands), "event_list_sha256": event_list_sha,
        "records_mlii_not_index0": inv.loc[inv.mlii_channel_index.fillna(0) != 0, "record_id"].tolist(),
        "records_mlii_absent": inv.loc[inv.mlii_channel_index.isna(), "record_id"].tolist(),
        "records_samples_at_adc_bounds": inv.loc[(inv.n_samples_at_adc_min.fillna(0) + inv.n_samples_at_adc_max.fillna(0)) > 0,
                                                 "record_id"].tolist(),
        "n_samples_outside_adc_range_total": int(inv.n_samples_outside_adc_range.fillna(0).sum()),
        "noise_annotation_subtypes": {str(k): int(v) for k, v in noise_all.subtype.value_counts().sort_index().items()},
        "non_beat_symbols_seen": sorted(sym_all.loc[~sym_all.is_beat, "symbol"].unique().tolist()),
        "beat_symbols_seen": sorted(sym_all.loc[sym_all.is_beat, "symbol"].unique().tolist()),
        "config_hash": config_hash(cfg), "code_hash": code_hash(), "run_id": run_id,
    }
    atomic_write_json(out / "cohort_summary.json", summary)
    atomic_write_json(stamp, {"key": key, "summary": summary,
                              "sha256": {p: sha256_file(out / p) for p in products}})
    return summary


def run_split(cfg: dict, out: Path, run_id: str) -> pd.DataFrame:
    path = out / "cohort_manifest.csv"
    if not path.exists():
        raise C.BlockedError("cohort_manifest.csv missing; run cohort first")
    man = pd.read_csv(path, dtype={"record_id": str, "subject_id": str})
    if "mlii_ch_idx" not in man and "mlii_channel_index" in man:
        man["mlii_ch_idx"] = man["mlii_channel_index"]
    if "included" not in man:
        man["included"] = man["in_stage0_cohort"]
    included = man[man["in_stage0_cohort"].astype(str).str.lower().eq("true")]
    counts = included.groupby("subject_id")["n_pvc_candidates_step1"].sum().astype(int).to_dict()
    assignments = C.split_subjects(list(counts), cfg["split"], counts)
    positive = [n for n in counts.values() if n >= 1]
    threshold = float(np.median(positive)) if positive else 0.0
    rows = []
    for s in sorted(counts):
        n = counts[s]
        rows.append(dict(subject_id=s, stratum="zero" if n == 0 else "low" if n <= threshold else "high",
                         subject_pvc_candidates=n, split=assignments[s], seed=cfg["split"]["seed"],
                         stratification=cfg["split"]["stratification"], median_threshold=threshold,
                         config_hash=config_hash(cfg), code_hash=code_hash(), run_id=run_id))
    assignment = pd.DataFrame(rows)
    if (assignment.split == "development").sum() != cfg["split"]["n_development_subjects"] or (assignment.split == "test").sum() != cfg["split"]["n_test_subjects"]:
        raise C.CohortError("split subject counts differ from configured 22/21")
    shared = man[man.record_id.isin(["201", "202"])]
    if len(shared) == 2 and shared.subject_id.nunique() != 1:
        raise C.CohortError("201 and 202 are not grouped as one subject")
    dev_set = {s for s, value in assignments.items() if value == "development"}
    test_set = {s for s, value in assignments.items() if value == "test"}
    if dev_set & test_set:
        raise C.CohortError("Development and Test subject sets overlap")
    prior = out / "split_assignment.csv"
    core = ["subject_id", "stratum", "subject_pvc_candidates", "split", "seed", "stratification", "median_threshold"]
    if prior.exists():
        old = pd.read_csv(prior, dtype={"subject_id": str})
        if not old[core].sort_values("subject_id").reset_index(drop=True).astype(str).equals(assignment[core].sort_values("subject_id").reset_index(drop=True).astype(str)):
            raise C.BlockedError("existing split_assignment.csv differs; split is immutable")
    else:
        atomic_write_csv(prior, assignment)
    man["split"] = man.subject_id.map(assignments)
    man["split_group"] = man["split"]
    man["split_status"] = np.where(man.subject_id.isin(assignments), "assigned", "not_applicable")
    man["config_hash"] = config_hash(cfg)
    man["code_hash"] = code_hash()
    man["run_id"] = run_id
    atomic_write_csv(path, man)
    return assignment


def run_development_sweep(cfg: dict, out: Path, run_id: str) -> None:
    require_dev_sweep_allowed(cfg)
    from topovital.stage0_pipeline import run_pipeline
    run_pipeline(cfg, out, run_id, "development", cfg["time"]["g_candidates_ms"])


def run_test(cfg: dict, out: Path, run_id: str, g_star: int) -> None:
    from topovital.stage0_pipeline import run_pipeline
    run_pipeline(cfg, out, run_id, "test", [g_star])
