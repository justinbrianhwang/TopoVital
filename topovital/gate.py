"""Stage 0 gating: pending-setting detection, g_star freeze, Test lock (brief §7)."""
from __future__ import annotations

from pathlib import Path

from topovital.cohort import BlockedError
from topovital.util import (atomic_write_json, atomic_write_yaml, code_hash, config_hash,
                            load_yaml, sha256_file, utc_now)

# Config sections whose null leaves block the Stage 0 event pipeline.
STAGE0_SECTIONS = ("protocol", "split", "time", "events", "quality", "normal_matching",
                   "template", "noise")
DECISION_REQUIRED = ("g_star", "decided_by", "decision_procedure", "rationale",
                     "delineation_procedure", "waveform_inspection_performed",
                     "dev_audit_sha256")

DOWNSTREAM_PENDING = {
    "note": "Stage 0 freeze does not freeze these. To be decided by the protocol's "
            "Development procedures in later stages.",
    "embedding": None, "N_pt": None, "r_max": None, "max_dimension": None,
    "persistence_image": None, "classifier": None, "decision_threshold": None,
    "ph_backend_reference": "gudhi (brief §12)",
}


def pending_settings(cfg: dict) -> list[str]:
    out = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif node is None:
            out.append(prefix)

    for sec in STAGE0_SECTIONS:
        walk(cfg.get(sec), sec)
    return out


def require_dev_sweep_allowed(cfg: dict) -> None:
    missing = pending_settings(cfg)
    if missing:
        raise BlockedError(
            "Development g-sweep blocked: the following Stage 0 settings are pending and can only "
            "be taken from the TopoVital v1.3 protocol source (not available in this workspace):\n  - "
            + "\n  - ".join(missing)
            + "\nFill them in configs/stage0.yaml from the protocol and re-run:\n"
              "  python -m topovital stage0 --split development --g-sweep")


def require_test_allowed(locked_path: Path, g_sweep: bool) -> dict:
    """Test aggregation only with a frozen locked config; candidate sweeps never."""
    if g_sweep:
        raise BlockedError("g candidate sweep on the Test split is not permitted (g_star is chosen "
                           "on Development only).")
    if not locked_path.exists():
        raise BlockedError(f"{locked_path} not found. Run freeze-stage0 first.")
    locked = load_yaml(locked_path)
    if locked.get("stage0_freeze_status") != "frozen" or locked.get("g_star") is None:
        raise BlockedError(f"Test blocked: stage0_freeze_status={locked.get('stage0_freeze_status')!r}, "
                           f"g_star={locked.get('g_star')!r}. Freeze Stage 0 on Development first:\n"
                           "  python -m topovital freeze-stage0 --decision configs/g_decision.yaml")
    if locked.get("stage0_config_hash") != config_hash(locked["stage0"]):
        raise BlockedError("locked_config.yaml stage0 section was modified after freezing.")
    if locked.get("code_hash") != code_hash():
        raise BlockedError("pipeline code changed since freeze (code_hash mismatch); Test runs must "
                           "use the frozen code.")
    return locked


def build_locked_config(cfg: dict, status: str, g_star, blocking: list[str], freeze_record=None) -> dict:
    stage0 = {k: cfg[k] for k in ("protocol", "data", "cohort", "split", "time", "events", "quality",
                                  "normal_matching", "template", "noise")}
    if g_star is not None:
        stage0 = {**stage0, "time": {**stage0["time"], "g_star_ms": g_star}}
    return {
        "stage0_freeze_status": status,
        "g_star": g_star,
        "generated_utc": utc_now(),
        "stage0_config_hash": config_hash(stage0),
        "source_config_hash": config_hash(cfg),
        "code_hash": code_hash(),
        "freeze_record": freeze_record,
        "blocking_reasons": blocking,
        "stage0": stage0,
        "downstream_pending": DOWNSTREAM_PENDING,
        "execution_resources": cfg.get("execution"),
    }


def freeze(cfg: dict, decision_path: Path, out_dir: Path) -> dict:
    missing = pending_settings(cfg)
    if missing:
        raise BlockedError("freeze-stage0 blocked: Stage 0 settings still pending: " + ", ".join(missing))
    dec = load_yaml(decision_path) or {}
    absent = [k for k in DECISION_REQUIRED if dec.get(k) in (None, "")]
    if absent:
        raise BlockedError(f"freeze-stage0 blocked: decision file {decision_path} lacks: {absent}")
    if dec["g_star"] not in cfg["time"]["g_candidates_ms"]:
        raise BlockedError(f"g_star={dec['g_star']} is not one of {cfg['time']['g_candidates_ms']}")
    if dec["waveform_inspection_performed"] is not True:
        raise BlockedError("freeze-stage0 blocked: waveform inspection has not been recorded as performed.")
    dev_audit = out_dir / "development" / "event_audit_dev.parquet"
    if not dev_audit.exists():
        raise BlockedError(f"freeze-stage0 blocked: {dev_audit} missing (Development sweep not run).")
    if sha256_file(dev_audit) != dec["dev_audit_sha256"]:
        raise BlockedError("decision refers to a different Development audit (sha256 mismatch).")
    record = {"frozen_utc": utc_now(), "decision_file": str(decision_path),
              "decision_sha256": sha256_file(decision_path), "decision": dec}
    locked = build_locked_config(cfg, "frozen", dec["g_star"], [], record)
    atomic_write_json(out_dir / "stage0_freeze_record.json", {**record, "code_hash": locked["code_hash"],
                                                               "stage0_config_hash": locked["stage0_config_hash"]})
    atomic_write_yaml(out_dir / "locked_config.yaml", locked)
    return locked
