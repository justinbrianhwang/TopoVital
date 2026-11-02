"""Approval, freeze and one-shot guards. No signal is read here."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import csv
from pathlib import Path

from topovital.cohort import BlockedError
from topovital.util import code_hash, load_yaml
from topovital_analysis.level1 import analysis_code_hash
from topovital_analysis.runtime import ANALYSIS_CODE_VERSION
from topovital_stage1.features import stage1_code_hash

from . import final_code_hash

MANIFEST = Path("outputs/final_test/final_test_freeze_manifest.json")
CONFIGS = ("final_test_evaluation.yaml", "stage0.yaml", "stage1.yaml",
           "stage1_regime1.yaml", "stage1_analysis.yaml", "stage1_factorial.yaml",
           "stage1_level3.yaml")
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "gudhi", "wfdb",
            "neurokit2", "PyWavelets", "pyarrow", "threadpoolctl")
MEMBERSHIP = ("outputs/stage1_features/development/snippet_allocation.csv",
              "outputs/final_test/test_noise_allocation.csv",
              "outputs/final_test/test_event_manifest.csv", "outputs/split_assignment.csv")


def _sha(path: Path) -> str:
    if not path.is_file():
        raise BlockedError(f"Freeze input missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pending(value) -> bool:
    if value is None or value == "PENDING":
        return True
    if isinstance(value, dict):
        return any(_pending(v) for v in value.values())
    if isinstance(value, list):
        return any(_pending(v) for v in value)
    return False


def _raw_data(root: Path, events) -> tuple[dict, str]:
    """Hash raw file bytes against the published input checksum inventory."""
    checksums = root / "outputs/input_checksums.csv"
    checksum_sha = _sha(checksums)
    with checksums.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {}
    for row in rows:
        relative = row.get("relative_path", "")
        if relative in expected:
            raise BlockedError(f"Duplicate input checksum row: {relative}")
        expected[relative] = row.get("sha256_expected", "")
    mitdb = "mit-bih-arrhythmia-database-1.0.0"
    nstdb = "mit-bih-noise-stress-test-database-1.0.0"
    names = {f"{mitdb}/{record}.{suffix}"
             for column in ("record_id", "normal_record_id")
             for record in events[column].astype(str)
             for suffix in ("dat", "hea", "atr")}
    names |= {f"{nstdb}/{noise}.{suffix}" for noise in ("bw", "ma", "em")
              for suffix in ("dat", "hea")}
    hashes = {}
    for relative in sorted(names):
        if not expected.get(relative):
            raise BlockedError(f"Input checksum row missing: {relative}")
        digest = _sha(root / "data/raw" / relative)
        if digest.lower() != expected[relative].lower():
            raise BlockedError(f"Raw data SHA256 mismatch: {relative}")
        hashes[relative] = digest
    return hashes, checksum_sha


def _decisions(events) -> dict:
    records = events[["record_id", "mlii_channel_index"]].drop_duplicates()
    if records.record_id.duplicated().any():
        raise BlockedError("Conflicting MLII channel for a Test record")
    channels = [{"record_id": str(row.record_id),
                 "mlii_channel_index": int(row.mlii_channel_index),
                 "nstdb_channel": int(row.mlii_channel_index),
                 "nstdb_channel_name": f"noise{int(row.mlii_channel_index) + 1}"}
                for row in records.sort_values("record_id", kind="stable").itertuples(index=False)]
    if any(row["nstdb_channel"] not in (0, 1) for row in channels):
        raise BlockedError("Unsupported MLII signal index")
    return {"U-01": {"rule": "non_overlapping_grid_seeded_permutation_test_block",
                      "approval_date": "2026-11-02"},
            "U-02": {"rule": "match_ecg_signal_index", "approval_date": "2026-11-02",
                      "per_record_channels": channels},
            "ph_failure_policy": "not_evaluable_with_common_set"}


def _inputs(root: Path) -> dict:
    import pandas as pd

    cfg = load_yaml(root / "configs/final_test_evaluation.yaml")
    settings = cfg["settings"]
    if (settings.get("test_noise_allocation", {}).get("rule") !=
            "non_overlapping_grid_seeded_permutation_test_block" or
            settings.get("noise_channel_rule") != "match_ecg_signal_index"):
        raise BlockedError("Approved U-01 and U-02 rules are required for freeze")
    model_dir = root / settings["models_from"]
    models = sorted(model_dir.glob("*.npz"))
    if len(models) != 5:
        raise BlockedError("Freeze requires exactly five final models")
    selection_path = root / settings["selection_record"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if {p.stem for p in models} != set(selection):
        raise BlockedError("Final model filenames differ from Development selection")
    paths = [*(root / "configs" / name for name in CONFIGS
               if name != "final_test_evaluation.yaml"),
             *(root / path for path in MEMBERSHIP),
             selection_path,
             *models]
    events = pd.read_csv(root / "outputs/final_test/test_event_manifest.csv",
                         dtype={"event_id": str, "record_id": str,
                                "normal_record_id": str, "subject_id": str})
    raw_data, checksum_sha = _raw_data(root, events)
    record_114 = events.loc[events.record_id.eq("114")]
    record_114_info = {"in_test": bool(len(record_114)),
                       "subject_id": str(record_114.subject_id.iloc[0]) if len(record_114) else None,
                       "n_events": int(len(record_114)),
                       "nstdb_channel": int(record_114.mlii_channel_index.iloc[0]) if len(record_114) else None}
    return {
        "final_code_hash": final_code_hash(),
        "analysis_code_hash": analysis_code_hash(),
        "stage1_code_hash": stage1_code_hash(),
        "stage0_code_hash": code_hash(),
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "final_test_settings_sha256": hashlib.sha256(json.dumps(
            {"settings": cfg["settings"], "output": cfg["output"]},
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "files": {p.relative_to(root).as_posix(): _sha(p) for p in paths},
        "raw_data": raw_data,
        "input_checksums_sha256": checksum_sha,
        "decisions": _decisions(events),
        "record_114": record_114_info,
        "thresholds": {name: {k: float(v[k]) for k in ("threshold_primary", "threshold_c1")}
                       for name, v in selection.items()},
        "package_versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def build_freeze_manifest(root: Path) -> dict:
    """Write a reviewable manifest; this operation does not grant Test approval."""
    from topovital.util import atomic_write_json
    root = Path(root).resolve()
    manifest = _inputs(root)
    atomic_write_json(root / MANIFEST, manifest)
    return manifest


def validate_freeze_manifest(root: Path, manifest: dict) -> None:
    try:
        current = _inputs(Path(root).resolve())
    except BlockedError:
        raise
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as exc:
        raise BlockedError(f"Final Test freeze input unavailable: {type(exc).__name__}") from exc
    if manifest != current:
        raise BlockedError("Final Test freeze manifest mismatch")


def require_test_approval(root: Path) -> dict:
    """This must be the first operation of every Test command."""
    root = Path(root).resolve()
    try:
        cfg = load_yaml(root / "configs/final_test_evaluation.yaml")
    except (OSError, ValueError) as exc:
        raise BlockedError("Final Test approval config unreadable") from exc
    if (cfg.get("status") != "approved_for_test_execution"
            or cfg.get("approved_for_test_execution") is not True
            or _pending(cfg.get("settings"))):
        raise BlockedError("Final Test requires approved status and complete settings")
    path = root / MANIFEST
    if not path.is_file():
        raise BlockedError("Final Test freeze manifest missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise BlockedError("Final Test freeze manifest unreadable") from exc
    validate_freeze_manifest(root, manifest)
    output = root / "outputs/final_test"
    allowed = {MANIFEST.name, "test_event_manifest.csv", "test_noise_allocation.csv",
               "dev_verification", "hashes", "resource_profile.csv", "public"}
    if (output / "test_predictions.parquet").exists():
        raise BlockedError("Final Test predictions already exist; quarantine required")
    profile = output / "resource_profile.csv"
    if profile.exists():
        try:
            with profile.open(newline="", encoding="utf-8") as stream:
                if any(row.get("command") == "run-test" for row in csv.DictReader(stream)):
                    raise BlockedError("Earlier Final Test run profile exists; quarantine required")
        except OSError as exc:
            raise BlockedError("Final Test profile unreadable; quarantine required") from exc
    if any(p.name not in allowed and not p.name.startswith("dev_verification")
           for p in output.iterdir()):
        raise BlockedError("Earlier Final Test output exists; quarantine required")
    if (output / "hashes").exists() and any((output / "hashes").iterdir()):
        raise BlockedError("Earlier Final Test hash output exists; quarantine required")
    return manifest
