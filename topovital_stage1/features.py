"""Development-only, resumable Stage 1 topology feature jobs."""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import gudhi
import numpy as np
import pandas as pd
import psutil

from topovital.cohort import BlockedError
from topovital.util import (atomic_write_bytes, atomic_write_csv, atomic_write_json,
                            canonical_json, config_hash)

from .core import delay_vectors, iter_segments, uniform_indices
from .ph import monitored_vr


DIAGRAM_KEYS = ("vr_h0", "vr_h1", "cubical_sub_finite", "cubical_sub_essential_births",
                "cubical_super_finite", "cubical_super_essential_births")
SCIENTIFIC_FIELDS = ("normalization", "embedding", "vietoris_rips", "cubical")
EXPECTED_SIMPLICES = 300 + math.comb(300, 2) + math.comb(300, 3)


def require_approved_parameters(cfg: dict) -> dict:
    """Refuse changes to any parameter that controls the approved feature calculation."""
    norm, emb, vr, cub = (cfg.get(name, {}) for name in SCIENTIFIC_FIELDS)
    expected = {
        "normalization.reference": (norm.get("reference"), "pre_window"),
        "normalization.percentile_method": (norm.get("percentile_method"), "linear"),
        "embedding.tau_samples": (emb.get("tau_samples"), 28),
        "embedding.m": (emb.get("m"), 5),
        "embedding.never_cross_t0": (emb.get("never_cross_t0"), True),
        "vietoris_rips.n_points_per_half": (vr.get("n_points_per_half"), 300),
        "vietoris_rips.subsample_rule": (vr.get("subsample_rule"), "uniform_stride"),
        "vietoris_rips.r_max": (vr.get("r_max"), "null_means_unbounded"),
        "vietoris_rips.max_dimension": (vr.get("max_dimension"), 2),
        "vietoris_rips.backend": (vr.get("backend"), "gudhi"),
        "cubical.sublevel": (cub.get("sublevel"), "x"),
        "cubical.superlevel": (cub.get("superlevel"), "-x"),
        "cubical.essential_births": (cub.get("essential_births"), ["min(x)", "-max(x)"]),
    }
    bad = [name for name, (actual, approved) in expected.items() if actual != approved]
    r_max_value = vr.get("r_max_value", 0)
    if not isinstance(r_max_value, (int, float)) or not math.isinf(r_max_value) or r_max_value < 0:
        bad.append("vietoris_rips.r_max_value")
    if "max_edge_length" in vr:
        bad.append("vietoris_rips.max_edge_length")
    if bad:
        raise BlockedError("Stage 1 scientific parameters differ from approved values: " + ", ".join(bad))
    return {name: cfg[name] for name in SCIENTIFIC_FIELDS}


def stage1_code_hash(package_dir: Path | None = None) -> str:
    directory = package_dir or Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def input_hash(half: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(half, dtype="<f8").tobytes()).hexdigest()


def make_cache_key(input_sha256: str, scientific: dict, version: str, code_sha256: str) -> str:
    content = {"input_sha256": input_sha256, "scientific_parameters": scientific,
               "gudhi_version": version, "stage1_code_hash": code_sha256}
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def job_path(jobs_dir: Path, info: dict) -> Path:
    event_id = str(info["event_id"])
    if not event_id or any(c in event_id for c in ("/", "\\", ":")) or event_id in (".", ".."):
        raise BlockedError(f"Unsafe event_id: {event_id!r}")
    return jobs_dir / f"{event_id}__{info['condition']}__{info['half']}.npz"


def save_job(path: Path, diagrams: dict, metadata: dict) -> None:
    payload = {key: np.asarray(diagrams[key], dtype=float) for key in DIAGRAM_KEYS}
    payload["metadata"] = np.array(canonical_json(metadata))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    atomic_write_bytes(path, buffer.getvalue())


def load_job(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as saved:
        diagrams = {key: saved[key] for key in DIAGRAM_KEYS}
        metadata = json.loads(str(saved["metadata"].item()))
    return diagrams, metadata


def cached_metadata(path: Path, key: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        diagrams, metadata = load_job(path)
        if metadata.get("cache_key") == key and metadata.get("validation_status") == "ok" and all(
            np.isfinite(values).all() for name, values in diagrams.items() if "essential" in name
        ):
            return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return None


def representative_events(events: pd.DataFrame, n: int) -> pd.DataFrame:
    if n < 1:
        raise ValueError("--events must be positive")
    ordered = events.sort_values(["subject_id", "event_id"], kind="stable")
    return ordered.drop_duplicates("subject_id", keep="first").head(n).reset_index(drop=True)


def check_memory(workers: int, available: int | None = None) -> int:
    if workers < 1:
        raise BlockedError("--workers must be at least 1")
    available = psutil.virtual_memory().available if available is None else available
    safe_max = max(0, (available - 1_000_000_000) // 600_000_000)
    if workers > safe_max:
        raise BlockedError(f"Insufficient available RAM for {workers} workers; safe maximum: {safe_max}")
    return int(safe_max)


def _cubical_diagram(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = gudhi.CubicalComplex(top_dimensional_cells=cells).persistence()
    h0 = np.asarray([pair for dim, pair in pairs if dim == 0], dtype=float).reshape(-1, 2)
    return h0[np.isfinite(h0[:, 1])], h0[np.isinf(h0[:, 1]), 0]


def validate_job(diagrams: dict, metadata: dict, half: np.ndarray) -> list[str]:
    messages = []
    if metadata["vr_status"] != "success":
        messages.append("vr_failed")
    else:
        h0, h1 = diagrams["vr_h0"], diagrams["vr_h1"]
        if int(np.isinf(h0[:, 1]).sum()) != 1:
            messages.append("vr_h0_essential_count")
        if int(np.isfinite(h0[:, 1]).sum()) != 299:
            messages.append("vr_h0_finite_count")
        if int(np.isinf(h1[:, 1]).sum()) != 0:
            messages.append("vr_h1_essential_count")
        if metadata.get("num_simplices") != EXPECTED_SIMPLICES:
            messages.append("vr_num_simplices")
    if metadata["cubical_status"] != "success":
        messages.append("cubical_failed")
    else:
        sub, sup = diagrams["cubical_sub_essential_births"], diagrams["cubical_super_essential_births"]
        if len(sub) != 1 or not np.isclose(sub[0], np.min(half), rtol=0, atol=1e-10):
            messages.append("cubical_sub_essential_birth")
        if len(sup) != 1 or not np.isclose(sup[0], -np.max(half), rtol=0, atol=1e-10):
            messages.append("cubical_super_essential_birth")
    for name in ("vr_h0", "vr_h1", "cubical_sub_finite", "cubical_super_finite"):
        pairs = diagrams[name]
        if pairs.ndim != 2 or pairs.shape[1] != 2 or np.isnan(pairs).any() or (pairs[:, 0] > pairs[:, 1]).any():
            messages.append(f"{name}_birth_after_death")
    return messages


def _empty_diagrams() -> dict:
    return {key: np.empty((0,) if "essential" in key else (0, 2), dtype=float) for key in DIAGRAM_KEYS}


def compute_job(info: dict, half: np.ndarray | None, path: Path, cache_key: str,
                input_sha256: str, cfg_hash: str, code_sha256: str, run_id: str,
                scientific: dict) -> dict:
    diagrams = _empty_diagrams()
    metadata = {key: info[key] for key in ("event_id", "subject_id", "record_id", "condition", "half")}
    metadata.update({key: info[key] for key in (
        "snippet_id", "snippet_start", "noise_type", "channel", "snr_target",
        "snr_actual", "scale_a", "norm_rel_error", "reuse_count",
    ) if key in info})
    metadata.update(n_delay_vectors=0, vr_status="failed_input", vr_failure_reason="",
                    vr_wall_time_s=0.0, vr_child_peak_rss_mb=0.0, num_simplices=0,
                    cubical_status="failed_input", cubical_wall_time_s=0.0,
                    cache_key=cache_key, input_sha256=input_sha256, gudhi_version=gudhi.__version__,
                    stage1_code_hash=code_sha256, config_hash=cfg_hash, run_id=run_id)
    if half is None:
        metadata["vr_failure_reason"] = info.get("normalization_failure_reason", "invalid_normalized_half")
    else:
        try:
            vectors = delay_vectors(half, scientific["embedding"]["tau_samples"],
                                    scientific["embedding"]["m"])
            metadata["n_delay_vectors"] = len(vectors)
            points = vectors[uniform_indices(len(vectors), scientific["vietoris_rips"]["n_points_per_half"])]
            result = monitored_vr(points, include_diagrams=True)
            metadata.update({key: value for key, value in result.items() if key not in ("h0", "h1")})
            if result["vr_status"] == "success":
                diagrams["vr_h0"], diagrams["vr_h1"] = result["h0"], result["h1"]
        except Exception as exc:
            metadata.update(vr_status="failed_input", vr_failure_reason=f"{type(exc).__name__}:{exc}")
        started = time.perf_counter()
        try:
            diagrams["cubical_sub_finite"], diagrams["cubical_sub_essential_births"] = _cubical_diagram(half)
            diagrams["cubical_super_finite"], diagrams["cubical_super_essential_births"] = _cubical_diagram(-half)
            metadata["cubical_status"] = "success"
        except Exception as exc:
            metadata["cubical_status"] = "failed_cubical"
            metadata["cubical_failure_reason"] = f"{type(exc).__name__}:{exc}"
        metadata["cubical_wall_time_s"] = time.perf_counter() - started
    for prefix, key in (("h0", "vr_h0"), ("h1", "vr_h1")):
        deaths = diagrams[key][:, 1]
        metadata[f"{prefix}_finite"] = int(np.isfinite(deaths).sum())
        metadata[f"{prefix}_essential"] = int(np.isinf(deaths).sum())
    for prefix in ("sub", "super"):
        metadata[f"cubical_{prefix}_finite"] = len(diagrams[f"cubical_{prefix}_finite"])
        metadata[f"cubical_{prefix}_essential"] = len(diagrams[f"cubical_{prefix}_essential_births"])
    messages = validate_job(diagrams, metadata, half) if half is not None else ["normalization_failed"]
    metadata["validation_status"] = "ok" if not messages else "failed"
    metadata["validation_messages"] = messages
    save_job(path, diagrams, metadata)
    return metadata


def run_features(events: pd.DataFrame, data_root: Path, out_dir: Path, cfg: dict,
                 workers: int, run_id: str) -> dict:
    scientific = require_approved_parameters(cfg)
    check_memory(workers)
    cfg_hash = config_hash(cfg)
    code_sha256 = stage1_code_hash()
    jobs_dir = out_dir / "jobs"
    expected = 4 * len(events)
    rows = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def finish(done):
            for future in done:
                rows.append(future.result())

        for info, half in iter_segments(events, data_root):
            path = job_path(jobs_dir, info)
            sha = input_hash(half) if half is not None else ""
            key = make_cache_key(sha, scientific, gudhi.__version__, code_sha256)
            cached = cached_metadata(path, key) if half is not None else None
            if cached is not None:
                rows.append({**cached, "cached": True, "run_id": run_id})
                continue
            pending.add(pool.submit(compute_job, info, half, path, key, sha, cfg_hash, code_sha256,
                                    run_id, scientific))
            if len(pending) >= workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                finish(done)
        if pending:
            done, _ = wait(pending)
            finish(done)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} feature jobs; got {len(rows)}")
    frame = pd.DataFrame(rows).sort_values(["event_id", "condition", "half"], kind="stable")
    frame["cached"] = frame["cached"].fillna(False).astype(bool) if "cached" in frame else False
    success = int(frame.validation_status.eq("ok").sum())
    counts = events.groupby("subject_id").size().rename("events").reset_index()
    job_counts = frame.groupby("subject_id").agg(jobs=("event_id", "size"),
                                                  success=("validation_status", lambda x: int(x.eq("ok").sum())))
    counts = counts.join(job_counts, on="subject_id")
    counts["failed"] = counts["jobs"] - counts["success"]
    counts["run_id"] = run_id
    summary = {"total_jobs": expected, "success": success, "failed": expected - success,
               "cached": int(frame.cached.sum()), "total_wall_time_s": time.perf_counter() - started,
               "max_child_peak_rss_mb": float(frame.vr_child_peak_rss_mb.fillna(0).max()),
               "workers_used": workers, "run_id": run_id}
    atomic_write_csv(out_dir / "jobs_ledger.csv", frame)
    atomic_write_csv(out_dir / "per_subject_counts.csv", counts)
    atomic_write_json(out_dir / "run_summary.json", summary)
    return summary
