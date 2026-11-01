"""Development-only Level 1 PH distance pipeline."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gudhi.hera
import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from topovital.util import (PROJECT_ROOT, atomic_write_csv, atomic_write_json,
                            atomic_write_parquet, canonical_json, peak_rss_mb)

from .distances import diagram_distances, finite_diagram
from .stats import separation


def analysis_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def validate_events(events: pd.DataFrame, expected: int | None = None) -> None:
    required = {"event_id", "subject_id", "split"}
    if not required.issubset(events.columns) or events.empty or events.event_id.duplicated().any():
        raise BlockedError("Invalid Development event list")
    if not events.split.eq("development").all():
        raise BlockedError("Test or non-Development subject rejected")
    if expected is not None and len(events) != expected:
        raise BlockedError(f"Expected {expected} Development events; got {len(events)}")
    for event_id in events.event_id.astype(str):
        if not re.fullmatch(r"[A-Za-z0-9_]+", event_id):
            raise BlockedError("Unsafe event identifier")


def _safe_job(path: Path, allowed: tuple[Path, Path]) -> Path:
    resolved = path.resolve()
    if resolved.suffix != ".npz" or resolved.parent not in tuple(p.resolve() for p in allowed):
        raise BlockedError("Job file outside Development directories")
    return resolved


def _snr_label(value) -> str:
    number = int(value)
    if number != value:
        raise BlockedError("Noninteger SNR filename value")
    return f"m{-number}" if number < 0 else str(number)


def window_specs(events: pd.DataFrame, cfg: dict, root: Path):
    inputs, scope = cfg["inputs"], cfg["scope"]
    allowed = (root / inputs["clean_jobs"], root / inputs["regime1_jobs"])
    clean = set(scope["conditions"]) - set(inputs["pre_half_of"])
    if clean != {"C0", "C1"} or set(inputs["pre_half_of"]) != {"C2", "C3"}:
        raise BlockedError("Unsupported Development condition configuration")
    for ev in events.itertuples(index=False):
        eid = str(ev.event_id)
        for condition in scope["conditions"]:
            if condition in clean:
                pre = _safe_job(allowed[0] / f"{eid}__{condition}__pre.npz", allowed)
                post = _safe_job(allowed[0] / f"{eid}__{condition}__post.npz", allowed)
                yield {"event_id": eid, "subject_id": str(ev.subject_id), "condition": condition,
                       "noise_type": "", "snr_target": None, "pre_path": pre, "post_path": post}
            else:
                source = inputs["pre_half_of"][condition]
                if source not in clean:
                    raise BlockedError("Invalid pre_half_of configuration")
                pre = _safe_job(allowed[0] / f"{eid}__{source}__pre.npz", allowed)
                for noise in scope["regime1_cells"]["noise_types"]:
                    for snr in scope["regime1_cells"]["snr_db"]:
                        post = _safe_job(allowed[1] / f"{eid}__{condition}__{noise}__{_snr_label(snr)}__post.npz", allowed)
                        yield {"event_id": eid, "subject_id": str(ev.subject_id), "condition": condition,
                               "noise_type": noise, "snr_target": snr,
                               "pre_path": pre, "post_path": post}


def _read_diagrams(path: Path, spec: dict, half: str, cfg: dict, root: Path, allowed):
    path = _safe_job(path, allowed)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata.get("validation_status") != "ok":
            raise ValueError("invalid_job_validation_status")
        if str(metadata.get("event_id")) != spec["event_id"] or metadata.get("half") != half:
            raise ValueError("job_identity_mismatch")
        expected_condition = spec["condition"] if half == "post" or spec["condition"] in ("C0", "C1") else cfg["inputs"]["pre_half_of"][spec["condition"]]
        if metadata.get("condition") != expected_condition:
            raise ValueError("job_condition_mismatch")
        diagrams = {}
        counts = {}
        for rep, key in cfg["inputs"]["representations"].items():
            values = finite_diagram(saved[key], vr_h0=key == "vr_h0", vr_h1=key == "vr_h1")
            diagrams[rep] = values
            counts[f"{rep}_finite_pairs"] = len(values)
    entry = {"job_path": path.relative_to(root.resolve()).as_posix(), "condition": expected_condition,
             "half": half, "noise_type": spec["noise_type"] if half == "post" else "",
             "snr_target": spec["snr_target"] if half == "post" else None,
             "sha256": digest, **counts}
    return diagrams, entry


def _worker(spec: dict, cfg: dict, root: Path, crosscheck: bool, crosscheck_delta: float):
    allowed = (root / cfg["inputs"]["clean_jobs"], root / cfg["inputs"]["regime1_jobs"])
    base = {key: spec[key] for key in ("event_id", "subject_id", "condition", "noise_type", "snr_target")}
    columns = [f"{rep}__{metric}" for rep in cfg["inputs"]["representations"] for metric in cfg["level1"]["metrics"]]
    row = {**base, **{name: np.nan for name in columns}, "status": "ok", "reason": ""}
    index = []
    checks = []
    try:
        pre, pre_entry = _read_diagrams(spec["pre_path"], spec, "pre", cfg, root, allowed)
        index.append(pre_entry)
        post, post_entry = _read_diagrams(spec["post_path"], spec, "post", cfg, root, allowed)
        index.append(post_entry)
        for rep in pre:
            values = diagram_distances(pre[rep], post[rep], cfg["level1"]["metrics"])
            for metric, value in values.items():
                row[f"{rep}__{metric}"] = value
                if crosscheck and metric in ("wasserstein_1", "wasserstein_2"):
                    hera = float(gudhi.hera.wasserstein_distance(
                        pre[rep], post[rep], order=int(metric[-1]), internal_p=np.inf,
                        delta=crosscheck_delta))
                    relative = abs(hera - value) / max(abs(value), 1e-12)
                    checks.append({**base, "representation": rep, "metric": metric,
                                   "exact": value, "hera": hera, "relative_difference": relative})
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        # Exception messages from file I/O can contain a user's absolute path.
        reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else type(exc).__name__
        row.update(status="not_evaluable", reason=reason)
        indexed = {entry["job_path"] for entry in index}
        for half in ("pre", "post"):
            path = _safe_job(spec[f"{half}_path"], allowed)
            if path.is_file():
                relative = path.relative_to(root.resolve()).as_posix()
                if relative not in indexed:
                    index.append({"job_path": relative, "condition":
                                  cfg["inputs"]["pre_half_of"].get(spec["condition"], spec["condition"]) if half == "pre" else spec["condition"],
                                  "half": half, "noise_type": spec["noise_type"] if half == "post" else "",
                                  "snr_target": spec["snr_target"] if half == "post" else None,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "status": "not_evaluable", "reason": reason})
    return row, index, checks, peak_rss_mb()


def _summaries(frame: pd.DataFrame, cfg: dict):
    keys = ["condition", "noise_type", "snr_target"]
    window_rows, subject_rows = [], []
    columns = [f"{rep}__{metric}" for rep in cfg["inputs"]["representations"] for metric in cfg["level1"]["metrics"]]
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        for column in columns:
            rep, metric = column.split("__")
            values = group[column].dropna().to_numpy(dtype=float)
            common = dict(zip(keys, group_key))
            common.update(representation=rep, metric=metric)
            window_rows.append({**common, "n": len(values), "median": float(np.median(values)) if len(values) else np.nan,
                                "q25": float(np.percentile(values, 25)) if len(values) else np.nan,
                                "q75": float(np.percentile(values, 75)) if len(values) else np.nan,
                                "status": "ok" if len(values) else "not_evaluable",
                                "reason": "" if len(values) else "no_finite_distances"})
            medians = group.groupby("subject_id")[column].median().dropna().to_numpy(dtype=float)
            subject_rows.append({**common, "n": len(medians), "n_subjects": len(medians),
                                 "median": float(np.median(medians)) if len(medians) else np.nan,
                                 "q25": float(np.percentile(medians, 25)) if len(medians) else np.nan,
                                 "q75": float(np.percentile(medians, 75)) if len(medians) else np.nan,
                                 "median_of_subject_medians": float(np.median(medians)) if len(medians) else np.nan,
                                 "status": "ok" if len(medians) else "not_evaluable",
                                 "reason": "" if len(medians) else "no_evaluable_subjects"})
    return pd.DataFrame(window_rows), pd.DataFrame(subject_rows)


def _separations(frame: pd.DataFrame, cfg: dict):
    sep_cfg = cfg["level1"]["separation"]
    rule = sep_cfg["threshold_rule"]
    match = re.search(r">=\s*(0(?:\.\d+)?|1(?:\.0+)?)", rule)
    if not match:
        raise BlockedError("Cannot parse LOSO sensitivity from config")
    target = float(match.group(1))
    boot = sep_cfg["bootstrap"]
    rows, details = [], []
    c1 = frame.loc[frame.condition.eq("C1")]
    for noise in cfg["scope"]["regime1_cells"]["noise_types"]:
        for snr in cfg["scope"]["regime1_cells"]["snr_db"]:
            c2 = frame.loc[frame.condition.eq("C2") & frame.noise_type.eq(noise) & frame.snr_target.eq(snr)]
            for rep in cfg["inputs"]["representations"]:
                for metric in cfg["level1"]["metrics"]:
                    column = f"{rep}__{metric}"
                    scored = pd.concat((c1, c2), ignore_index=True)[["subject_id", "condition", column]].rename(columns={column: "score"})
                    result, detail = separation(scored, "C1", "C2", target, int(boot["resamples"]), int(boot["seed"]))
                    fields = {"noise_type": noise, "snr_target": snr, "representation": rep, "metric": metric}
                    rows.append({**fields, **result})
                    for record in detail.to_dict("records"):
                        details.append({**fields, **record})
    return pd.DataFrame(rows), pd.DataFrame(details)


def run_level1(events: pd.DataFrame, cfg: dict, root: Path, output: Path, workers: int, *, pilot: bool = False):
    started = time.perf_counter()
    validate_events(events, None if pilot else int(cfg["scope"]["events"]))
    if cfg["scope"]["split"] != "development_only" or cfg["inputs"]["pairs"] != "finite_only":
        raise BlockedError("Unsupported analysis scope")
    if workers < 1:
        raise BlockedError("--workers must be positive")
    specs = sorted(window_specs(events, cfg, root), key=lambda item: (item["event_id"], item["condition"], item["noise_type"], str(item["snr_target"])))
    # Direct expression: two clean windows plus two injected conditions per cell.
    expected = len(events) * (2 + 2 * len(cfg["scope"]["regime1_cells"]["noise_types"]) * len(cfg["scope"]["regime1_cells"]["snr_db"]))
    if len(specs) != expected:
        raise BlockedError("Unexpected number of Development windows")
    rows, entries, checks, peaks = [], [], [], []
    delta = float(re.search(r"delta\s*=\s*([0-9.eE-]+)", cfg["level1"]["crosscheck"]).group(1)) if "delta=" in cfg["level1"]["crosscheck"] else 1e-4
    if workers == 1:
        results = (_worker(spec, cfg, root, i < 50, delta) for i, spec in enumerate(specs))
        for row, index, cross, peak in results:
            rows.append(row); entries.extend(index); checks.extend(cross); peaks.append(peak)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, spec, cfg, root, i < 50, delta): spec for i, spec in enumerate(specs)}
            for future in as_completed(futures):
                row, index, cross, peak = future.result()
                rows.append(row); entries.extend(index); checks.extend(cross); peaks.append(peak)
    frame = pd.DataFrame(rows).sort_values(["event_id", "condition", "noise_type", "snr_target"], kind="stable")
    index = (pd.DataFrame(entries).drop_duplicates("job_path").sort_values("job_path") if entries
             else pd.DataFrame(columns=["job_path", "condition", "half", "noise_type", "snr_target", "sha256"]))
    manifest = hashlib.sha256(canonical_json(index[["job_path", "sha256"]].to_dict("records")).encode()).hexdigest()
    summary_window, summary_subject = _summaries(frame, cfg)
    separation_table, per_subject = _separations(frame, cfg)
    cross = pd.DataFrame(checks)
    if len(cross):
        cross["max_relative_difference"] = cross.groupby(["representation", "metric"])["relative_difference"].transform("max")
    atomic_write_csv(output / "diagram_index.csv", index)
    atomic_write_parquet(output / "ph_distances.parquet", frame)
    atomic_write_csv(output / "distance_summary_window.csv", summary_window)
    atomic_write_csv(output / "distance_summary_subject.csv", summary_subject)
    atomic_write_csv(output / "c1_c2_separation.csv", separation_table)
    atomic_write_csv(output / "c1_c2_per_subject.csv", per_subject)
    atomic_write_csv(output / "hera_crosscheck.csv", cross)
    run_summary = {"windows": len(frame), "distances": int(frame[[c for c in frame if "__" in c]].notna().sum().sum()),
                   "failures": int(frame.status.ne("ok").sum()), "wall_time_s": time.perf_counter() - started,
                   "parent_peak_rss_mb": peak_rss_mb(),
                   "worker_peak_rss_mb": max((p for p in peaks if p is not None), default=None),
                   "workers": workers, "config_sha256": hashlib.sha256((root / "configs" / "stage1_analysis.yaml").read_bytes()).hexdigest(),
                   "code_hash": analysis_code_hash(), "input_npz_manifest_sha256": manifest,
                   "hera_max_relative_difference": float(cross.relative_difference.max()) if len(cross) else None}
    atomic_write_json(output / "run_summary.json", run_summary)
    return run_summary
