"""GUDHI resource pilot; VR execution is isolated in a monitored child process."""
from __future__ import annotations

import io
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gudhi
import numpy as np
import psutil

from topovital.util import PROJECT_ROOT

from .core import delay_vectors, uniform_indices

MEMORY_LIMIT_BYTES = 6_000_000_000


def pair_counts(pairs) -> dict:
    result = {"h0_pairs": 0, "h1_pairs": 0, "h0_infinite": 0, "h1_infinite": 0}
    for dim, (_, death) in pairs:
        if dim in (0, 1):
            result[f"h{dim}_pairs"] += 1
            result[f"h{dim}_infinite"] += int(np.isinf(death))
    return result


def vr_persistence(points: np.ndarray) -> dict:
    result = vr_diagrams(points)
    return {key: value for key, value in result.items() if key not in ("h0", "h1")}


def vr_diagrams(points: np.ndarray) -> dict:
    tree = gudhi.RipsComplex(points=np.asarray(points, dtype=float)).create_simplex_tree(max_dimension=2)
    pairs = tree.persistence()
    diagrams = {f"h{dim}": np.asarray([pair for d, pair in pairs if d == dim], dtype=float).reshape(-1, 2)
                for dim in (0, 1)}
    return {"num_simplices": tree.num_simplices(), **pair_counts(pairs), **diagrams}


def cubical_counts(half: np.ndarray) -> dict:
    x = np.asarray(half, dtype=float)
    out = {}
    for name, cells in (("sublevel", x), ("superlevel", -x)):
        pairs = gudhi.CubicalComplex(top_dimensional_cells=cells).persistence()
        out.update({f"{name}_{key}": value for key, value in pair_counts(pairs).items()})
    return out


def monitored_vr(points: np.ndarray, timeout_s: float | None = None,
                 include_diagrams: bool = False) -> dict:
    """Run exactly one VR job in one child and poll its peak memory every 50 ms."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(points, dtype=float), allow_pickle=False)
    started = time.perf_counter()
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    diagram_dir = tempfile.TemporaryDirectory(prefix=".stage1_vr_", dir=PROJECT_ROOT) if include_diagrams else None
    diagram_path = Path(diagram_dir.name) / "diagrams.npz" if diagram_dir else None
    command = [sys.executable, "-m", "topovital_stage1.vr_child"]
    if diagram_path is not None:
        command.extend(["--diagrams", str(diagram_path)])
    proc = subprocess.Popen(command,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env)
    child = psutil.Process(proc.pid)
    peak = 0
    status = "success"
    reason = ""
    try:
        proc.stdin.write(base64.b64encode(buffer.getvalue()))
        proc.stdin.close()
        proc.stdin = None
        while proc.poll() is None:
            try:
                info = child.memory_info()
                peak = max(peak, int(getattr(info, "peak_wset", info.rss)))
            except psutil.Error:
                pass
            if peak > MEMORY_LIMIT_BYTES:
                status, reason = "failed_memory", "child_peak_rss_exceeded_6_gb"
                proc.kill()
                break
            if timeout_s is not None and time.perf_counter() - started > timeout_s:
                status, reason = "failed_timeout", "child_timeout"
                proc.kill()
                break
            time.sleep(0.05)
        try:
            info = child.memory_info()
            peak = max(peak, int(getattr(info, "peak_wset", info.rss)))
        except psutil.Error:
            pass
        stdout, stderr = proc.communicate()
        if status == "success":
            if proc.returncode != 0:
                status = "failed_vr"
                reason = stderr.decode("utf-8", errors="replace")[-1000:] or f"child_exit_code_{proc.returncode}"
            else:
                result = json.loads(stdout.decode("utf-8"))
                if diagram_path is not None:
                    with np.load(diagram_path, allow_pickle=False) as saved:
                        result["h0"] = saved["h0"]
                        result["h1"] = saved["h1"]
                peak = max(peak, int((result.pop("child_peak_rss_mb", 0) or 0) * 2**20))
                if peak > MEMORY_LIMIT_BYTES:
                    if diagram_dir is not None:
                        diagram_dir.cleanup()
                    return {"vr_status": "failed_memory",
                            "vr_failure_reason": "child_peak_rss_exceeded_6_gb",
                            "vr_wall_time_s": time.perf_counter() - started,
                            "vr_child_peak_rss_mb": peak / 2**20}
                if diagram_dir is not None:
                    diagram_dir.cleanup()
                return {"vr_status": status, "vr_failure_reason": "", **result,
                        "vr_wall_time_s": time.perf_counter() - started,
                        "vr_child_peak_rss_mb": peak / 2**20}
    except Exception as exc:
        status, reason = "failed_vr", f"{type(exc).__name__}:{exc}"
        if proc.poll() is None:
            proc.kill()
        proc.communicate()
    if diagram_dir is not None:
        diagram_dir.cleanup()
    return {"vr_status": status, "vr_failure_reason": reason,
            "vr_wall_time_s": time.perf_counter() - started,
            "vr_child_peak_rss_mb": peak / 2**20}


def pilot_half(half: np.ndarray) -> dict:
    result = {"pilot_tau": 10, "pilot_m": 5, "pilot_n_points": 300,
              "subsample_rule": "np.linspace(0,n-1,300).round() (pilot pending approval)",
              "r_max_rule": "unset (full Rips complex; upper-bound cost)",
              "vr_max_dimension": 2}
    try:
        vectors = delay_vectors(half, 10, 5)
        idx = uniform_indices(len(vectors))
        result["n_delay_vectors"] = len(vectors)
        result.update(monitored_vr(vectors[idx]))
    except Exception as exc:
        result.update(vr_status="failed_input", vr_failure_reason=f"{type(exc).__name__}:{exc}")
    try:
        result.update(cubical_counts(half))
        result["cubical_status"] = "success"
    except Exception as exc:
        result.update(cubical_status="failed_cubical",
                      cubical_failure_reason=f"{type(exc).__name__}:{exc}")
    return result
