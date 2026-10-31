"""Hashing, atomic writes, config loading, run IDs and resource profiling."""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode("utf-8"))


def code_hash() -> str:
    """SHA256 over the package source files (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(cfg_hash: str) -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg_hash[:8]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def atomic_write_yaml(path: Path, obj) -> None:
    atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def atomic_write_csv(path: Path, df) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    atomic_write_text(path, buf.getvalue())


def atomic_write_parquet(path: Path, df) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), sink)
    _atomic_write_bytes(path, sink.getvalue().to_pybytes())


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MiB (None if unavailable)."""
    try:
        import psutil

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows
        if peak is not None:
            return peak / 2**20
    except Exception:
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 1024
    except Exception:
        return None


RESOURCE_FIELDS = ["run_id", "started_utc", "command", "status", "exit_code", "wall_time_s",
                   "peak_rss_mb", "cache_used", "n_jobs", "thread_env", "note"]


@contextmanager
def profiled(out_dir: Path, run_id: str, command: str, n_jobs: int = 1):
    """Record wall time / peak RSS / status of a command to resource_profile.csv."""
    rec = {"cache_used": False, "note": "", "status": "success", "exit_code": 0}
    started = utc_now()
    t0 = time.perf_counter()
    try:
        yield rec
    except SystemExit as e:
        rec["exit_code"] = e.code if isinstance(e.code, int) else 1
        if rec["exit_code"] != 0 and rec["status"] == "success":
            rec["status"] = "failed"
        raise
    except BaseException as e:
        rec["status"] = "failed"
        rec["exit_code"] = 1
        rec["note"] = (rec["note"] + f" | {type(e).__name__}: {e}").strip(" |")
        raise
    finally:
        row = {
            "run_id": run_id, "started_utc": started, "command": command,
            "status": rec["status"], "exit_code": rec["exit_code"],
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "peak_rss_mb": None if (p := peak_rss_mb()) is None else round(p, 1),
            "cache_used": rec["cache_used"], "n_jobs": n_jobs,
            "thread_env": ";".join(f"{v}={os.environ.get(v, '')}" for v in THREAD_ENV_VARS),
            "note": rec["note"],
        }
        path = out_dir / "resource_profile.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=RESOURCE_FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerow(row)
        atomic_write_text(path, buf.getvalue())
