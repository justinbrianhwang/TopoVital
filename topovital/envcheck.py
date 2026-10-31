"""Record the actual execution environment (env-check)."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from topovital.util import THREAD_ENV_VARS, PROJECT_ROOT, utc_now

PACKAGES = {  # import name -> distribution name
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "pyarrow": "pyarrow",
    "yaml": "PyYAML", "wfdb": "wfdb", "neurokit2": "neurokit2", "matplotlib": "matplotlib",
    "psutil": "psutil", "pytest": "pytest", "gudhi": "gudhi", "sklearn": "scikit-learn",
}
REQUIRED = {"numpy", "scipy", "pandas", "pyarrow", "yaml", "wfdb", "psutil"}


def _optional_gpu() -> dict:
    """Optional GPU probe. Absence or failure is never an error."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"nvidia_smi_found": False, "gpus": [], "required": False}
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        gpus = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return {"nvidia_smi_found": True, "gpus": gpus, "required": False}
    except Exception as e:  # pragma: no cover
        return {"nvidia_smi_found": True, "gpus": [], "probe_error": repr(e), "required": False}


def collect() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    du = shutil.disk_usage(PROJECT_ROOT)
    pkgs = {}
    for mod, dist in PACKAGES.items():
        try:
            importlib.import_module(mod)
            pkgs[dist] = {"import_ok": True, "version": metadata.version(dist)}
        except Exception as e:
            pkgs[dist] = {"import_ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "collected_utc": utc_now(),
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "python": {"version": sys.version, "executable": sys.executable,
                   "implementation": platform.python_implementation()},
        "cpu": {"physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "processor": platform.processor()},
        "memory_bytes": {"total": vm.total, "available": vm.available},
        "disk_bytes_at_project": {"total": du.total, "free": du.free},
        "packages": pkgs,
        "execution_resource_settings": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
        "gpu_optional_probe": _optional_gpu(),
    }


def missing_required(env: dict) -> list[str]:
    return [d for m, d in PACKAGES.items() if m in REQUIRED and not env["packages"][d]["import_ok"]]
