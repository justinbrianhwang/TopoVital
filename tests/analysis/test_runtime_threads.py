"""BLAS thread settings for the Development analysis CLI and saved results."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from conftest import requires_data
from topovital.cohort import BlockedError
from topovital_analysis import runtime


ROOT = Path(__file__).resolve().parents[2]
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
LEVEL3_OOF = ROOT / "outputs/stage1_analysis/development/level3/oof_predictions.parquet"


def _set_thread_env(monkeypatch):
    for name in THREAD_VARS:
        monkeypatch.setenv(name, "1")


def _cli_env():
    env = os.environ.copy()
    for name in THREAD_VARS:
        env.pop(name, None)
    return env


def test_verify_blas_threads_accepts_one(monkeypatch):
    _set_thread_env(monkeypatch)
    monkeypatch.setattr(runtime, "threadpool_info", lambda: [
        {"internal_api": "openblas", "num_threads": 1},
        {"internal_api": "openmp", "num_threads": 1},
    ])
    assert runtime.verify_blas_threads() == {
        "env": {name: "1" for name in THREAD_VARS},
        "threadpools": [("openblas", 1), ("openmp", 1)],
    }


def test_verify_blas_threads_rejects_loaded_pool(monkeypatch):
    _set_thread_env(monkeypatch)
    monkeypatch.setattr(runtime, "threadpool_info", lambda: [
        {"internal_api": "openblas", "num_threads": 12},
    ])
    with pytest.raises(BlockedError, match="openblas=12"):
        runtime.verify_blas_threads()


def test_verify_blas_threads_rejects_env(monkeypatch):
    _set_thread_env(monkeypatch)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setattr(runtime, "threadpool_info", lambda: [
        {"internal_api": "openblas", "num_threads": 1},
    ])
    with pytest.raises(BlockedError, match="MKL_NUM_THREADS='4'"):
        runtime.verify_blas_threads()


def test_cli_blocks_explicit_four_threads():
    env = _cli_env()
    env["OPENBLAS_NUM_THREADS"] = "4"
    result = subprocess.run(
        [sys.executable, "-m", "topovital_analysis", "level1", "--workers", "0"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3
    assert "Blocked: BLAS thread check" in result.stderr
    assert "OPENBLAS_NUM_THREADS" in result.stderr


def test_cli_defaults_to_one_before_imports():
    result = subprocess.run(
        [sys.executable, "-m", "topovital_analysis", "level1", "--workers", "0"],
        cwd=ROOT, env=_cli_env(), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3
    assert "Blocked: --workers must be positive" in result.stderr
    assert "BLAS thread check" not in result.stderr


@requires_data
@pytest.mark.skipif(not LEVEL3_OOF.exists(), reason="Level 3 OOF reference missing")
def test_s210_one_thread_level3_ph_reproduction():
    """The version 1 reference (code hash 2f6a3586…) used 1 BLAS thread;
    the factorial pi_only reference used 12 threads and is intentionally not expected to match.
    """
    script = textwrap.dedent("""
        import json
        import topovital_analysis.__main__  # set thread defaults before NumPy imports

        import numpy as np
        import pandas as pd
        from topovital.util import PROJECT_ROOT, load_yaml
        from topovital_stage1.core import load_dev_events
        from topovital_analysis.factorial import _arrays, factorial_weights
        from topovital_analysis.level2 import _load_windows, _feature_matrix, _fit_score
        from topovital_analysis.persistence_images import estimate_regions
        from topovital_analysis.runtime import verify_blas_threads

        verify_blas_threads()
        root = PROJECT_ROOT
        cfg = load_yaml(root / "configs/stage1_analysis.yaml")
        events = load_dev_events(root / "outputs/development/event_audit_dev.parquet",
                                 root / "outputs/split_assignment.csv")
        windows = _load_windows(events, cfg, root)
        held = "S210"
        reference = pd.read_parquet(root / "outputs/stage1_analysis/development/level3/oof_predictions.parquet")
        reference = reference[(reference.model == "ph_only") & (reference.subject_id == held)]
        assert len(reference) == 950
        candidate = (int(reference.resolution.iloc[0]),
                     float(reference.sigma_pixels.iloc[0]), float(reference.C.iloc[0]))
        assert candidate == (10, 2.0, 10.0)
        reps = list(cfg["inputs"]["representations"])
        quantiles = cfg["level2"]["persistence_image"]["range_quantiles"]
        train = [i for i, window in enumerate(windows) if window["subject_id"] != held]
        test = [i for i, window in enumerate(windows) if window["subject_id"] == held]
        assert len(test) == 950
        regions = estimate_regions(
            windows, [i for i in train if windows[i]["condition"] in ("C0", "C1")],
            reps, quantiles, excluded_subject=held,
        )
        labels = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
        weights = factorial_weights(*_arrays(windows, train))
        x_train = _feature_matrix(windows, train, "pi_only", regions, candidate, reps)
        x_test = _feature_matrix(windows, test, "pi_only", regions, candidate, reps)
        scores, _ = _fit_score(x_train, labels, weights, x_test, candidate[2], [])
        keys = ["event_id", "condition", "noise_type", "snr_target"]
        fitted = pd.DataFrame([{**{key: windows[i][key] for key in keys}, "refit_score": score}
                               for i, score in zip(test, scores)])
        joined = fitted.merge(reference, on=keys, validate="one_to_one")
        assert len(joined) == 950
        print(json.dumps({"rows": len(joined),
                          "max_abs_diff": float((joined.refit_score - joined.score).abs().max())}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=_cli_env(),
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    import json
    comparison = json.loads(result.stdout)
    assert comparison["rows"] == 950
    assert comparison["max_abs_diff"] <= 1e-12
