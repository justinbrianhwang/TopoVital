"""Apply the five frozen Development models without fitting on Test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from topovital.cohort import BlockedError
from topovital_analysis.level3 import KEYS
from topovital_analysis.persistence_images import pi_features

from .gate import _sha


def load_models(root: Path, settings: dict, manifest: dict) -> tuple[dict, dict]:
    root = Path(root)
    record_path = root / settings["selection_record"]
    relative = record_path.relative_to(root).as_posix()
    if _sha(record_path) != manifest["files"].get(relative):
        raise BlockedError("Development selection record SHA256 mismatch")
    selection = json.loads(record_path.read_text(encoding="utf-8"))
    models = {}
    for name in selection:
        path = root / settings["models_from"] / f"{name}.npz"
        relative = path.relative_to(root).as_posix()
        if _sha(path) != manifest["files"].get(relative):
            raise BlockedError(f"Final model SHA256 mismatch: {name}")
        with np.load(path, allow_pickle=False) as saved:
            models[name] = {k: saved[k].copy() for k in saved.files}
        candidate = models[name]["candidate"]
        chosen = selection[name]
        if (not np.isclose(candidate[2], chosen["C"]) or
                (chosen["resolution"] is not None and
                 not np.isclose(candidate[0], chosen["resolution"])) or
                (chosen["sigma_pixels"] is not None and
                 not np.isclose(candidate[1], chosen["sigma_pixels"]))):
            raise BlockedError(f"Final model candidate mismatch: {name}")
    if len(models) != 5:
        raise BlockedError("Expected five final models")
    return models, selection


def design_matrix(windows: list[dict], classical: pd.DataFrame, saved: dict) -> np.ndarray:
    if len(windows) != len(classical):
        raise BlockedError("Classical and PH window counts differ")
    names = list(map(str, saved["feature_names"]))
    classical_names = list(map(str, saved["imputation_columns"]))
    medians = np.asarray(saved["imputation_medians"], dtype=float)
    if len(classical_names) != len(medians):
        raise BlockedError("Saved imputation schema mismatch")
    if not set(classical_names).issubset(classical):
        raise BlockedError("Missing classical model column")
    classical_matrix = classical[classical_names].to_numpy(dtype=float, copy=True)
    missing = ~np.isfinite(classical_matrix)
    if missing.any():
        classical_matrix[missing] = np.broadcast_to(medians, classical_matrix.shape)[missing]
    if np.isfinite(saved["candidate"][0]):
        representations = list(map(str, saved["region_names"]))
        regions = {rep: tuple(None if np.isnan(x) else float(x) for x in saved["regions"][i])
                   for i, rep in enumerate(representations)}
        resolution = int(saved["candidate"][0])
        width = sum(3 * (resolution if rep == "vr_h0" else resolution * resolution)
                    for rep in representations)
        pi_matrix = np.stack([pi_features(w, regions, resolution,
                                         float(saved["candidate"][1]), representations)
                              if w.get("pi_failure_reason", "") == "" else
                              np.full(width, np.nan)
                              for w in windows])
        matrix = np.concatenate((pi_matrix, classical_matrix), axis=1)
        expected = [*(f"pi_{i}" for i in range(pi_matrix.shape[1])), *classical_names]
    else:
        matrix = classical_matrix
        expected = classical_names
    if names != expected or matrix.shape[1] != len(saved["scaler_mean"]):
        raise BlockedError("Saved feature order differs from Test design matrix")
    return matrix


def score_matrix(matrix: np.ndarray, saved: dict) -> np.ndarray:
    mean = np.asarray(saved["scaler_mean"], dtype=float)
    scale = np.asarray(saved["scaler_scale"], dtype=float)
    coef = np.asarray(saved["logistic_coef"], dtype=float).reshape(-1)
    intercept = float(np.asarray(saved["logistic_intercept"]).reshape(-1)[0])
    if matrix.shape[1] != len(mean) or len(scale) != len(mean) or len(coef) != len(mean):
        raise BlockedError("Saved estimator dimension mismatch")
    score = np.full(len(matrix), np.nan)
    valid = np.isfinite(matrix).all(axis=1)
    score[valid] = expit(((matrix[valid] - mean) / scale) @ coef + intercept)
    return score


def apply_models(windows: list[dict], classical: pd.DataFrame, models: dict,
                 selection: dict, *, split: str) -> pd.DataFrame:
    if len(windows) != len(classical):
        raise BlockedError("Classical and diagram windows are not aligned")
    for i, window in enumerate(windows):
        if any((pd.isna(classical.iloc[i][key]) and window[key] is not None) or
               (not pd.isna(classical.iloc[i][key]) and
                str(classical.iloc[i][key]) != str(window[key]))
               for key in ("event_id", "subject_id", "condition", "noise_type")) or not (
                   (pd.isna(classical.iloc[i]["snr_target"]) and window["snr_target"] is None) or
                   classical.iloc[i]["snr_target"] == window["snr_target"]):
            raise BlockedError("Classical and diagram window keys differ")
    rows = []
    for name, saved in models.items():
        matrix = design_matrix(windows, classical, saved)
        scores = score_matrix(matrix, saved)
        for window, score in zip(windows, scores):
            if window.get("failure_reason"):
                score = np.nan
            rows.append({**{k: window[k] for k in KEYS}, "split": split, "fold": "final_test",
                         "model": name, "score": float(score),
                         "threshold_primary": float(selection[name]["threshold_primary"]),
                         "threshold_c1": float(selection[name]["threshold_c1"]),
                         "status": "ok" if np.isfinite(score) else "not_evaluable",
                         "reason": "" if np.isfinite(score) else
                         (window.get("failure_reason") or window.get("pi_failure_reason") or
                          "classical_feature_unavailable")})
    return pd.DataFrame(rows)
