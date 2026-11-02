"""Synthetic Level 3 feature and training invariants."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_analysis import classical_features as cf
from topovital_analysis import level3

ROOT = Path(__file__).resolve().parents[2]
CFG_L3 = load_yaml(ROOT / "configs/stage1_level3.yaml")
CFG_F = load_yaml(ROOT / "configs/stage1_factorial.yaml")
CFG_A = load_yaml(ROOT / "configs/stage1_analysis.yaml")


def test_rhythm_known_rr_and_too_few():
    peaks = np.arange(180, 1800, 360)
    values, reasons = cf.rhythm(np.zeros(1800), peaks)
    assert values["n_beats"] == len(peaks)
    assert values["median_rr_s"] == pytest.approx(1)
    assert np.isnan(values["rr_cv"]) and reasons["rr_cv"] == "nonpositive_ratio"
    values, reasons = cf.rhythm(np.zeros(1800), np.array([100]))
    assert np.isnan(values["median_rr_s"]) and reasons["median_rr_s"] == "too_few_beats"
    irregular = np.array([100, 460, 820, 1100, 1520])
    values, _ = cf.rhythm(np.zeros(1800), irregular)
    assert values["rr_cv"] == pytest.approx(np.log(np.std(np.diff(irregular)) / np.median(np.diff(irregular))))


def test_boundary_uses_premature_beat_after_first_post():
    peaks = np.array([100, 460, 820, 1180, 1740, 1900, 2190, 2300, 2700, 3100])
    values, reasons, latency, _ = cf._boundary_with_latency(np.zeros(3600), filtered=np.zeros(3600), peaks=peaks)
    reference = np.median(np.diff(peaks[:5]))
    assert latency == pytest.approx(100 / 360)
    assert values["min_rr_ratio_post"] == pytest.approx(np.log(110 / reference))
    assert values["compensatory_ratio"] == pytest.approx(np.log((110 + 400) / (2 * reference)))
    assert not reasons
    assert "first_beat_latency_s" not in values


def test_template_pre_reference_and_edge_segments():
    length = 1800
    shape = np.sin(np.linspace(0, np.pi, 253))
    pre = np.zeros(length)
    for p in (300, 700, 1100):
        pre[p-90:p+163] = shape
    peaks = np.array([10, 300, 700, 1100, 1750])
    values, reasons, reference = cf._template_with_reference(pre, filtered=pre, peaks=peaks)
    assert values["min_template_correlation"] == pytest.approx(1)
    assert len(cf.beat_segments(pre, peaks)) == 3
    post = -pre
    post_values, _ = cf.template(post, reference, filtered=post, peaks=peaks)
    assert post_values["min_template_correlation"] == pytest.approx(-1)


def test_dwt_psd_entropy_and_sqi_ranges():
    rng = np.random.default_rng(2)
    t = np.arange(1800) / 360
    sine = np.sin(2*np.pi*10*t)
    d, _ = cf.dwt(sine)
    p, _ = cf.psd(sine)
    e_sine, _ = cf.spectral_entropy(sine)
    e_noise, _ = cf.spectral_entropy(rng.standard_normal(1800))
    assert sum(d.values()) == pytest.approx(1)
    assert sum(v for k, v in p.items() if k.startswith("relative_power")) == pytest.approx(1)
    assert p["relative_power_5_15"] > .9
    assert 0 <= e_sine["normalized_entropy"] < e_noise["normalized_entropy"] <= 1


def test_sqi_with_stage0_config_and_synthetic_pulse_train():
    cfg = load_yaml(ROOT / "configs/stage0.yaml")
    samples = np.arange(5 * 360)
    signal = 0.02 * np.sin(2 * np.pi * samples / 360)
    for peak in range(180, len(samples), 360):
        signal += np.exp(-0.5 * ((samples - peak) / 8) ** 2)
    values, _ = cf.sqi(signal, cfg)
    assert set(values) == set(cf.GROUPS["sqi"])
    assert all(np.isfinite(value) for value in values.values())


def test_log_diff_and_indicators(monkeypatch):
    reasons = {}
    assert np.isnan(cf._log_ratio(0, reasons, "x")) and reasons["x"] == "nonpositive_ratio"
    assert np.isnan(cf._log_ratio(np.inf, reasons, "x"))
    assert cf._log_ratio(np.e, reasons, "x") == pytest.approx(1)
    monkeypatch.setattr(cf, "filtered_peaks", lambda x: (np.asarray(x), np.array([250, 650, 1050, 1450])))
    monkeypatch.setattr(cf, "qrs", lambda *a, **k: ({n: 1. for n in cf.GROUPS["qrs"]}, {}))
    monkeypatch.setattr(cf, "sqi", lambda *a, **k: ({n: 1. for n in cf.GROUPS["sqi"]}, {}))
    pre = np.sin(np.arange(1800) / 20)
    a = cf._feature_job(({k: "x" for k in cf.KEYS}, pre, pre, pre, pre, {}))[0]
    b = cf._feature_job(({k: "x" for k in cf.KEYS}, pre, np.zeros(1800), pre, np.zeros(1800), {}))[0]
    for col in (c for c in a if c.endswith("__pre")):
        np.testing.assert_allclose(a[col], b[col], equal_nan=True)
    assert not np.isnan(b["amplitude__min__diff"])
    assert b["template_post_missing"] == int(any(np.isnan(b[f"template__{n}__post"]) for n in cf.GROUPS["template"]))
    assert np.isnan(b["template__min_template_correlation__diff"])


def test_imputation_training_only():
    train = np.array([[1., np.nan], [3., np.nan]])
    test = np.array([[np.nan, 1.e30]])
    log = []
    a, b = level3.impute_train(train, test, ["x", "y"], "fold", log)
    np.testing.assert_allclose(a, [[1, 0], [3, 0]])
    np.testing.assert_allclose(b, [[2, 1.e30]])
    assert log[0]["median"] == 2 and log[1]["all_missing"]
    assert [row["n_missing_test"] for row in log] == [1, 0]


def _efficiency_inputs(monkeypatch):
    windows = [{"event_id": s, "subject_id": s, "condition": condition,
                "noise_type": "", "snr_target": None}
               for s in ("A", "B", "C") for condition in ("C0", "C1")]
    classical = pd.DataFrame([{**{key: w[key] for key in level3.KEYS},
        "rhythm__x": float(i) if i != 2 else np.nan,
        "boundary__x": float(i + 10) if i != 3 else np.nan,
        "boundary_missing": 0} for i, w in enumerate(windows)])
    cfg_a = deepcopy(CFG_A)
    cfg_a["level2"]["classifier"]["c_grid"] = [1, 2]
    cfg_a["level2"]["persistence_image"]["candidates"] = {"resolution": [10], "sigma_pixels": [1, 2]}
    monkeypatch.setattr(level3, "estimate_regions", lambda *a, **k: {})
    monkeypatch.setattr(level3, "_region_rows", lambda *a: [])
    monkeypatch.setattr(level3, "_range_log", lambda *a: [])
    monkeypatch.setattr(level3, "_arrays", lambda *a: ())
    monkeypatch.setattr(level3, "factorial_weights", lambda *a: np.ones(4))
    monkeypatch.setattr(level3, "weight_checks", lambda *a: {})
    return windows, classical, cfg_a


def test_cv_caches_identical_pi_and_imputes_classical_once_per_fold(monkeypatch):
    windows, classical, cfg_a = _efficiency_inputs(monkeypatch)
    matrix_calls, imputer_calls, fit_inputs = [], [], []

    def matrix(rows, indices, model, regions, candidate, reps):
        assert model == "pi_only"
        matrix_calls.append((tuple(indices), candidate[:2]))
        return np.asarray([[i + candidate[0] / 10, i + candidate[1] / 10] for i in indices])

    original_imputer = level3.impute_train
    def imputer(a, b, columns, fold, logs):
        assert columns == ["rhythm__x", "boundary__x", "boundary_missing"]
        assert a.shape[1] == b.shape[1] == len(columns)
        imputer_calls.append(fold)
        return original_imputer(a, b, columns, fold, logs)

    def fit(a, y, weights, b, c, fits, *, label=None):
        fit_inputs.append((label, a.copy(), b.copy()))
        fits.append({"wall_time_s": 0., "convergence_warnings": 0})
        return np.zeros(len(b)), None

    monkeypatch.setattr(level3, "_feature_matrix", matrix)
    monkeypatch.setattr(level3, "impute_train", imputer)
    monkeypatch.setattr(level3, "_fit_score", fit)
    monkeypatch.setattr(level3, "stratified_auroc", lambda *a: .5)
    logs = []
    level3._cross_validate_level3(windows, classical, CFG_L3, cfg_a, ["A", "B", "C"],
        [], {}, [], [], [], logs, "outer_X", lambda *a, **k: None)
    assert len(imputer_calls) == 3
    assert len(logs) == 3 * 3
    assert {(r["fold"], r["column"]) for r in logs} == {
        (fold, col) for fold in imputer_calls for col in ("rhythm__x", "boundary__x", "boundary_missing")}
    assert len(matrix_calls) == 3 * 2 * 2
    for held in ("A", "B", "C"):
        train = [i for i, w in enumerate(windows) if w["subject_id"] != held]
        for sigma in (1, 2):
            label = f"outer_X__inner_{held}:ph_only:R10:sigma{float(sigma)}:C1.0"
            cached = next(a for name, a, _ in fit_inputs if name == label)
            fresh = np.asarray([[i + 1, i + sigma / 10] for i in train])
            np.testing.assert_array_equal(cached, fresh)


def test_refit_saves_classical_medians_by_name(monkeypatch, tmp_path):
    windows, classical, cfg_a = _efficiency_inputs(monkeypatch)
    matrix_calls = []
    def matrix(rows, indices, model, regions, candidate, reps):
        matrix_calls.append((tuple(indices), candidate[:2]))
        return np.asarray([[i + .5] for i in indices])

    class Estimator:
        def __init__(self, width):
            self.named_steps = {
                "standardscaler": type("Scaler", (), {"mean_": np.zeros(width), "scale_": np.ones(width)})(),
                "logisticregression": type("Classifier", (), {"coef_": np.zeros((1, width)),
                                                       "intercept_": np.zeros(1)})()}

    def fit(a, y, weights, b, c, fits, *, label=None):
        return np.zeros(len(b)), Estimator(a.shape[1])

    monkeypatch.setattr(level3, "_feature_matrix", matrix)
    monkeypatch.setattr(level3, "_fit_score", fit)
    chosen = ((10, 1, 1), .5, .5, .5)
    selected = {name: chosen for name in ("ph_only", "classical_plus_ph", "classical_plus_ph_no_boundary")}
    logs = []
    level3._refit_level3(windows, classical, [0, 1, 2, 3], [4, 5], selected, CFG_L3,
                         cfg_a, [], {}, [], [], logs, "refit", save=tmp_path)
    assert len(logs) == 3
    assert len(matrix_calls) == 2
    expected = {r["column"]: r["median"] for r in logs}
    assert expected["rhythm__x"] == 1 and expected["boundary__x"] == 11
    for model in selected:
        with np.load(tmp_path / "final_models" / f"{model}.npz") as saved:
            names = saved["imputation_columns"].tolist()
            assert names == level3.model_columns(classical, model, CFG_L3)
            np.testing.assert_array_equal(saved["imputation_medians"], [expected[name] for name in names])
            assert saved["feature_names"].tolist() == (["pi_0"] + names)


def test_model_columns_and_early_gates(tmp_path):
    assert len(level3._models(CFG_L3)) == 5
    frame = pd.DataFrame({"rhythm__n_beats__pre": [2], "boundary__rr_cross_ratio": [1],
                          "boundary_missing": [0], "amplitude__min__pre": [0]})
    assert level3.model_columns(frame, "ph_only", CFG_L3) == []
    assert level3.model_columns(frame, "classical_only_no_boundary", CFG_L3) == [
        "rhythm__n_beats__pre", "amplitude__min__pre"]
    events = pd.DataFrame({"event_id": ["x"], "subject_id": ["S105"], "split": ["development"]})
    with pytest.raises(BlockedError, match="output directory"):
        level3.run_level3(events, CFG_L3, CFG_F, CFG_A, tmp_path, tmp_path / "wrong", pilot=True)
    cfg = deepcopy(CFG_L3)
    cfg["approved_for_full_run"] = False
    with pytest.raises(BlockedError, match="full run"):
        level3.run_level3(events, cfg, CFG_F, CFG_A, tmp_path,
                          tmp_path / "outputs/stage1_analysis/development/level3")
    with pytest.raises(BlockedError, match="stop rule"):
        source = tmp_path / "outputs/stage1_analysis/development/level3_features"
        source.mkdir(parents=True)
        (source / "extraction_summary.json").write_text('{"stop_rule_passed": false}')
        (source / "classical_features.parquet").write_bytes(b"placeholder")
        level3.run_level3(events, CFG_L3, CFG_F, CFG_A, tmp_path,
                          tmp_path / "outputs/stage1_analysis/development/level3_pilot", pilot=True)


def test_paired_bootstrap_uses_same_subjects():
    cells = [("bw", 24)]
    rows = []
    for subject in ("A", "B"):
        for condition, score in (("C0", .1), ("C1", .9), ("C2", .2), ("C3", .8)):
            rows.append({"subject_id": subject, "event_id": subject, "condition": condition,
                         "noise_type": "bw" if condition in ("C2", "C3") else "",
                         "snr_target": 24 if condition in ("C2", "C3") else None,
                         "score": score, "threshold_primary": .5, "threshold_c1": .5})
    a = pd.DataFrame(rows)
    b = a.copy()
    a.loc[a.subject_id.eq("A") & a.condition.eq("C0"), "score"] = .6
    b.loc[b.subject_id.eq("B") & b.condition.eq("C0"), "score"] = .6
    draws = level3._paired_draws(a, b, ["A", "B"], cells, "primary", "subject_equal_weight", 20, 7,
                                  ["c0_fpr"])
    rng = np.random.default_rng(7)
    expected = [np.mean([1 if i == 0 else -1 for i in rng.integers(0, 2, 2)]) for _ in range(20)]
    np.testing.assert_allclose(draws["c0_fpr"], expected, atol=1e-12)


def test_factorial_manifest_checks_exact_relative_paths(tmp_path):
    base = tmp_path / "outputs/stage1_analysis/development"
    source = base / "factorial"
    source.mkdir(parents=True)
    digests = []
    for name in ("oof_predictions.parquet", "inner_cv_selection.csv"):
        data = name.encode()
        (source / name).write_bytes(data)
        digests.append(f"{hashlib.sha256(data).hexdigest()} *factorial/{name}")
    manifest = base / "hashes/factorial_outputs.sha256"
    manifest.parent.mkdir()
    manifest.write_text("\n".join(["0"*64 + " *factorial_pilot/oof_predictions.parquet", *digests]))
    assert level3._verify_manifest(tmp_path, CFG_L3) == source
    manifest.write_text("\n".join(["0"*64 + " *factorial/oof_predictions.parquet", digests[1]]))
    with pytest.raises(BlockedError, match="SHA256 mismatch"):
        level3._verify_manifest(tmp_path, CFG_L3)


def test_ph_only_cv_matches_factorial_pi_only_on_synthetic_windows(monkeypatch):
    from topovital_analysis import factorial

    cfg_a = deepcopy(CFG_A)
    cfg_a["level2"]["classifier"]["c_grid"] = [1]
    cfg_a["level2"]["persistence_image"]["candidates"] = {"resolution": [10], "sigma_pixels": [1]}
    cfg_a["level2"]["features"]["models"] = ["pi_only"]
    cells = [(n, s) for n in cfg_a["scope"]["regime1_cells"]["noise_types"]
             for s in cfg_a["scope"]["regime1_cells"]["snr_db"]]
    windows = []
    for subject in ("A", "B", "C"):
        for condition, noise, snr in [("C0", "", None), ("C1", "", None)] + [
                (condition, noise, snr) for noise, snr in cells for condition in ("C2", "C3")]:
            windows.append({"event_id": subject, "subject_id": subject, "condition": condition,
                            "noise_type": noise, "snr_target": snr})
    classical = pd.DataFrame([{**{k: w[k] for k in level3.KEYS},
                               "rhythm__n_beats__pre": 2., "boundary__rr_cross_ratio": 1.,
                               "boundary_missing": 0} for w in windows])
    def matrix(rows, indices, model, regions, candidate, reps):
        if model == "amplitude_only":
            return np.zeros((len(indices), 1))
        assert model == "pi_only"
        return np.asarray([[.9 if rows[i]["condition"] in ("C1", "C3") else .1] for i in indices])

    def fit(x_train, y, weights, x_test, c, fits, *, label=None):
        fits.append({"wall_time_s": 0., "convergence_warnings": 0})
        return x_test[:, 0], None

    for module in (factorial, level3):
        monkeypatch.setattr(module, "estimate_regions", lambda *a, **k: {})
        monkeypatch.setattr(module, "_region_rows", lambda *a: [])
        monkeypatch.setattr(module, "_range_log", lambda *a: [])
        monkeypatch.setattr(module, "_feature_matrix", matrix)
        monkeypatch.setattr(module, "_fit_score", fit)
    old, _, _ = factorial._cross_validate_factorial(windows, cfg_a, ["A", "B", "C"], [], {},
        [], [], [], "outer_X", lambda *a, **k: None)
    new, _, _ = level3._cross_validate_level3(windows, classical, CFG_L3, cfg_a,
        ["A", "B", "C"], [], {}, [], [], [], [], "outer_X", lambda *a, **k: None)
    assert new["ph_only"] == old["pi_only"]
