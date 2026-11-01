from __future__ import annotations

import json
import inspect
from pathlib import Path
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_analysis.level1 import validate_events
from topovital_analysis.level2 import (_fit_score, assert_training_subjects, candidate_grid,
                                       choose_candidate, fit_weights, require_full_approval, sample_weights,
                                       threshold_from_inner, run_level2)
from topovital_analysis.persistence_images import (amplitude_features, estimate_regions,
                                                    out_of_range, persistence_image)


CONFIG = load_yaml(Path(__file__).parents[2] / "configs/stage1_analysis.yaml")


def test_integrated_gaussian_mass_and_outside_tail():
    region = (0., 10., 0., 10.)
    pair = np.array([[5., 10.]])  # persistence 5, weight 0.5
    image = persistence_image(pair, region, 20, .1)
    assert image.shape == (20, 20)
    assert image.sum() == pytest.approx(.5, abs=1e-6)
    outside = np.array([[-2., 3.]])
    image = persistence_image(outside, region, 10, 1.)
    expected = .5 * (ndtr((10 + 2) / 1) - ndtr((0 + 2) / 1)) * (ndtr((10 - 5) / 1) - ndtr((0 - 5) / 1))
    assert image.sum() == pytest.approx(expected, rel=1e-12)
    assert out_of_range(outside, region) == (1, 5.)


def test_vr_h0_one_dimensional_and_no_birth_clipping():
    region = (None, None, 0., 10.)
    image = persistence_image([[0., 5.]], region, 20, .2)
    assert image.shape == (20,)
    assert image.sum() == pytest.approx(.5, abs=1e-6)
    assert out_of_range([[100., 105.]], region) == (0, 0.)


def _window(subject, condition, birth):
    pair = np.array([[birth, birth + 1.]])
    return {"subject_id": subject, "condition": condition,
            "pre": {"cub_sub_h0": pair, "vr_h0": np.array([[0., 1.]])},
            "post": {"cub_sub_h0": pair, "vr_h0": np.array([[0., 1.]])}}


def test_region_excludes_held_subject():
    windows = [_window("A", "C0", 0), _window("A", "C1", 1),
               _window("B", "C0", 1000), _window("B", "C1", 2000)]
    quantiles = CONFIG["level2"]["persistence_image"]["range_quantiles"]
    region = estimate_regions(windows, [0, 1], ["cub_sub_h0", "vr_h0"], quantiles, excluded_subject="B")
    assert region["cub_sub_h0"][1] < 10
    with pytest.raises(AssertionError, match="held-out"):
        estimate_regions(windows, [0, 1, 2], ["cub_sub_h0"], quantiles, excluded_subject="B")
    with pytest.raises(AssertionError, match="held-out"):
        assert_training_subjects(["A", "B"], "B")


def test_subject_class_weights():
    subjects = ["A"] * 4 + ["B"] * 5
    conditions = ["C1", "C2", "C2", "C2", "C1", "C1", "C2", "C2", "C2"]
    weights = sample_weights(subjects, conditions)
    assert weights.sum() == pytest.approx(1)
    for subject in ("A", "B"):
        for condition in ("C1", "C2"):
            mask = (np.array(subjects) == subject) & (np.array(conditions) == condition)
            assert weights[mask].sum() == pytest.approx(.25)


def test_fit_weights_have_mean_one_and_preserve_ratios():
    weights = sample_weights(["A", "A", "A", "B", "B"], ["C1", "C2", "C2", "C1", "C2"])
    scaled = fit_weights(weights)
    np.testing.assert_allclose(scaled, weights * len(weights))
    assert scaled.mean() == pytest.approx(1)
    np.testing.assert_allclose(scaled / scaled[0], weights / weights[0])


def test_fit_receives_mean_one_weights(monkeypatch):
    from topovital_analysis import level2

    model = Mock()
    model.predict_proba.return_value = np.array([[.25, .75]])
    monkeypatch.setattr(level2, "make_pipeline", lambda *args: model)
    weights = sample_weights(["A", "A", "A", "B", "B"], ["C1", "C2", "C2", "C1", "C2"])
    _fit_score(np.arange(5).reshape(-1, 1), np.array([1, 0, 0, 1, 0]), weights,
               np.array([[0]]), 1., [])
    model.fit.assert_called_once()
    received = model.fit.call_args.kwargs["logisticregression__sample_weight"]
    assert received.mean() == pytest.approx(1)
    np.testing.assert_allclose(received, weights * len(weights))


def test_scaler_is_fit_on_training_rows_only():
    x_train = np.array([[0.], [1.], [2.], [3.]])
    y_train = np.array([0, 0, 1, 1])
    _, model = _fit_score(x_train, y_train, np.ones(4) / 4, np.array([[1000.]]), 1., [])
    assert model.named_steps["standardscaler"].mean_[0] == pytest.approx(1.5)


def test_candidate_tie_break_and_threshold():
    scores = {(20, 1., 1.): .8, (10, 1., 1.): .8, (10, 2., 1.): .8,
              (10, 2., .1): .8, (10, 2., 10.): .79}
    assert choose_candidate(scores, CONFIG["level2"]["cross_validation"]["tie_break"]) == (10, 2., .1)
    assert len(candidate_grid(CONFIG, "pi_only")) == 20
    assert len(candidate_grid(CONFIG, "amplitude_only")) == 5
    frame = pd.DataFrame({"subject_id": ["A", "A", "B", "B"],
                          "condition": ["C1"] * 4, "score": [2., 3., 4., 5.]})
    assert threshold_from_inner(frame, .5) == 4.


def test_amplitude_pre_post_delta():
    window = {"amplitude": {"pre": (-2., 3.), "post": (1., 5.)}}
    assert amplitude_features(window).tolist() == [-2., 3., 5., 1., 5., 4., 3., 2., -1.]


def test_full_gate_and_test_subject():
    cfg = deepcopy(CONFIG)
    cfg["level2"]["approved_for_full_run"] = False
    with pytest.raises(BlockedError, match="full run"):
        require_full_approval(cfg, pilot=False)
    require_full_approval(cfg, pilot=True)
    cfg["level2"]["approved_for_full_run"] = True
    assert require_full_approval(cfg, pilot=False) is None
    with pytest.raises(BlockedError, match="Test"):
        validate_events(pd.DataFrame({"event_id": ["x"], "subject_id": ["T"], "split": ["test"]}))


def test_synthetic_pilot_output_and_fold_isolation(tmp_path, monkeypatch):
    from topovital_analysis import level2

    cfg = deepcopy(CONFIG)
    cfg["scope"]["events"] = 10
    cfg["scope"]["regime1_cells"] = {"noise_types": ["bw"], "snr_db": [12]}
    cfg["level2"]["persistence_image"]["candidates"] = {"resolution": [2], "sigma_pixels": [1]}
    cfg["level2"]["classifier"]["c_grid"] = [1]
    cfg["level2"]["reporting"]["bootstrap"]["resamples"] = 3
    config_path = tmp_path / "configs/stage1_analysis.yaml"
    config_path.parent.mkdir()
    config_path.write_bytes((Path(__file__).parents[2] / "configs/stage1_analysis.yaml").read_bytes())
    windows = []
    for s in range(10):
        for condition, offset in (("C0", 0.), ("C1", 2.), ("C2", .2), ("C3", 2.2)):
            birth = offset + s * .01
            diagram = {rep: np.array([[birth if rep != "vr_h0" else 0.,
                                       birth + 1. if rep != "vr_h0" else 1.]])
                       for rep in cfg["inputs"]["representations"]}
            windows.append({"event_id": f"e{s}", "subject_id": f"S{s}", "condition": condition,
                            "noise_type": "bw" if condition in ("C2", "C3") else "",
                            "snr_target": 12 if condition in ("C2", "C3") else None,
                            "pre": diagram, "post": diagram,
                            "amplitude": {"pre": (birth, birth + 2.), "post": (birth, birth + 2.)}})
    monkeypatch.setattr(level2, "_load_windows", lambda *args: windows)
    shared_calls = []
    shared = level2._cross_validate_candidates
    def track_shared(*args, **kwargs):
        shared_calls.append(args[9])
        return shared(*args, **kwargs)
    monkeypatch.setattr(level2, "_cross_validate_candidates", track_shared)
    events = pd.DataFrame({"event_id": [f"e{s}" for s in range(10)],
                           "subject_id": [f"S{s}" for s in range(10)], "split": ["development"] * 10})
    output = tmp_path / "output"
    summary = run_level2(events, cfg, tmp_path, output, pilot=True)
    assert summary["fits"] == 30  # 9 inner subjects x 3 models x 1 candidate, plus 3 refits
    assert summary["sample_weight_scale"] == "mean_1 (relative weights from sample_weights)"
    assert json.loads((output / "run_summary.json").read_text())["sample_weight_scale"] == summary["sample_weight_scale"]
    predictions = pd.read_parquet(output / "oof_predictions.parquet")
    assert len(predictions) == 12
    assert predictions.subject_id.eq("S0").all()
    regions = pd.read_csv(output / "pi_regions.csv")
    assert regions.fold.str.startswith("outer_S0").all()
    assert (output / "pi_out_of_range_detail.parquet").is_file()
    assert (output / "mmd_exploratory.csv").is_file()
    assert shared_calls == ["outer_S0"]
    assert summary["final_selection_fits"] == 0
    assert not (output / "dev_selection_record.json").exists()
    assert not (output / "final_selection_cv.csv").exists()


def test_final_dev_selection_all_subjects_and_artifacts(tmp_path, monkeypatch):
    from topovital_analysis import level2

    cfg = deepcopy(CONFIG)
    cfg["level2"]["persistence_image"]["candidates"] = {"resolution": [2], "sigma_pixels": [1]}
    cfg["level2"]["classifier"]["c_grid"] = [1]
    windows = []
    for s in range(10):
        for condition, offset in (("C0", 0.), ("C1", 2.), ("C2", .2), ("C3", 2.2)):
            birth = offset + s * .01
            diagram = {rep: np.array([[birth if rep != "vr_h0" else 0.,
                                       birth + 1. if rep != "vr_h0" else 1.]])
                       for rep in cfg["inputs"]["representations"]}
            windows.append({"event_id": f"e{s}", "subject_id": f"S{s}", "condition": condition,
                            "pre": diagram, "post": diagram,
                            "amplitude": {"pre": (birth, birth + 2.), "post": (birth, birth + 2.)}})
    assert list(inspect.signature(level2.final_dev_selection).parameters) == ["windows", "cfg", "output", "fits"]
    calls = []
    shared = level2._cross_validate_candidates
    def track_shared(*args, **kwargs):
        calls.append((args[2], args[9]))
        return shared(*args, **kwargs)
    monkeypatch.setattr(level2, "_cross_validate_candidates", track_shared)
    fits = []
    result = level2.final_dev_selection(windows, cfg, tmp_path, fits)
    assert calls == [([f"S{s}" for s in range(10)], "final_dev")]
    assert {record["label"].split(":")[0] for record in fits if "__held_" in record["label"]} == {
        f"final_dev__held_S{s}" for s in range(10)}
    assert result["final_selection_fits"] == 33
    cv = pd.read_csv(tmp_path / "final_selection_cv.csv")
    assert len(cv) == 3
    assert cv.selected.all()
    record = json.loads((tmp_path / "dev_selection_record.json").read_text())
    for model in cfg["level2"]["features"]["models"]:
        item = record[model]
        assert {"resolution", "sigma_pixels", "C", "cv_score", "operating_threshold",
                "target_sensitivity", "pi_regions", "feature_count", "n_training_windows",
                "selection_basis"} <= item.keys()
        assert item["n_training_windows"] == 20
        assert item["selection_basis"] == "separate LOSO over all Development subjects; outer-fold results not used"
        with np.load(tmp_path / "final_models" / f"{model}.npz", allow_pickle=False) as saved:
            assert {"scaler_mean", "scaler_scale", "logistic_coef", "logistic_intercept",
                    "region_names", "regions", "candidate"} <= set(saved.files)
            assert saved["scaler_mean"].shape == (item["feature_count"],)
