import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital_analysis.factorial import _arrays, factorial_weights
from topovital_analysis.level2 import fit_weights
from topovital_analysis.persistence_images import estimate_regions
from topovital_followup.error_monitor.audit import assert_unchanged, file_hashes, inner_provenance_check
from topovital_followup.error_monitor.core import forbid_final_model, subject_rows
from topovital_followup.error_monitor.evaluation import accepted_indices, detection, risk_rows
from topovital_followup.error_monitor.features import check_features, monitor_matrix, pi_matrix
from topovital_followup.error_monitor.monitors import fit_monitor
from topovital_followup.error_monitor.pilot import compute_fold

from conftest import synthetic


def _key(frame):
    return frame[["event_id", "subject_id", "condition", "noise_type", "snr_target"]].fillna("NA").astype(str).agg("|".join, axis=1)


def test_01_outer_provenance(fold):
    windows, _, result = fold
    store = result["store"]
    assert all("S04" not in u.subjects for u in store.units.values())
    assert all("S04" not in b.subjects for b in store.basefits.values())
    assert all("S04" not in t.subject_id.values for t in result["tables"].values())
    assert all("S04" not in f.subjects for f in result["monitor_fits"])
    assert any(a["name"] == "outer_provenance" and a["passed"] for a in result["audit"])
    for fitted in result["monitor_fits"]:
        if fitted.fallback or fitted.model == "M0":
            continue
        expected = len(subject_rows(windows, fitted.subjects))
        imputer, scaler = fitted.estimator.steps[:2]
        assert len(imputer[1].statistics_) == scaler[1].n_features_in_
        assert np.all(scaler[1].n_samples_seen_ == expected)


def test_02_inner_table_recompute(fold):
    _, _, result = fold
    store = result["store"]
    T = frozenset(store.windows[i]["subject_id"] for i in range(len(store.windows))) - {"S04"}
    U, held = T - {"S00"}, "S00"
    wrong = result["tables"][T].loc[lambda f: f.subject_id.isin(U)]
    with pytest.raises(BlockedError):
        inner_provenance_check(held, U, wrong, result["eval_frames"][held])
    assert inner_provenance_check(held, U, result["tables"][U], result["eval_frames"][held])["passed"]


def test_03_subject_block_integrity(fold):
    windows, _, result = fold
    for frame in [*result["tables"].values(), *result["eval_frames"].values()]:
        for subject, group in frame.groupby("subject_id"):
            expected = {i for i, w in enumerate(windows) if w["subject_id"] == subject}
            assert set(group.window_index) == expected
            assert group.role.nunique() == 1 if "role" in group else True


def test_04_forbidden_base_prediction(fold):
    windows, classical, result = fold
    with pytest.raises(BlockedError):
        forbid_final_model("outputs/final_models/model.pkl")
    base = next(iter(result["store"].basefits.values()))
    with pytest.raises(BlockedError):
        base.predict(windows, classical, subject_rows(windows, {next(iter(base.subjects))}))


def test_05_error_labels_recompute(fold):
    windows, classical, result = fold
    fits = {b.id: b for b in result["store"].basefits.values()}
    for table in result["tables"].values():
        for row in table.iloc[::max(1, len(table) // 8)].itertuples():
            predicted = fits[row.generating_basefit_id].predict(windows, classical, [row.window_index]).iloc[0]
            assert predicted.y_hat == row.y_hat
            assert row.e == int(row.y_hat != row.y)


def test_06_outer_frame_identical(fold):
    _, _, result = fold
    frame = result["outer_frame"]
    predictions = result["predictions"]
    assert len(predictions) == len(frame)
    assert _key(predictions).equals(_key(frame))
    assert predictions.e.equals(frame.e)
    assert all(f"{m}_score" in predictions for m in ("M0", "M1", "M2", "M3", "M4"))
    assert next(a for a in result["audit"] if a["name"] == "identical_base_predictions")["passed"]


def test_07_forbidden_features(fold, small_cfg):
    windows, classical, result = fold
    for name in ("condition", "snr_target", "noise_type", "subject_id", "event_id", "y", "e", "z"):
        with pytest.raises(BlockedError):
            check_features([name], small_cfg["forbidden_inputs"])
    for model in ("M1", "M2", "M3", "M4"):
        _, names, _, _ = monitor_matrix(model, result["outer_frame"], classical, windows, [],
                                        small_cfg["persistence_image"] | {"forbidden_inputs": small_cfg["forbidden_inputs"]},
                                        frozenset({"S00", "S01", "S02", "S03"}))
        assert not set(names) & {"condition", "snr_target", "noise_type", "subject_id", "event_id", "y", "e", "z"}


def test_08_factorial_weights(fold):
    windows, _, result = fold
    for unit in result["store"].units.values():
        rows = subject_rows(windows, unit.subjects)
        weights = factorial_weights(*_arrays(windows, rows))
        assert weights.sum() == pytest.approx(1)
        assert fit_weights(weights).mean() == pytest.approx(1)
        assert unit.weight_checks["min_subject_sum"] == pytest.approx(1 / len(unit.subjects))
        assert unit.weight_checks["C2_min_cell_sum"] == pytest.approx(1 / 72)


def test_09_monitor_weights(fold, monkeypatch):
    windows, _, result = fold
    frame = next(t for t in result["tables"].values() if t.e.nunique() == 2)
    seen = {}
    from sklearn.pipeline import Pipeline
    original = Pipeline.fit
    def capture(self, X, y=None, **kwargs):
        seen["weights"] = kwargs["logisticregression__sample_weight"].copy()
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", capture)
    fitted = fit_monitor("M1", .1, frame[["d", "abs_d", "p", "y_hat"]].to_numpy(), frame,
                         windows, [], set(frame.subject_id), {})
    assert fitted.estimator[-1].class_weight is None
    assert np.allclose(seen["weights"], fit_weights(factorial_weights(*_arrays(windows, frame.window_index))))


def test_10_pi_regions_and_medians(fold, small_cfg):
    windows, classical, result = fold
    training = frozenset({"S00", "S01", "S02", "S03"})
    source = [i for i, w in enumerate(windows) if w["subject_id"] in training and w["condition"] in ("C0", "C1")]
    _, _, actual, _ = pi_matrix(windows, training, subject_rows(windows, training), small_cfg["persistence_image"])
    assert actual == estimate_regions(windows, source, small_cfg["persistence_image"]["representations"],
                                      small_cfg["persistence_image"]["range_quantiles"])
    for unit in result["store"].units.values():
        for name, median in unit.medians.items():
            values = classical.iloc[subject_rows(windows, unit.subjects)][name]
            assert median == pytest.approx(values.median())


def test_11_label_permutation_acceptance(fold):
    frame = fold[2]["outer_frame"].copy()
    scores = fold[2]["predictions"]["M2_score"].to_numpy()
    first = accepted_indices(frame, scores, .8, 20261031)
    frame["e"] = frame.e.sample(frac=1, random_state=5).to_numpy()
    assert np.array_equal(first, accepted_indices(frame, scores, .8, 20261031))


def test_12_full_coverage_risk(fold):
    windows, _, result = fold
    frame = result["outer_frame"]
    rows = risk_rows(frame, result["predictions"]["M2_score"], windows, [], [1.0], 20261031)
    assert rows[0]["risk"] == pytest.approx(frame.e.mean())


def test_13_fallback_and_hash_ties(fold):
    windows, _, result = fold
    frame = result["outer_frame"].copy()
    frame.e = 0
    fitted = fit_monitor("M1", .1, np.ones((len(frame), 1)), frame, windows, [], {"S04"}, {})
    assert fitted.fallback and np.all(fitted.score(np.ones((len(frame), 1))) == 0)
    assert np.isnan(detection(frame, np.zeros(len(frame)))["auroc"])
    first = accepted_indices(frame, np.ones(len(frame)), .8, 20261031)
    frame.e = 1
    assert np.array_equal(first, accepted_indices(frame, np.ones(len(frame)), .8, 20261031))


def test_14_repeated_and_shuffled_fold(fold, small_cfg):
    _, _, original = fold
    for order in ("sorted", "shuffled"):
        windows, classical = synthetic(order=order)
        result = compute_fold(windows, classical, "S04", small_cfg, threshold_target=.9,
                              hashes=("feature", "config", "code"), log=lambda _: None)
        a, b = original["predictions"], result["predictions"]
        left = a.assign(key=_key(a)).set_index("key").sort_index()
        right = b.assign(key=_key(b)).set_index("key").sort_index()
        for model in ("M0", "M1", "M2", "M3", "M4"):
            np.testing.assert_allclose(left[f"{model}_score"], right[f"{model}_score"], atol=1e-10)
        assert left.e.equals(right.e)


def test_15_protected_hashes(tmp_path):
    path = tmp_path / "protected.txt"
    path.write_text("first")
    before = file_hashes(tmp_path, ["*.txt"])
    path.write_text("second")
    with pytest.raises(BlockedError):
        assert_unchanged(before, file_hashes(tmp_path, ["*.txt"]))


def test_16_blas_subprocess():
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    process = subprocess.run([sys.executable, "-m", "topovital_followup.error_monitor", "blas-check"],
                             capture_output=True, text=True, check=True)
    report = json.loads(process.stdout)
    assert all(report["env"][name] == "1" for name in names)
    assert report["threadpools"] and all(count == 1 for _, count in report["threadpools"])
