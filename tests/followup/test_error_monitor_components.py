import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital_followup.error_monitor.audit import assert_unchanged, inner_provenance_check, outer_check
from topovital_followup.error_monitor.core import forbid_evaluated_subject, forbid_final_model
from topovital_followup.error_monitor.evaluation import accepted_indices, detection, risk_rows
from topovital_followup.error_monitor.features import check_features, SCORE, monitor_matrix
from topovital_followup.error_monitor.monitors import fit_monitor, operating_threshold


def test_guards():
    with pytest.raises(BlockedError):
        forbid_evaluated_subject({"S1"}, "S1")
    with pytest.raises(BlockedError):
        forbid_final_model("outputs/level3/final_models/model.joblib")
    for name in ("condition", "snr_target", "subject_id", "e", "counterfactual_pair_differences"):
        with pytest.raises(BlockedError):
            check_features([name], ["condition"])
    with pytest.raises(BlockedError):
        assert_unchanged({"a": "1"}, {"a": "2"})


def test_ranking_is_label_free_and_ties_deterministic():
    frame = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "condition": ["C0"] * 4,
                          "noise_type": [""] * 4, "snr_target": [np.nan] * 4,
                          "e": [0, 1, 0, 1]})
    first = accepted_indices(frame, [1] * 4, .5, 20261031)
    frame.e = 1 - frame.e
    assert np.array_equal(first, accepted_indices(frame, [1] * 4, .5, 20261031))
    assert len(first) == 2
    assert detection(frame.assign(e=0), [1] * 4)["reason"] == "one_class"


def test_operating_threshold():
    scores = {"a": np.array([0., 1.]), "b": np.array([0., 1.])}
    assert operating_threshold(scores, .8) == 1.


def test_score_features_match_config():
    assert SCORE == ("d", "abs_d", "p", "y_hat")


def test_shuffled_window_alignment():
    windows = [dict(event_id=e, subject_id=s, condition="C0", noise_type="clean", snr_target=np.nan)
               for e, s in [("b", "S2"), ("a", "S1"), ("c", "S1")]]
    frame = pd.DataFrame([{**windows[i], "window_index": i, "d": i, "abs_d": i,
                           "p": .5, "y_hat": 0} for i in [2, 1]])
    classical = pd.DataFrame([{**w, "sqi__test": i} for i, w in enumerate(windows)])
    matrix, names, _, _ = monitor_matrix("M1", frame, classical, windows, [0, 2],
                                         {"forbidden_inputs": []})
    assert names == [*SCORE, "sqi__test"]
    assert matrix[:, -1].tolist() == [2, 1]
    with pytest.raises(BlockedError):
        monitor_matrix("M1", frame.assign(window_index=[0, 1]), classical, windows, [0, 1],
                       {"forbidden_inputs": []})


def test_inner_wrong_subset_is_detected():
    wrong = pd.DataFrame({"subject_id": ["S1"], "training_subjects": ["S1,S2"]})
    evaluation = pd.DataFrame({"subject_id": ["S2"], "training_subjects": ["S1"]})
    with pytest.raises(BlockedError):
        inner_provenance_check("S2", {"S1"}, wrong, evaluation)
    correct = wrong.assign(training_subjects="S0")
    assert inner_provenance_check("S2", {"S0", "S1"}, correct, evaluation)["passed"]


def test_outer_subject_any_provenance_is_rejected():
    class Fit:
        subjects = frozenset({"S9"})
    empty = pd.DataFrame({"subject_id": [], "training_subjects": []})
    with pytest.raises(BlockedError):
        outer_check("S9", [Fit()], [], [empty], [], [], [])
    with pytest.raises(BlockedError):
        outer_check("S9", [], [], [empty], [], [{"S9"}], [])


def test_single_class_monitor_fallback_and_na(monkeypatch):
    import topovital_followup.error_monitor.monitors as monitors
    monkeypatch.setattr(monitors, "_arrays", lambda windows, indices: (None,) * 4)
    monkeypatch.setattr(monitors, "factorial_weights", lambda *args: np.array([.25, .75]))
    frame = pd.DataFrame({"e": [1, 1], "window_index": [1, 0]})
    fitted = fit_monitor("M1", 1, np.ones((2, 1)), frame, [], [], {"S1"}, {})
    assert fitted.fallback.startswith("constant_monitor_fallback")
    assert np.all(fitted.score(np.ones((2, 1))) == 1)
    assert np.isnan(detection(frame, [1, 1])["auroc"])
