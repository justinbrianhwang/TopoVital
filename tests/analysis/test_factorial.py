from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_analysis.factorial import (factorial_weights, threshold_c1,
                                          threshold_weighted_pvc, validate_config, weight_checks)
from topovital_analysis.factorial_metrics import (comparison_with_level2, metric_values,
                                                   stratified_auroc, _paired, _rate,
                                                   _verified_level2, _weighted_auc)
from topovital_analysis.stats import signed_auroc


ROOT = Path(__file__).parents[2]
CFG_F = load_yaml(ROOT / "configs/stage1_factorial.yaml")
CFG_A = load_yaml(ROOT / "configs/stage1_analysis.yaml")
CELLS = [(n, s) for n in ("bw", "ma", "em") for s in (24, 18, 12, 6, 0, -6)]


def _threshold_weighted_pvc_reference(frame, target):
    cells = sorted({(r.noise_type, r.snr_target) for r in frame.itertuples() if r.condition == "C3"})
    if len(cells) != 18 or frame.empty or not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        return np.nan
    groups = [g for _, g in frame.groupby("subject_id", sort=True)]
    positive = frame.loc[frame.condition.isin(("C1", "C3")), "score"]
    for t in np.sort(positive.unique())[::-1]:
        values = []
        for group in groups:
            c1 = group.loc[group.condition.eq("C1"), "score"]
            c3 = [group.loc[group.condition.eq("C3") & group.noise_type.eq(n) & group.snr_target.eq(s), "score"] for n, s in cells]
            if c1.empty or any(p.empty for p in c3):
                return np.nan
            values.append(.5 * np.mean(c1 >= t) + .5 * np.mean([np.mean(p >= t) for p in c3]))
        if np.mean(values) >= target:
            return float(t)
    return np.nan


def _reference_part(frame, condition, cell=None):
    part = frame.loc[frame.condition.eq(condition)]
    if cell is not None and condition in ("C2", "C3"):
        part = part.loc[part.noise_type.eq(cell[0]) & part.snr_target.eq(cell[1])]
    return part


def _reference_weighted_auc(frame, cells):
    weights = np.zeros(len(frame), dtype=float)
    for subject in frame.subject_id.unique():
        sm = frame.subject_id.eq(subject).to_numpy()
        for condition in ("C0", "C1", "C2", "C3"):
            for cell in ([None] if condition in ("C0", "C1") else cells):
                mask = sm & frame.condition.eq(condition).to_numpy()
                if cell is not None:
                    mask &= frame.noise_type.eq(cell[0]).to_numpy() & frame.snr_target.eq(cell[1]).to_numpy()
                if not mask.any():
                    return np.nan
                weights[mask] = 1 / (len(frame.subject_id.unique()) * 4 *
                                   (1 if cell is None else len(cells)) * mask.sum())
    pos = frame.condition.isin(("C1", "C3")).to_numpy()
    score = frame.score.to_numpy(dtype=float)
    if not np.isfinite(score).all():
        return np.nan
    order = np.argsort(score, kind="mergesort")
    score, pos, weights = score[order], pos[order], weights[order]
    numerator, negative_before = 0., 0.
    for group in np.split(np.arange(len(score)), np.flatnonzero(np.diff(score)) + 1):
        positive_weight = weights[group][pos[group]].sum()
        negative_weight = weights[group][~pos[group]].sum()
        numerator += positive_weight * (negative_before + .5 * negative_weight)
        negative_before += negative_weight
    return float(numerator / (weights[pos].sum() * weights[~pos].sum()))


def _reference_rate(frame, condition, threshold, cell=None):
    part = _reference_part(frame, condition, cell)
    if part.empty or not np.isfinite(part.score.to_numpy(dtype=float)).all() or not np.isfinite(part[threshold].to_numpy(dtype=float)).all():
        return np.nan
    return float(np.mean(part.score.to_numpy(dtype=float) >= part[threshold].to_numpy(dtype=float)))


def _reference_paired(frame, changed, original, cells):
    values = []
    base = _reference_part(frame, original)[["event_id", "subject_id", "score"]].rename(columns={"score": "base"})
    for cell in cells:
        altered = _reference_part(frame, changed, cell)[["event_id", "subject_id", "score"]]
        joined = altered.merge(base, on=["event_id", "subject_id"], validate="one_to_one")
        if len(joined) != len(altered) or not len(joined):
            return np.nan
        values.append(float((joined.score - joined.base).mean()))
    return float(np.mean(values)) if values else np.nan


def _reference_metric_values(frame, cells, rule="primary"):
    if not cells:
        return {}
    c1 = signed_auroc(_reference_part(frame, "C1").score, _reference_part(frame, "C0").score)
    c3 = np.array([signed_auroc(_reference_part(frame, "C3", c).score, _reference_part(frame, "C2", c).score) for c in cells])
    c2_art = np.array([signed_auroc(_reference_part(frame, "C2", c).score, _reference_part(frame, "C0").score) for c in cells])
    c3_art = np.array([signed_auroc(_reference_part(frame, "C3", c).score, _reference_part(frame, "C1").score) for c in cells])
    threshold = "threshold_primary" if rule == "primary" else "threshold_c1"
    rates = {k: _reference_rate(frame, k, threshold) if k in ("C0", "C1") else
             float(np.mean([_reference_rate(frame, k, threshold, c) for c in cells]))
             for k in ("C0", "C1", "C2", "C3")}
    result = {"stratified_auroc": stratified_auroc(frame, cells),
              "auroc_c1_vs_c0": c1, "auroc_c3_vs_c2": float(np.mean(c3)),
              "weighted_pooled_auroc": _reference_weighted_auc(frame, cells),
              "auroc_c2_vs_c0": float(np.mean(c2_art)), "auroc_c3_vs_c1": float(np.mean(c3_art)),
              "abs_auroc_c2_vs_c0_minus_half": float(np.mean(np.abs(c2_art - .5))),
              "abs_auroc_c3_vs_c1_minus_half": float(np.mean(np.abs(c3_art - .5))),
              "paired_change_normal": _reference_paired(frame, "C2", "C0", cells),
              "paired_change_pvc": _reference_paired(frame, "C3", "C1", cells),
              "c1_sensitivity": rates["C1"], "c3_sensitivity": rates["C3"],
              "c0_fpr": rates["C0"], "c2_fpr": rates["C2"]}
    result["weighted_balanced_accuracy"] = .25 * (rates["C1"] + rates["C3"] + 2 - rates["C0"] - rates["C2"])
    result["rate_change_fpr"] = rates["C2"] - rates["C0"]
    result["rate_change_sens"] = rates["C3"] - rates["C1"]
    return result


def rows(subjects=("A", "B"), low_c3=False):
    records = []
    for subject_index, subject in enumerate(subjects):
        count = subject_index + 1
        for event in range(count):
            for condition, score in (("C0", .1), ("C1", .9)):
                records.append({"subject_id": subject, "event_id": f"{subject}_{event}", "condition": condition,
                                "noise_type": "", "snr_target": np.nan, "score": score})
            for noise, snr in CELLS:
                for condition, score in (("C2", .2), ("C3", .3 if low_c3 else .8)):
                    records.append({"subject_id": subject, "event_id": f"{subject}_{event}", "condition": condition,
                                    "noise_type": noise, "snr_target": snr, "score": score})
    return pd.DataFrame(records)


@pytest.mark.parametrize("seed", [0, 1, 7, 19])
def test_numpy_metrics_match_references(seed):
    rng = np.random.default_rng(seed)
    frame = rows(("A", "B", "C"))
    frame["score"] = (rng.choice(np.linspace(.05, .95, 11), size=len(frame)) if seed % 2 == 0
                      else rng.uniform(.05, .95, size=len(frame)))
    frame["threshold_primary"] = rng.choice((.3, .5, .7), size=len(frame))
    frame["threshold_c1"] = rng.choice((.4, .6), size=len(frame))
    for target in (.5, .9):
        np.testing.assert_allclose(threshold_weighted_pvc(frame, target),
                                   _threshold_weighted_pvc_reference(frame, target), rtol=0, atol=0, equal_nan=True)
    for selected in (CELLS, CELLS[:3], [CELLS[4]]):
        scoped = frame.loc[frame.condition.isin(("C0", "C1")) |
                           pd.Series(list(zip(frame.noise_type, frame.snr_target)), index=frame.index).isin(selected)]
        np.testing.assert_allclose(_weighted_auc(scoped, selected), _reference_weighted_auc(scoped, selected), rtol=1e-12, equal_nan=True)
        for rule in ("primary", "c1"):
            threshold = "threshold_primary" if rule == "primary" else "threshold_c1"
            for condition in ("C0", "C1"):
                np.testing.assert_allclose(_rate(scoped, condition, threshold), _reference_rate(scoped, condition, threshold), rtol=1e-12, equal_nan=True)
            for condition in ("C2", "C3"):
                for cell in selected:
                    np.testing.assert_allclose(_rate(scoped, condition, threshold, cell), _reference_rate(scoped, condition, threshold, cell), rtol=1e-12, equal_nan=True)
            actual = metric_values(scoped, selected, rule)
            expected = _reference_metric_values(scoped, selected, rule)
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-12, equal_nan=True)
        for changed, original in (("C2", "C0"), ("C3", "C1")):
            np.testing.assert_allclose(_paired(scoped, changed, original, selected),
                                       _reference_paired(scoped, changed, original, selected), rtol=1e-12, equal_nan=True)


def test_numpy_metrics_undefined_matches_reference():
    frame = rows(("A", "B"))
    frame["threshold_primary"] = .5
    frame["threshold_c1"] = .5
    frame = frame.loc[~(frame.subject_id.eq("A") & frame.condition.eq("C3") &
                        frame.noise_type.eq("bw") & frame.snr_target.eq(24))]
    np.testing.assert_allclose(threshold_weighted_pvc(frame, .9),
                               _threshold_weighted_pvc_reference(frame, .9), equal_nan=True)
    actual, expected = metric_values(frame, CELLS), _reference_metric_values(frame, CELLS)
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-12, equal_nan=True)


def test_weights_and_checks():
    frame = rows()
    cells = list(zip(frame.noise_type, frame.snr_target))
    w = factorial_weights(frame.subject_id, frame.condition, cells, frame.event_id)
    check = weight_checks(frame.subject_id, frame.condition, cells, frame.event_id, w)
    assert check["total"] == pytest.approx(1)
    assert check["min_subject_sum"] == pytest.approx(.5)
    assert check["C2_min_cell_sum"] == pytest.approx(1 / 72)
    assert check["weighted_cov_pvc_artefact"] == pytest.approx(0, abs=1e-12)
    assert check["fit_weight_mean"] == pytest.approx(1)
    missing = frame.loc[~(frame.subject_id.eq("A") & frame.condition.eq("C1"))]
    with pytest.raises(ValueError):
        factorial_weights(missing.subject_id, missing.condition, list(zip(missing.noise_type, missing.snr_target)), missing.event_id)
    missing = frame.loc[~(frame.subject_id.eq("A") & frame.condition.eq("C3") & frame.noise_type.eq("bw") & frame.snr_target.eq(24))]
    with pytest.raises(ValueError):
        factorial_weights(missing.subject_id, missing.condition, list(zip(missing.noise_type, missing.snr_target)), missing.event_id)


def test_stratified_auc_and_undefined():
    frame = rows(("A",))
    assert stratified_auroc(frame) == pytest.approx(1)
    assert np.isnan(stratified_auroc(frame.loc[~(frame.condition.eq("C2") & frame.noise_type.eq("bw") & frame.snr_target.eq(24))], CELLS))


def test_thresholds_diverge_when_c3_low():
    frame = rows(("A", "B"), low_c3=True)
    assert threshold_c1(frame, .9) == pytest.approx(.9)
    assert threshold_weighted_pvc(frame, .9) == pytest.approx(.3)
    assert np.isnan(threshold_weighted_pvc(frame.loc[~(frame.condition.eq("C3") & frame.noise_type.eq("bw") & frame.snr_target.eq(24))], .9))


def test_metric_changes():
    frame = rows(("A",))
    frame["threshold_primary"] = .5
    frame["threshold_c1"] = .5
    values = metric_values(frame, CELLS)
    assert values["paired_change_normal"] == pytest.approx(.1)
    assert values["paired_change_pvc"] == pytest.approx(-.1)
    assert values["rate_change_fpr"] == pytest.approx(0)
    assert values["rate_change_sens"] == pytest.approx(0)
    assert values["weighted_pooled_auroc"] == pytest.approx(1)


def test_config_and_full_gate(tmp_path, monkeypatch):
    from topovital_analysis import factorial
    bad = deepcopy(CFG_F)
    bad["thresholds"]["primary"]["target"] = .8
    with pytest.raises(BlockedError):
        validate_config(bad, CFG_A)
    events = pd.DataFrame({"event_id": ["e"], "subject_id": ["S105"], "split": ["development"]})
    unapproved = deepcopy(CFG_F)
    unapproved["approved_for_full_run"] = False
    with pytest.raises(BlockedError, match="full run"):
        factorial.run_factorial(events, unapproved, CFG_A, tmp_path, tmp_path / "factorial")
    assert CFG_F["approved_for_full_run"] is True
    assert validate_config(CFG_F, CFG_A) is None


def test_comparison_hash_mismatch(tmp_path):
    frame = rows(("A",))
    frame["model"] = "pi_only"
    frame["threshold_primary"] = .5
    frame["threshold_c1"] = .5
    with pytest.raises(BlockedError, match="missing"):
        comparison_with_level2(frame, tmp_path, CFG_F)


def test_inner_cv_keeps_held_subject_out_of_features_and_regions(monkeypatch):
    from topovital_analysis import factorial

    frame = rows(("A", "B", "C"))
    windows = frame.to_dict("records")
    cfg = deepcopy(CFG_A)
    cfg["level2"]["features"]["models"] = ["amplitude_only"]
    cfg["level2"]["classifier"]["c_grid"] = [1]
    seen = []
    logs = []
    current_held = [None]

    def region(windows_arg, indices, reps, quantiles, *, excluded_subject=None):
        current_held[0] = excluded_subject
        assert {windows_arg[i]["condition"] for i in indices} == {"C0", "C1"}
        assert excluded_subject not in {windows_arg[i]["subject_id"] for i in indices}
        seen.append((excluded_subject, "region"))
        return {}

    def feature(windows_arg, indices, model, regions, candidate, reps):
        held = current_held[0]
        names = {windows_arg[i]["subject_id"] for i in indices}
        if len(names) == 1 and held in names:
            pass  # evaluation rows
        else:
            assert held not in names  # training and scaler rows
        values = {"C0": .1, "C1": .9, "C2": .2, "C3": .8}
        return np.array([[values[windows_arg[i]["condition"]]] for i in indices])

    def fit(x_train, y, weights, x_eval, c, fits, *, label=None):
        assert len(x_train) >= len(x_eval)
        assert weights.mean() > 0
        fits.append({"wall_time_s": 0., "convergence_warnings": 0})
        return x_eval[:, 0], None

    monkeypatch.setattr(factorial, "estimate_regions", region)
    monkeypatch.setattr(factorial, "_feature_matrix", feature)
    monkeypatch.setattr(factorial, "_fit_score", fit)
    monkeypatch.setattr(factorial, "_range_log", lambda *args: [])
    scores, predictions, checks = factorial._cross_validate_factorial(
        windows, cfg, ["A", "B", "C"], [], {}, [], [], [], "outer_X",
        lambda message, **kwargs: logs.append((message, kwargs)))
    assert all(v == pytest.approx(1) for v in scores["amplitude_only"][(None, None, 1.)])
    assert len(checks) == 3
    assert [s for s, kind in seen] == ["A", "B", "C"]
    assert [message.split(":")[0] for message, _ in logs] == [f"outer_X__inner_{s}" for s in ("A", "B", "C")]
    assert all("elapsed_s=" in message and "fits=1" in message and kwargs == {"flush": True}
               for message, kwargs in logs)
    for held, prediction in zip(("A", "B", "C"), predictions["amplitude_only"][(None, None, 1.)]):
        assert prediction.subject_id.eq(held).all()


def test_comparison_join_and_paired_difference(tmp_path):
    frame = rows(("A", "B"))
    frame["model"] = "amplitude_only"
    frame["threshold_primary"] = .5
    frame["threshold_c1"] = .5
    old = frame.drop(columns=["threshold_primary", "threshold_c1"]).copy()
    old["score"] = old.score - .05
    old.loc[old.condition.eq("C3"), "score"] -= .2
    old["threshold"] = .5
    source = tmp_path / CFG_F["reporting"]["comparison_with_level2"]["source"]
    source.parent.mkdir(parents=True)
    old.to_parquet(source)
    manifest = source.parent.parent / "hashes" / "level2_outputs.sha256"
    manifest.parent.mkdir()
    relative = source.relative_to(manifest.parent.parent).as_posix()
    manifest.write_text(f"{hashlib.sha256(source.read_bytes()).hexdigest()} *{relative}\n")
    cfg = deepcopy(CFG_F)
    cfg["reporting"]["bootstrap"]["resamples"] = 3
    compared = comparison_with_level2(frame, tmp_path, cfg)
    rng = np.random.default_rng(int(cfg["reporting"]["bootstrap"]["seed"]))
    expected_draws = []
    for _ in range(3):
        indices = rng.integers(0, 2, 2)
        counts = np.bincount(indices, minlength=2)
        blocks = [i for i, count in enumerate(counts) for _ in range(count)]
        new_draw = pd.concat([frame.loc[frame.subject_id.eq(("A", "B")[i])].assign(subject_id=f"draw_{j}")
                              for j, i in enumerate(blocks)], ignore_index=True)
        old_draw = pd.concat([old.loc[old.subject_id.eq(("A", "B")[i])].assign(subject_id=f"draw_{j}")
                              for j, i in enumerate(blocks)], ignore_index=True)
        old_draw["threshold_c1"] = old_draw.threshold
        nv = _reference_metric_values(new_draw, CELLS, "c1")
        ov = _reference_metric_values(old_draw, CELLS, "c1")
        expected_draws.append({metric: nv[metric] - ov[metric] for metric in nv})
    observed = compared.loc[compared.noise_type.eq("overall") & compared.rule.eq("c1") &
                            compared.aggregation.eq("window_level")]
    for result in observed.itertuples():
        values = np.asarray([draw[result.metric] for draw in expected_draws])
        good = values[np.isfinite(values)]
        if len(good) / len(values) >= .95:
            np.testing.assert_allclose([result.ci_low, result.ci_high], np.quantile(good, [.025, .975]), rtol=1e-12)
        else:
            assert np.isnan(result.ci_low) and np.isnan(result.ci_high)
    change = compared.loc[compared.metric.eq("paired_change_normal") & compared.rule.eq("c1") &
                          compared.noise_type.eq("overall") & compared.aggregation.eq("subject_equal_weight")].iloc[0]
    assert change.difference == pytest.approx(0)
    assert change.ci_low == pytest.approx(0)
    pvc_change = compared.loc[compared.metric.eq("paired_change_pvc") & compared.rule.eq("c1") &
                              compared.noise_type.eq("overall") & compared.aggregation.eq("subject_equal_weight")].iloc[0]
    assert pvc_change.difference == pytest.approx(.2)
    assert pvc_change.ci_low == pytest.approx(.2)
    assert compared.loc[compared.rule.eq("primary"), "level2_value"].isna().all()
    duplicated = pd.concat((old, old.iloc[:1]))
    duplicated.to_parquet(source)
    manifest.write_text(f"{hashlib.sha256(source.read_bytes()).hexdigest()} *{relative}\n")
    with pytest.raises(BlockedError, match="one-to-one"):
        comparison_with_level2(frame, tmp_path, cfg)
    manifest.write_text(f"{'0' * 64} *{relative}\n")
    with pytest.raises(BlockedError, match="SHA256"):
        comparison_with_level2(frame, tmp_path, cfg)


def test_level2_manifest_matches_exact_relative_path(tmp_path):
    source = tmp_path / "outputs/stage1_analysis/development/level2/oof_predictions.parquet"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"score": [.2]}).to_parquet(source)
    manifest = source.parent.parent / "hashes/level2_outputs.sha256"
    manifest.parent.mkdir()
    relative = source.relative_to(manifest.parent.parent).as_posix()
    good = hashlib.sha256(source.read_bytes()).hexdigest()
    pilot = "level2_pilot/oof_predictions.parquet"
    manifest.write_text(f"{'0' * 64} *{pilot}\n{good} *{relative}\n")
    assert _verified_level2(tmp_path, source).score.iloc[0] == .2
    manifest.write_text(f"{good} *{pilot}\n{'0' * 64} *{relative}\n")
    with pytest.raises(BlockedError, match="SHA256"):
        _verified_level2(tmp_path, source)
    manifest.write_text(f"{good} *{pilot}\n")
    with pytest.raises(BlockedError, match="SHA256"):
        _verified_level2(tmp_path, source)
    manifest.write_text(f"{good} *{relative}\n{good} *{relative}\n")
    with pytest.raises(BlockedError, match="SHA256"):
        _verified_level2(tmp_path, source)


def test_pilot_and_full_share_cv_path_without_level2_writes(tmp_path, monkeypatch):
    from topovital_analysis import factorial

    cfg_f = deepcopy(CFG_F)
    cfg_a = deepcopy(CFG_A)
    cfg_a["scope"]["events"] = 10
    cfg_a["level2"]["features"]["models"] = ["amplitude_only"]
    cfg_a["level2"]["classifier"]["c_grid"] = [1]
    cfg_f["pilot_outer_subject"] = "S0"
    for name in ("stage1_factorial.yaml", "stage1_analysis.yaml"):
        target = tmp_path / "configs" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((ROOT / "configs" / name).read_bytes())
    events = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)],
                           "subject_id": [f"S{i}" for i in range(10)], "split": "development"})
    frame = rows(tuple(f"S{i}" for i in range(10)))
    windows = frame.to_dict("records")
    monkeypatch.setattr(factorial, "_load_windows", lambda *args: windows)
    monkeypatch.setattr(factorial, "metric_tables", lambda *args: tuple(pd.DataFrame() for _ in range(4)))
    monkeypatch.setattr(factorial, "comparison_with_level2", lambda *args: pd.DataFrame())
    monkeypatch.setattr(factorial, "_range_log", lambda *args: [])
    calls = []

    def cv(windows_arg, cfg, subjects, reps, quantiles, fits, region_rows, range_rows, fold_prefix, log):
        calls.append(fold_prefix)
        fits.append({"wall_time_s": .01, "convergence_warnings": 0})
        candidate = (None, None, 1.)
        pieces = [frame.loc[frame.subject_id.eq(s)].copy() for s in subjects]
        return ({"amplitude_only": {candidate: [1.] * len(subjects)}},
                {"amplitude_only": {candidate: pieces}}, [])

    def refit(windows_arg, train, test, selected, cfg, reps, quantiles, fits, checks, label, held=None, save=None):
        fits.append({"wall_time_s": .01, "convergence_warnings": 0})
        return [], [], {rep: (0., 1., 0., 1.) for rep in reps}

    monkeypatch.setattr(factorial, "_cross_validate_factorial", cv)
    monkeypatch.setattr(factorial, "_refit", refit)
    pilot_out = tmp_path / "factorial_pilot"
    pilot = factorial.run_factorial(events, cfg_f, cfg_a, tmp_path, pilot_out, pilot=True, log=lambda *a, **k: None)
    assert pilot["outer_folds"] == 1 and pilot["final_selection_fits"] == 0
    assert calls == ["outer_S0"]
    assert not (tmp_path / "level2").exists()
    cfg_f["approved_for_full_run"] = True
    calls.clear()
    full = factorial.run_factorial(events, cfg_f, cfg_a, tmp_path, tmp_path / "factorial", log=lambda *a, **k: None)
    assert full["outer_folds"] == 10 and full["final_selection_fits"] == 2
    assert calls == [*[f"outer_S{i}" for i in range(10)], "final_dev"]


def test_synthetic_pilot_end_to_end(tmp_path, monkeypatch):
    from topovital_analysis import factorial

    cfg_f = deepcopy(CFG_F)
    cfg_a = deepcopy(CFG_A)
    cfg_a["scope"]["events"] = 10
    cfg_a["level2"]["features"]["models"] = ["amplitude_only"]
    cfg_a["level2"]["classifier"]["c_grid"] = [1]
    cfg_f["reporting"]["bootstrap"]["resamples"] = 2
    cfg_f["pilot_outer_subject"] = "S0"
    for name in ("stage1_factorial.yaml", "stage1_analysis.yaml"):
        path = tmp_path / "configs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / "configs" / name).read_bytes())
    events = pd.DataFrame({"event_id": [f"e{i}" for i in range(10)],
                           "subject_id": [f"S{i}" for i in range(10)], "split": "development"})
    windows = []
    for i in range(10):
        for condition, noise, snr in [("C0", "", None), ("C1", "", None)] + [
            (k, n, s) for n, s in CELLS for k in ("C2", "C3")]:
            pvc = condition in ("C1", "C3")
            amplitude = 2. if pvc else 0.
            diagram = {rep: np.array([[0., 1.]]) for rep in cfg_a["inputs"]["representations"]}
            windows.append({"event_id": f"e{i}", "subject_id": f"S{i}", "condition": condition,
                            "noise_type": noise, "snr_target": snr, "pre": diagram, "post": diagram,
                            "amplitude": {"pre": (amplitude + i * .01, amplitude + i * .01 + 1),
                                          "post": (amplitude + i * .01, amplitude + i * .01 + 1)}})
    monkeypatch.setattr(factorial, "_load_windows", lambda *args: windows)
    monkeypatch.setattr(factorial, "comparison_with_level2", lambda *args: pd.DataFrame())
    out = tmp_path / "factorial_pilot"
    summary = factorial.run_factorial(events, cfg_f, cfg_a, tmp_path, out, pilot=True, log=lambda *a, **k: None)
    assert summary["fits"] == 10
    assert summary["outer_folds"] == 1
    assert len(pd.read_parquet(out / "oof_predictions.parquet")) == 38
    assert len(pd.read_csv(out / "weights_check.csv")) == 10
    assert (out / "metrics_window.csv").is_file()
    assert not (tmp_path / "level2").exists()
