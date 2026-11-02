"""Synthetic checks for the G-06 Development ablation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_analysis import g06_ablation as g06
from topovital_analysis import __main__ as cli


ROOT = Path(__file__).resolve().parents[2]
CFG_G = load_yaml(ROOT / "configs/stage1_g06_ablation.yaml")
CFG_L3 = load_yaml(ROOT / "configs/stage1_level3.yaml")
CFG_F = load_yaml(ROOT / "configs/stage1_factorial.yaml")
CFG_A = load_yaml(ROOT / "configs/stage1_analysis.yaml")


def columns():
    result = []
    for name, spec in CFG_G["feature_groups"].items():
        result.extend(spec["indicators"])
        result.extend(f"{spec['prefixes'][0]}synthetic_{i}" for i in range(
            CFG_G["expected_column_counts"][name] - len(spec["indicators"])))
    return result


def test_manifest_exact_partition_and_failures():
    cols = columns()
    manifest = g06.feature_group_manifest(cols, CFG_G)
    assert len(manifest) == 97
    assert manifest.loc[manifest.column.eq("boundary_missing"), "kind"].item() == "indicator"
    assert manifest.groupby("group").size().to_dict() == {
        name: CFG_G["expected_column_counts"][name] for name in CFG_G["feature_groups"]}
    with pytest.raises(BlockedError, match="exactly one group"):
        g06.feature_group_manifest([*cols[:-1], "unknown__x"], CFG_G)
    with pytest.raises(BlockedError, match="Duplicate"):
        g06.feature_group_manifest([*cols, cols[0]], CFG_G)
    duplicate = deepcopy(CFG_G)
    duplicate["feature_groups"]["B_qrs_template"]["prefixes"].append("rhythm__")
    with pytest.raises(BlockedError, match="exactly one group"):
        g06.feature_group_manifest(cols, duplicate)
    wrong = deepcopy(CFG_G)
    wrong["expected_column_counts"]["A_rhythm"] += 1
    with pytest.raises(BlockedError, match="counts differ"):
        g06.feature_group_manifest(cols, wrong)
    with pytest.raises(BlockedError, match="Listed indicator missing"):
        g06.feature_group_manifest([c for c in cols if c != "boundary_missing"], CFG_G)


def test_sixteen_model_column_lists_and_not_applicable():
    manifest = g06.feature_group_manifest(columns(), CFG_G)
    models, skipped = g06.model_column_lists(manifest, CFG_G)
    assert len(models) == 16 and skipped == []
    assert len(models["classical_only"]) == 97
    assert len(models["classical_only_no_boundary"]) == 93
    for ref in CFG_G["references"]:
        for group in CFG_G["ablations"]["remove"]:
            name = f"{ref}__minus_{group}"
            expected = [c for c in models[ref] if c not in set(manifest.loc[manifest.group.eq(group), "column"])]
            assert models[name] == expected
    changed = deepcopy(CFG_G)
    changed["ablations"]["remove"].append("boundary")
    _, skipped = g06.model_column_lists(manifest, changed)
    assert skipped == [{"model": "classical_only_no_boundary__minus_boundary", "reason": "removal removes no column"}]


def test_reference_mismatch_is_reported():
    prediction = pd.DataFrame([{"model": "classical_only", "event_id": "e", "subject_id": "s",
                                "condition": "C0", "noise_type": None, "snr_target": np.nan,
                                "fold": "outer_s", "score": .5, "threshold_primary": .4,
                                "threshold_c1": .3, "C": 1.}])
    selection = pd.DataFrame([{"model": "classical_only", "fold": "outer_s", "C": 1.,
                               "subject_mean_auroc": .6}])
    cfg = deepcopy(CFG_G)
    cfg["references"] = {"classical_only": CFG_G["references"]["classical_only"]}
    old = prediction.copy()
    old["score"] = .7
    check = g06.reference_reproduction(prediction, selection, old, selection, cfg)
    assert not check.all_equal.all()
    assert not check.loc[check.item.eq("oof_score"), "all_equal"].item()


def test_refuses_existing_summary(tmp_path):
    out = tmp_path / CFG_G["output"]
    out.mkdir(parents=True)
    (out / "run_summary.json").write_text("{}")
    with pytest.raises(BlockedError, match="refusing to overwrite"):
        g06.run_g06_ablation(pd.DataFrame(), CFG_G, CFG_L3, CFG_F, CFG_A, tmp_path, out)


def _prepared_inputs(tmp_path, monkeypatch):
    for name in ("stage1_g06_ablation.yaml", "stage1_level3.yaml", "stage1_factorial.yaml", "stage1_analysis.yaml"):
        target = tmp_path / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "configs" / name).read_bytes())
    source = tmp_path / "outputs/stage1_analysis/development/level3_features"
    source.mkdir(parents=True)
    key_rows = [{"event_id": f"e{i}", "subject_id": f"S{i:03d}", "condition": "C0",
                 "noise_type": None, "snr_target": np.nan} for i in range(10)]
    classical = pd.DataFrame([{**row, **{c: 0. for c in columns()}} for row in key_rows])
    feature_path = source / "classical_features.parquet"
    classical.to_parquet(feature_path)
    (source / "extraction_summary.json").write_text(json.dumps({
        "stop_rule_passed": True, "classical_features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest()}))
    l3 = tmp_path / "outputs/stage1_analysis/development/level3"
    l3.mkdir(parents=True)
    pd.concat([pd.DataFrame(key_rows).assign(model=ref, fold=lambda f: "outer_" + f.subject_id,
                                               score=.5, threshold_primary=.5, threshold_c1=.5, C=1.)
               for ref in CFG_G["references"]]).to_parquet(l3 / "oof_predictions.parquet")
    pd.DataFrame({"model": ["classical_only"], "fold": ["outer_S000"], "C": [1.],
                  "subject_mean_auroc": [.5]}).to_csv(l3 / "inner_cv_selection.csv", index=False)
    monkeypatch.setattr(g06, "validate_events", lambda *a: None)
    monkeypatch.setattr(g06, "_load_windows", lambda *a: key_rows)
    return key_rows


def test_preflight_exists_before_any_fit(tmp_path, monkeypatch):
    _prepared_inputs(tmp_path, monkeypatch)
    out = tmp_path / "debug"
    def stop_before_fit(*args, **kwargs):
        assert (out / "hashes/preflight.json").is_file()
        assert (out / "ablation_config.yaml").read_bytes() == (tmp_path / "configs/stage1_g06_ablation.yaml").read_bytes()
        raise RuntimeError("fit reached")
    monkeypatch.setattr(g06, "_inner_cv", stop_before_fit)
    with pytest.raises(RuntimeError, match="fit reached"):
        g06.run_g06_ablation(pd.DataFrame(), CFG_G, CFG_L3, CFG_F, CFG_A, tmp_path, out, max_outer_folds=1)


def test_reference_mismatch_blocks_without_paired_comparison(tmp_path, monkeypatch):
    key_rows = _prepared_inputs(tmp_path, monkeypatch)
    subjects = [row["subject_id"] for row in key_rows]
    grid = g06.candidate_grid(CFG_A, "amplitude_only")
    def inner(*args, **kwargs):
        models = args[3]
        scores = {m: {c: [0.5] for c in grid} for m in models}
        frames = {m: {c: [pd.DataFrame([{**row, "score": .5} for row in key_rows[1:]])] for c in grid}
                  for m in models}
        return scores, frames
    monkeypatch.setattr(g06, "_inner_cv", inner)
    selected = {m: ((None, None, 1.), .5, .5, .5) for m in
                [*CFG_G["references"], *(f"{r}__minus_{group}" for r in CFG_G["ablations"]["from"]
                                           for group in CFG_G["ablations"]["remove"])]}
    monkeypatch.setattr(g06, "_selection", lambda *a: (selected, [
        {"model": m, "resolution": None, "sigma_pixels": None, "C": c[2],
         "subject_mean_auroc": .5, "not_evaluable_subjects": 0, "selected": c[2] == 1.}
        for m in selected for c in grid]))
    monkeypatch.setattr(g06, "_outer_refit", lambda *a: [
        {**key_rows[0], "fold": f"outer_{subjects[0]}", "model": m, "score": .7,
         "threshold_primary": .5, "threshold_c1": .5, "C": 1.}
        for m in selected])
    monkeypatch.setattr(g06, "metric_tables", lambda *a: tuple(pd.DataFrame({"model": ["x"]}) for _ in range(4)))
    out = tmp_path / "debug"
    with pytest.raises(BlockedError, match="reference reproduction failed"):
        g06.run_g06_ablation(pd.DataFrame(), CFG_G, CFG_L3, CFG_F, CFG_A, tmp_path, out, max_outer_folds=1)
    assert json.loads((out / "run_summary.json").read_text())["reference_reproduction_passed"] is False
    assert (out / "reference_reproduction_check.csv").is_file()
    assert not (out / "paired_comparison.csv").exists()


def test_cli_approval_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "verify_blas_threads", lambda: {})
    monkeypatch.setattr(cli, "load_yaml", lambda path: {"approved_for_full_run": False})
    assert cli.main(["g06-ablation"]) == 3
    assert "approved_for_full_run" in capsys.readouterr().err
