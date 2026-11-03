import pandas as pd
import pytest
import re

import tools.make_paper_artifacts as paper


def metric_frame(value):
    return pd.DataFrame([{"model": model, "noise_type": "overall", "snr_target": "all",
                          "rule": rule, "metric": metric, "value": value,
                          "ci_low": value - .1, "ci_high": value + .1}
                         for model in paper.MODELS for rule in ("primary", "c1")
                         for metric in paper.METRICS])


def test_performance_keeps_development_and_test_columns_separate():
    def source(value):
        frame = metric_frame(value)
        return {"metrics_subject": frame, "metrics_window": frame,
                "bootstrap_ci": frame}
    table = paper.performance_table({"dev": source(.7), "test": source(.9)})
    assert len(table) == 5
    assert (table[f"{paper.D} | stratified_auroc | subject_equal_weight"] == .7).all()
    assert (table[f"{paper.T} | stratified_auroc | subject_equal_weight"] == .9).all()
    assert all(column.startswith((paper.D, paper.T)) for column in table.columns[2:])
    md = paper.markdown(table, "Test")
    assert "0.700" in md and "0.900" in md
    assert "0.700 [0.600, 0.800]" in md
    flat = paper.flatten_columns(table)
    assert all("|" not in column for column in flat.columns)
    assert "0.700 [0.600, 0.800]" in paper.markdown(flat, "Test")


def test_cohort_counts_use_oof_windows_and_check_both_splits():
    oof = pd.DataFrame([{"event_id": event, "subject_id": "S001", "model": model,
                         "condition": str(window), "noise_type": "bw", "snr_target": 0}
                        for event in ("event-1", "event-2") for window in range(38)
                        for model in ("classical_only", "ph_only")])
    data = {
        "dev": {"oof_predictions": oof,
                "per_subject_metrics": pd.DataFrame([{
                    "subject_id": "S001", "model": "classical_only", "noise_type": "overall",
                    "snr_target": "all", "rule": "primary", "metric": "stratified_auroc"}])},
        "test": {"test_event_manifest": pd.DataFrame([{"event_id": "test-1", "subject_id": "S002"}]),
                 "evaluable_set_summary": pd.DataFrame([{
                     "row_type": "count", "total_windows": 38, "common_evaluable_windows": 37}])},
    }
    table = paper.cohort_table(data)
    assert table.loc[0, ["subjects", "events", "windows"]].tolist() == [1, 2, 76]
    assert table.loc[1, ["events", "windows", "common_evaluable_windows", "excluded_windows"]].tolist() == [1, 38, 37, 1]
    data["dev"]["oof_predictions"] = oof.iloc[:-2]
    with pytest.raises(AssertionError, match="Development windows"):
        paper.cohort_table(data)
    data["dev"]["oof_predictions"] = oof
    data["test"]["evaluable_set_summary"].loc[0, "total_windows"] = 37
    with pytest.raises(AssertionError, match="Test windows before exclusions"):
        paper.cohort_table(data)


def test_cohort_source_files_are_hashed(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    pd.DataFrame({"event_id": ["dev-1"]}).to_parquet(outputs / "oof.parquet")
    pd.DataFrame({"event_id": ["test-1"]}).to_csv(outputs / "manifest.csv", index=False)
    monkeypatch.setattr(paper, "INPUTS", {
        "dev": {"oof_predictions": "oof.parquet"},
        "test": {"test_event_manifest": "manifest.csv"},
    })
    data, digests = paper.load_inputs(tmp_path)
    assert data["dev"]["oof_predictions"].event_id.tolist() == ["dev-1"]
    assert data["test"]["test_event_manifest"].event_id.tolist() == ["test-1"]
    assert set(digests) == {"outputs/oof.parquet", "outputs/manifest.csv"}
    assert all(len(digest) == 64 for digest in digests.values())


def test_influence_is_plain_subject_mean_and_checks_official_value():
    rows = []
    for index in range(11):
        subject = "S230" if index == 10 else f"S{index:03d}"
        for model in paper.MODELS:
            value = .5 if "plus_ph" not in model else .6 + index * .01
            rows.append({"subject_id": subject, "model": model, "noise_type": "overall",
                         "snr_target": "all", "rule": "primary", "metric": "stratified_auroc",
                         "value": value})
    per_subject = pd.DataFrame(rows)
    official = pd.DataFrame([{"model_a": a, "model_b": b, "noise_type": "overall",
                              "snr_target": "all", "rule": "primary",
                              "aggregation": "subject_equal_weight", "metric": "stratified_auroc",
                              "difference": .15} for a, b in (paper.PAIRS[0], paper.PAIRS[-1])])
    result = paper.influence_table(per_subject, official)
    assert len(result) == 4
    assert result.iloc[0].subject_equal_weight_difference == pytest.approx(.15)
    assert result.iloc[1].subject_equal_weight_difference == pytest.approx(.145)
    assert result.analysis_status.nunique() == 1
    official.loc[0, "difference"] = .2
    with pytest.raises(AssertionError, match="reproduce"):
        paper.influence_table(per_subject, official)


def test_build_writes_only_inside_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "load_inputs", lambda root: (
        {"dev": {}, "test": {"per_subject_metrics": None, "comparison": None}}, {}))
    for function in ("cohort_table", "performance_table", "comparison_table", "noise_table",
                     "subject_table", "failures_table", "ablation_table", "influence_table"):
        monkeypatch.setattr(paper, function, lambda *a, **k: pd.DataFrame({
            "Development (nested LOSO OOF) | subject_equal_weight | difference": [1.23456],
            "pair": ["A | B"]}))
    for function in ("fig_model_auroc", "fig_paired", "fig_by_snr", "fig_subject"):
        monkeypatch.setattr(paper, function, lambda *a, **k: None)
    result = paper.build(tmp_path)
    assert result["outputs"]
    assert all(path.relative_to(tmp_path).parts[0] == "paper"
               for path in tmp_path.rglob("*") if path.is_file())
    for path in (tmp_path / "paper/tables").glob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()[2:]
        assert len(lines) == 3
        assert len({len(re.findall(r"(?<!\\)\|", line)) for line in lines}) == 1
        csv_columns = pd.read_csv(path.with_suffix(".csv")).columns
        assert all("|" not in column for column in csv_columns)
