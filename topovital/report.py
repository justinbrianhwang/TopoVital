"""File-backed Stage 0 report; no counts are inferred from configuration."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topovital.gate import pending_settings
from topovital.util import atomic_write_text, load_yaml, utc_now


def _table(df: pd.DataFrame) -> str:
    clean = df.astype(object).where(df.notna(), "")
    headers = "| " + " | ".join(map(str, clean.columns)) + " |"
    rule = "|" + "---|" * len(clean.columns)
    rows = ["| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
            for row in clean.itertuples(index=False, name=None)]
    return "\n".join([headers, rule, *rows])


def _csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_report(cfg: dict, out: Path) -> None:
    cohort = _json(out / "cohort_summary.json")
    locked_path = out / "locked_config.yaml"
    locked = load_yaml(locked_path) if locked_path.exists() else None
    lines = ["# TopoVital Stage 0 report", "", f"Generated UTC: {utc_now()}", "",
             "All counts below are read from output files.", ""]
    lines += ["## Status", ""]
    if locked:
        lines += [f"- Freeze: {locked.get('stage0_freeze_status')}",
                  f"- Frozen g_star: {locked.get('g_star')}"]
    lines += [f"- Pending Stage 0 settings: {', '.join(pending_settings(cfg)) or 'none'}", ""]
    lines += ["## Cohort", ""]
    if cohort:
        for k in ("n_source_records", "n_source_subjects", "n_cohort_records",
                  "n_cohort_subjects", "n_step1_pvc_candidates", "event_list_sha256"):
            lines.append(f"- {k}: {cohort.get(k)}")
    else:
        lines.append("Cohort summary has not been generated.")
    lines.append("")
    env = _json(out / "environment.json")
    lines += ["## Environment", ""]
    if env:
        lines += [f"- OS: {env.get('os')}", f"- Python: {env.get('python')}",
                  f"- CPU: {env.get('cpu')}", f"- Memory bytes: {env.get('memory_bytes')}",
                  f"- Package versions: {env.get('packages')}",
                  f"- Execution resources: {env.get('execution_resource_settings')}", ""]
    else:
        lines += ["Environment inventory has not been generated.", ""]
    checksums = _csv(out / "input_checksums.csv")
    lines += ["## Input provenance", ""]
    if checksums is not None:
        lines += [_table(checksums), ""]
    else:
        lines += ["Input checksums have not been generated.", ""]
    split = _csv(out / "split_assignment.csv")
    lines += ["## Subject split", ""]
    if split is not None:
        lines += [_table(split.groupby(["stratum", "split"]).size().rename("n_subjects").reset_index()), ""]
    else:
        lines += ["Split has not been generated.", ""]
    lines += ["## Attrition", ""]
    attr = _csv(out / "attrition_summary.csv")
    lines += [_table(attr) if attr is not None else "Attrition file has not been generated.", ""]
    for split_name, suffix in (("Development", "dev"), ("Test", "test")):
        folder = out / split_name.lower()
        detail = _csv(folder / f"attrition_{suffix}.csv")
        counts = _csv(folder / f"per_subject_counts_{suffix}.csv")
        lines += [f"## {split_name} Stage 0", ""]
        if detail is not None:
            lines += ["Cumulative survivors, including common and Regime 2 branches; S_eff counts subjects with at least one surviving event.",
                      "", _table(detail), ""]
        else:
            lines += ["Stage 0 aggregation has not been generated.", ""]
        if counts is not None:
            lines += ["Per-subject event counts (including zero-event subjects):", "", _table(counts), ""]
    summary = _csv(out / "development" / "g_selection_summary.csv")
    rule = _json(out / "development" / "g_rule_result.json")
    lines += ["## Development g selection and full-window bSQI sensitivity", ""]
    if summary is not None:
        lines += [_table(summary), ""]
    else:
        lines += ["g selection summary has not been generated.", ""]
    if rule is not None:
        lines += [f"- Rule g_star: {rule.get('rule_g_star')}",
                  f"- Coverage threshold: {rule.get('coverage_threshold')}", ""]
    lines += ["## Verification", ""]
    tests = out / "test_results.txt"
    lines += ["```", tests.read_text(encoding="utf-8", errors="replace").strip() if tests.exists() else "No saved test result.", "```", ""]
    resource = _csv(out / "resource_profile.csv")
    lines += ["## Resource profile", ""]
    lines += [_table(resource) if resource is not None else "Resource profile has not been generated.", ""]
    atomic_write_text(out / "stage0_report.md", "\n".join(lines))
