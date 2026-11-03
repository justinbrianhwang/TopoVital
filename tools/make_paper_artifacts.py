"""Build publication tables and figures from frozen Development and Test results."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from topovital.util import sha256_file
MODELS = {
    "classical_only": ("Classical", "#2a78d6"),
    "classical_plus_ph": ("Classical+PH", "#eb6834"),
    "classical_only_no_boundary": ("Classical (no boundary)", "#1baf7a"),
    "classical_plus_ph_no_boundary": ("Classical+PH (no boundary)", "#eda100"),
    "ph_only": ("PH only", "#e87ba4"),
}
PAIRS = [
    ("classical_plus_ph", "classical_only"),
    ("classical_plus_ph", "ph_only"),
    ("classical_only", "ph_only"),
    ("classical_only", "classical_only_no_boundary"),
    ("classical_plus_ph", "classical_plus_ph_no_boundary"),
    ("classical_plus_ph_no_boundary", "classical_only_no_boundary"),
]
D = "Development (nested LOSO OOF)"
T = "Independent Test"
SNR = [24, 18, 12, 6, 0, -6]
METRICS = ["stratified_auroc", "c0_fpr", "c2_fpr", "c1_sensitivity",
           "c3_sensitivity", "auroc_c2_vs_c0", "auroc_c3_vs_c1"]


def select(frame, **filters):
    result = frame
    for column, value in filters.items():
        result = result.loc[result[column].astype(str) == str(value)]
    if len(result) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(result)}")
    return result.iloc[0]


def overall(frame, **filters):
    return select(frame, noise_type="overall", snr_target="all", **filters)


def performance_table(data, rule="primary"):
    rows = []
    for model, (label, _) in MODELS.items():
        row = {"model": label, "rule": rule}
        for split, source in ((D, data["dev"]), (T, data["test"])):
            for metric in METRICS:
                subject = overall(source["metrics_subject"], model=model, rule=rule, metric=metric)
                row[f"{split} | {metric} | subject_equal_weight"] = subject.value
                if metric == "stratified_auroc":
                    ci = overall(source["bootstrap_ci"], model=model, rule=rule, metric=metric)
                    row[f"{split} | stratified_auroc | ci_low"] = ci.ci_low
                    row[f"{split} | stratified_auroc | ci_high"] = ci.ci_high
                    window = overall(source["metrics_window"], model=model, rule=rule, metric=metric)
                    row[f"{split} | stratified_auroc | window_level"] = window.value
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(data):
    rows = []
    for index, (a, b) in enumerate(PAIRS):
        row = {"pair": f"{MODELS[a][0]} − {MODELS[b][0]}",
               "primary": index == 0, "rule": "primary", "metric": "stratified_auroc"}
        for split, source in ((D, data["dev"]), (T, data["test"])):
            frame = source["comparison_audit"] if split == D and index == 5 else source["comparison"]
            for aggregation in ("subject_equal_weight", "window_level"):
                item = overall(frame, model_a=a, model_b=b, rule="primary",
                               metric="stratified_auroc", aggregation=aggregation)
                for field in ("difference", "ci_low", "ci_high"):
                    row[f"{split} | {aggregation} | {field}"] = item[field]
        rows.append(row)
    return pd.DataFrame(rows)


def influence_table(per_subject, comparisons):
    rows = []
    for a, b in (PAIRS[0], PAIRS[-1]):
        subset = per_subject.loc[(per_subject.noise_type == "overall") &
                                 (per_subject.snr_target.astype(str) == "all") &
                                 (per_subject.rule == "primary") &
                                 (per_subject.metric == "stratified_auroc") &
                                 (per_subject.model.isin((a, b)))]
        pivot = subset.pivot(index="subject_id", columns="model", values="value")
        if len(pivot) != 11 or pivot.isna().any().any() or "S230" not in pivot.index:
            raise ValueError("Expected 11 complete Test subjects including S230")
        differences = pivot[a] - pivot[b]
        official = overall(comparisons, model_a=a, model_b=b, rule="primary",
                           aggregation="subject_equal_weight", metric="stratified_auroc")
        all_mean = float(differences.mean())
        if abs(all_mean - float(official.difference)) > 1e-12:
            raise AssertionError("Per-subject mean does not reproduce official difference")
        for scope, values in (("all 11 subjects", differences),
                              ("excluding S230", differences.drop("S230"))):
            rows.append({"split": T, "pair": f"{MODELS[a][0]} − {MODELS[b][0]}",
                         "subjects": scope, "n_subjects": len(values),
                         "subject_equal_weight_difference": float(values.mean()),
                         "analysis_status": "post hoc, not prespecified; no CI computed"})
    return pd.DataFrame(rows)


def cohort_table(data):
    dev_subjects = set(data["dev"]["per_subject_metrics"].loc[
        lambda f: (f.model == "classical_only") & (f.noise_type == "overall") &
        (f.snr_target == "all") & (f.rule == "primary") &
        (f.metric == "stratified_auroc"), "subject_id"])
    dev_oof = data["dev"]["oof_predictions"].loc[
        lambda f: f.model == "classical_only"]
    dev_events = dev_oof.event_id.nunique()
    dev_windows = len(dev_oof[["event_id", "condition", "noise_type", "snr_target"]].drop_duplicates())
    test_events = data["test"]["test_event_manifest"]
    test_summary = data["test"]["evaluable_set_summary"]
    counts = test_summary.loc[test_summary.row_type == "count"]
    test_event_count = test_events.event_id.nunique()
    test_windows = int(counts.total_windows.sum())
    if dev_windows != dev_events * 38:
        raise AssertionError("Development windows must equal events * 38")
    if test_windows != test_event_count * 38:
        raise AssertionError("Test windows before exclusions must equal events * 38")
    return pd.DataFrame([
        {"split": D, "subjects": len(dev_subjects), "events": dev_events,
         "windows": dev_windows,
         "common_evaluable_windows": pd.NA, "excluded_windows": pd.NA},
        {"split": T, "subjects": test_events.subject_id.nunique(),
         "events": test_event_count, "windows": test_windows,
         "common_evaluable_windows": int(counts.common_evaluable_windows.sum()),
         "excluded_windows": int(test_windows - counts.common_evaluable_windows.sum())},
    ])


def noise_table(test):
    rows = []
    for noise, snr in [("protocol_9_cell", "12_6_0")] + [(n, str(s)) for n in ("bw", "ma", "em") for s in SNR]:
        row = {"split": T, "noise_type": noise, "snr_target_db": snr,
               "aggregation": "subject_equal_weight", "rule": "primary"}
        for model, (name, _) in MODELS.items():
            item = select(test["metrics_subject"], model=model, noise_type=noise,
                          snr_target=snr, rule="primary", metric="stratified_auroc")
            row[name] = item.value
        item = select(test["comparison"], model_a=PAIRS[0][0], model_b=PAIRS[0][1],
                      noise_type=noise, snr_target=snr, rule="primary",
                      aggregation="subject_equal_weight", metric="stratified_auroc")
        row.update(primary_difference=item.difference, primary_ci_low=item.ci_low,
                   primary_ci_high=item.ci_high)
        rows.append(row)
    return pd.DataFrame(rows)


def subject_table(test):
    p = test["per_subject_metrics"]
    p = p.loc[(p.noise_type == "overall") & (p.snr_target == "all") &
              (p.rule == "primary") & (p.metric == "stratified_auroc")]
    pivot = p.pivot(index="subject_id", columns="model", values="value")
    events = test["test_event_manifest"].groupby("subject_id").event_id.nunique()
    rows = []
    for subject in sorted(pivot.index):
        row = {"split": T, "subject_id": subject, "events": int(events[subject])}
        for model, (name, _) in MODELS.items():
            row[name] = pivot.loc[subject, model]
        row["primary_difference"] = pivot.loc[subject, PAIRS[0][0]] - pivot.loc[subject, PAIRS[0][1]]
        rows.append(row)
    return pd.DataFrame(rows)


def failures_table(test):
    failure = test["failure_log"].copy()
    excluded = test["evaluable_set_summary"].loc[lambda f: f.row_type == "excluded_window"]
    totals = excluded.groupby("subject_id").total_windows.sum()
    failure.insert(0, "split", T)
    failure["excluded_windows_by_subject"] = failure.subject_id.map(totals).fillna(0).astype(int)
    return failure


def ablation_table(dev):
    frame = dev["ablation"]
    rows = []
    for a, b in frame.loc[(frame.noise_type == "overall") & (frame.snr_target == "all") &
                          (frame.rule == "primary"), ["model_a", "model_b"]].drop_duplicates().itertuples(index=False):
        row = {"split": D, "ablation": b, "reference": MODELS[a][0],
               "aggregation": "subject_equal_weight"}
        for metric, label in (("stratified_auroc", "delta_auroc"),
                              ("c0_fpr", "delta_c0_fpr"), ("c2_fpr", "delta_c2_fpr")):
            item = overall(frame, model_a=a, model_b=b, rule="primary",
                           aggregation="subject_equal_weight", metric=metric)
            row[label] = item.difference
            if metric == "stratified_auroc":
                row["ci_low"], row["ci_high"] = item.ci_low, item.ci_high
        rows.append(row)
    if len(rows) != 14:
        raise ValueError("Expected 14 Development ablations")
    return pd.DataFrame(rows)


def markdown(frame, caption):
    frame = frame.copy()
    ci_groups = []
    for column in frame.columns:
        separator = " | " if " | " in column else ", "
        if column.endswith(separator + "stratified_auroc" + separator + "subject_equal_weight"):
            prefix = column.rsplit(separator, 1)[0]
            ci_groups.append((column, prefix + separator + "ci_low", prefix + separator + "ci_high"))
        elif column.endswith(separator + "difference"):
            prefix = column.rsplit(separator, 1)[0]
            ci_groups.append((column, prefix + separator + "ci_low", prefix + separator + "ci_high"))
    ci_groups.extend((("primary_difference", "primary_ci_low", "primary_ci_high"),
                      ("delta_auroc", "ci_low", "ci_high")))
    for value, low, high in ci_groups:
        if all(name in frame for name in (value, low, high)):
            frame[value] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
                            for v, lo, hi in zip(frame[value], frame[low], frame[high])]
            frame = frame.drop(columns=[low, high])

    def render(value):
        if pd.isna(value): return ""
        if isinstance(value, (float, np.floating)): return f"{value:.3f}"
        return str(value).replace("|", "\\|")
    headings = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = [f"# {caption}", "", "| " + " | ".join(headings) + " |",
             "| " + " | ".join(["---"] * len(headings)) + " |"]
    lines += ["| " + " | ".join(render(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines) + "\n"


def style_axis(ax, *, xgrid=True):
    ax.set_facecolor("#fcfcfb")
    ax.tick_params(colors="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color="#e6e6e2", linewidth=0.8)
        ax.set_axisbelow(True)


def save_figure(fig, base):
    fig.patch.set_facecolor("#fcfcfb")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="#fcfcfb")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="#fcfcfb",
                metadata={"CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
                          "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc)})
    plt.close(fig)


def fig_model_auroc(data, base):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True, sharey=True)
    for ax, (split, source) in zip(axes, ((D, data["dev"]), (T, data["test"]))):
        for i, (model, (name, color)) in enumerate(MODELS.items()):
            s = overall(source["metrics_subject"], model=model, rule="primary", metric="stratified_auroc")
            w = overall(source["metrics_window"], model=model, rule="primary", metric="stratified_auroc")
            ci = overall(source["bootstrap_ci"], model=model, rule="primary", metric="stratified_auroc")
            ax.errorbar(s.value, i - .13, xerr=[[s.value-ci.ci_low], [ci.ci_high-s.value]],
                        fmt="o", color=color, markersize=8, elinewidth=1.5, capsize=3,
                        label="subject_equal_weight" if i == 0 else None)
            ax.plot(w.value, i + .13, "s", color=color, markersize=8,
                    label="window_level" if i == 0 else None)
        ax.set_yticks(range(len(MODELS)), [v[0] for v in MODELS.values()])
        ax.set_title(split); ax.set_xlabel("Stratified AUROC")
        style_axis(ax)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, .07, 1, 1]); save_figure(fig, base)


def fig_paired(table, base):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, split in zip(axes, (D, T)):
        for i, row in table.iterrows():
            for aggregation, offset, marker in (("subject_equal_weight", -.14, "o"),
                                                ("window_level", .14, "s")):
                prefix = f"{split} | {aggregation} | "
                value = row[prefix + "difference"]
                ax.errorbar(value, i + offset,
                            xerr=[[value-row[prefix+"ci_low"]], [row[prefix+"ci_high"]-value]],
                            fmt=marker, color="#eb6834" if i == 0 else "#52514e",
                            markersize=8, elinewidth=1.5, capsize=3,
                            label=aggregation if i == 0 else None)
        ax.axvline(0, color="#0b0b0b", linewidth=1.2)
        ax.set_yticks(range(len(table)), table.pair)
        ax.set_title(split); ax.set_xlabel("Paired stratified AUROC difference")
        style_axis(ax); ax.legend(frameon=False)
        if ax.get_yticklabels():
            ax.get_yticklabels()[0].set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.tight_layout(); save_figure(fig, base)


def fig_by_snr(source, title, base):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, noise in zip(axes, ("bw", "ma", "em")):
        endpoints = []
        for model, (name, color) in MODELS.items():
            values = [select(source["metrics_subject"], model=model, noise_type=noise,
                             snr_target=str(s), rule="primary", metric="stratified_auroc").value for s in SNR]
            ax.plot(SNR, values, color=color, linewidth=2, marker="o", markersize=8)
            endpoints.append((values[-1], name, color))
        previous = -float("inf")
        for value, name, color in sorted(endpoints):
            label_y = max(value, previous + .018)
            ax.plot([-6, -7], [value, label_y], color=color, linewidth=.8)
            ax.text(-7.3, label_y, name, fontsize=8, color=color, va="center")
            previous = label_y
        ax.set_xlim(26, -22); ax.set_xticks(SNR); ax.set_title(f"{title} · {noise}")
        ax.set_xlabel("SNR (dB)"); style_axis(ax)
    axes[0].set_ylabel("Subject-equal stratified AUROC")
    fig.text(.5, .005, "Circle markers: subject_equal_weight", ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=[0, .035, 1, 1]); save_figure(fig, base)


def fig_subject(table, base):
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(table))
    ax.scatter(table.primary_difference, y, s=70, color="#eb6834")
    for i, row in table.iterrows():
        ax.annotate(f"n={row.events}", (row.primary_difference, i),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0, color="#0b0b0b", linewidth=1.2)
    ax.set_yticks(y, table.subject_id); ax.invert_yaxis()
    ax.set_title(T); ax.set_xlabel("Classical+PH − Classical, stratified AUROC")
    style_axis(ax); fig.tight_layout(); save_figure(fig, base)


INPUTS = {
    "dev": {"metrics_subject": "stage1_analysis/development/level3/metrics_subject.csv",
            "oof_predictions": "stage1_analysis/development/level3/oof_predictions.parquet",
            "metrics_window": "stage1_analysis/development/level3/metrics_window.csv",
            "bootstrap_ci": "stage1_analysis/development/level3/bootstrap_ci.csv",
            "comparison": "stage1_analysis/development/level3/model_comparison.csv",
            "per_subject_metrics": "stage1_analysis/development/level3/per_subject_metrics.csv",
            "comparison_audit": "stage1_analysis/development/level3_audit/model_comparison_no_boundary_pair.csv",
            "ablation": "stage1_analysis/development/g06_ablation/paired_comparison.csv"},
    "test": {"metrics_subject": "final_test/metrics_subject.csv",
             "metrics_window": "final_test/metrics_window.csv",
             "bootstrap_ci": "final_test/bootstrap_ci.csv",
             "comparison": "final_test/paired_model_comparison.csv",
             "per_subject_metrics": "final_test/per_subject_metrics.csv",
             "evaluable_set_summary": "final_test/evaluable_set_summary.csv",
             "failure_log": "final_test/failure_log.csv",
             "test_event_manifest": "final_test/test_event_manifest.csv",
             "run_summary": "final_test/public/test_run/run_summary.json"},
}


def load_inputs(root=ROOT):
    data, digests = {}, {}
    for split, mapping in INPUTS.items():
        data[split] = {}
        for name, relative in mapping.items():
            path = root / "outputs" / relative
            if path.suffix == ".json":
                data[split][name] = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".parquet":
                data[split][name] = pd.read_parquet(path)
            else:
                data[split][name] = pd.read_csv(path)
            digests[path.relative_to(root).as_posix()] = sha256_file(path)
    return data, digests


def flatten_columns(frame):
    result = frame.copy()
    result.columns = [", ".join(map(str, column)) if isinstance(column, tuple)
                      else str(column).replace(" | ", ", ") for column in result.columns]
    return result


def build(root=ROOT):
    root = Path(root)
    data, inputs = load_inputs(root)
    paper = root / "paper"
    tables = paper / "tables"; figures = paper / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    products = {
        "table1_cohort": (cohort_table(data), "Cohort counts"),
        "table2_model_performance": (performance_table(data), "Model performance; primary rule"),
        "table2b_model_performance_c1_rule": (performance_table(data, "c1"), "Model performance; c1 rule"),
        "table3_paired_comparisons": (comparison_table(data), "Paired model comparisons; primary row marked"),
        "table4_test_by_noise_snr": (noise_table(data["test"]), "Independent Test by noise and SNR"),
        "table5_test_per_subject": (subject_table(data["test"]), "Independent Test per subject. S230 has one event."),
        "table6_test_failures": (failures_table(data["test"]), "Independent Test failures and excluded windows"),
        "table7_g06_ablation": (ablation_table(data["dev"]), "Development only, exploratory; no feature selection"),
        "table8_s230_influence": (influence_table(data["test"]["per_subject_metrics"], data["test"]["comparison"]),
                                 "Independent Test S230 influence; descriptive, post hoc"),
    }
    for name, (frame, caption) in products.items():
        display_frame = flatten_columns(frame)
        display_frame.to_csv(tables / f"{name}.csv", index=False, lineterminator="\n")
        (tables / f"{name}.md").write_text(markdown(display_frame, caption), encoding="utf-8")
    fig_model_auroc(data, figures / "fig1_model_auroc")
    fig_paired(products["table3_paired_comparisons"][0], figures / "fig2_paired_differences")
    fig_by_snr(data["test"], T, figures / "fig3_test_auroc_by_snr")
    fig_subject(products["table5_test_per_subject"][0], figures / "fig4_test_per_subject_difference")
    fig_by_snr(data["dev"], D, figures / "fig5_dev_auroc_by_snr")
    readme = ["# Paper artifacts", "", "Tables and figures are generated from existing results.", "",
              "## Tables", ""]
    readme += [f"- `{name}`: {caption}." for name, (_, caption) in products.items()]
    readme += ["", "## Figures", "",
               "- `fig1_model_auroc`: Development and Test metrics and bootstrap CIs.",
               "- `fig2_paired_differences`: Development and Test paired comparisons.",
               "- `fig3_test_auroc_by_snr`: Test subject-equal metrics.",
               "- `fig4_test_per_subject_difference`: Test per-subject metrics and event manifest.",
               "- `fig5_dev_auroc_by_snr`: Development subject-equal metrics.", "",
               "## Sources", ""]
    readme += [f"- `{name}`" for name in sorted(inputs)]
    (paper / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    outputs = {p.relative_to(root).as_posix(): sha256_file(p)
               for directory in (tables, figures) for p in sorted(directory.iterdir()) if p.is_file()}
    outputs["paper/README.md"] = sha256_file(paper / "README.md")
    manifest = {"inputs": inputs, "outputs": outputs}
    (paper / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build()
    print(f"Wrote {len(result['outputs']) + 1} paper files")
