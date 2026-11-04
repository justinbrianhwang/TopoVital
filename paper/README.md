# Paper artifacts

Tables and figures are generated from existing results.

## Tables

- `table1_cohort`: Cohort counts.
- `table2_model_performance`: Model performance; primary rule.
- `table2b_model_performance_c1_rule`: Model performance; c1 rule.
- `table3_paired_comparisons`: Paired model comparisons; primary row marked.
- `table4_test_by_noise_snr`: Independent Test by noise and SNR.
- `table5_test_per_subject`: Independent Test per subject. S230 has one event..
- `table6_test_failures`: Independent Test failures and excluded windows.
- `table7_g06_ablation`: Development only, exploratory; no feature selection.
- `table8_s230_influence`: Independent Test S230 influence; descriptive, post hoc.

## Figures

- `fig1_model_auroc.png`, `fig1_model_auroc.pdf`, `fig1_model_auroc.svg`: Development and Test metrics and bootstrap CIs.
- `fig2_paired_differences.png`, `fig2_paired_differences.pdf`, `fig2_paired_differences.svg`: Development and Test paired comparisons.
- `fig3_test_auroc_by_snr.png`, `fig3_test_auroc_by_snr.pdf`, `fig3_test_auroc_by_snr.svg`: Test subject-equal metrics.
- `fig4_test_per_subject_difference.png`, `fig4_test_per_subject_difference.pdf`, `fig4_test_per_subject_difference.svg`: Test per-subject metrics and event manifest.
- `fig5_dev_auroc_by_snr.png`, `fig5_dev_auroc_by_snr.pdf`, `fig5_dev_auroc_by_snr.svg`: Development subject-equal metrics.

Figures use display labels for aggregation, noise type, metric, and model pairs.

## Sources

- `outputs/final_test/bootstrap_ci.csv`
- `outputs/final_test/evaluable_set_summary.csv`
- `outputs/final_test/failure_log.csv`
- `outputs/final_test/metrics_subject.csv`
- `outputs/final_test/metrics_window.csv`
- `outputs/final_test/paired_model_comparison.csv`
- `outputs/final_test/per_subject_metrics.csv`
- `outputs/final_test/public/test_run/run_summary.json`
- `outputs/final_test/test_event_manifest.csv`
- `outputs/stage1_analysis/development/g06_ablation/paired_comparison.csv`
- `outputs/stage1_analysis/development/level3/bootstrap_ci.csv`
- `outputs/stage1_analysis/development/level3/metrics_subject.csv`
- `outputs/stage1_analysis/development/level3/metrics_window.csv`
- `outputs/stage1_analysis/development/level3/model_comparison.csv`
- `outputs/stage1_analysis/development/level3/oof_predictions.parquet`
- `outputs/stage1_analysis/development/level3/per_subject_metrics.csv`
- `outputs/stage1_analysis/development/level3_audit/model_comparison_no_boundary_pair.csv`
