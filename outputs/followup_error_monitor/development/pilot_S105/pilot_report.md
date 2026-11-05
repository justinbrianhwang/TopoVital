# Follow-up v0.1 pilot report: error monitoring, single outer fold (S105 held out)

**Scope:**
- Development-only exploratory follow-up study.
- One outer fold: S105 held out, 9 training subjects.
- No bootstrap, no full run, no Test data, no external data.

**This is not evidence of a PH benefit.** One held-out subject cannot support a claim, and the 10 Development
subjects have been analysed many times before. Settings are in `configs/followup_error_monitor_v0_1.yaml`
(SHA256 `124b8023…`); decisions are in `design_decisions.md`.

## 1. Preservation of existing material
- **Protected files.** The SHA256 of 9,464 protected files are identical before and after the run
  (`hashes/prerun.json`, `hashes/postrun.json`). These include configs, Stage 0/1 outputs, Level 1–3, G-06, final
  Test outputs, raw data and the existing packages. An independent snapshot of 9,465 files taken before
  implementation started also matches.
- **New writes.** Files were written only to `outputs/followup_error_monitor/`. New code is in
  `topovital_followup/`, and the new config is `configs/followup_error_monitor_v0_1.yaml`.
- **Not read.** No Test file and no NSTDB record was read, and no PH was recomputed. PIs were built from the
  existing 5,040 Development diagrams; their combined digest is in `input_manifest.json`.

## 2. Data used
- **Counts.** 10 subjects, 126 events and 4,788 windows (C0 126, C1 126, C2 2,268, C3 2,268). All of these match
  the expected counts.
- **Held-out subject S105.** 29 events, 1,102 windows.
- **Training subjects (9).** S116, S123, S210, S213, S215, S219, S221, S233 and S234, with 97 events and 3,686
  windows. Their event counts range from 1 to 27.

## 3. Leakage control
All 22 audit checks passed (`leakage_audit.json`).
- **Regression check.** BaseFit(Dev ∖ {S105}) reproduces the Level 3 `outer_S105` `classical_only` fold: selected
  C 0.001, threshold 0.40376, and all 1,102 S105 scores within 1e-12. This verifies that the implementation is
  identical; the Level 3 predictions are not used as labels.
- **Outer subject.** S105 is absent from all 646 fit units and 46 BaseFits, from every error table, from the inner
  evaluation frames, and from every PI-region, imputation and scaler training set (731 subject sets checked).
- **Error-table rows.** The row's subject is absent from its generating BaseFit for all 33,174 rows.
- **Inner folds.** For each of the 9 inner held-out subjects j:
  - the monitor was trained on a separately built ErrorTable(U_j), never on a subset of ErrorTable(T);
  - j is absent from every generating fit;
  - j was labelled by BaseFit(U_j).
- **Other checks.**
  - **Shared base predictions.** All five monitors were scored on one S105 base-prediction table (same hash).
  - **Forbidden features.** None present. The score features are exactly d, |d|, p and ŷ.
  - **Model and weights.** The full-Development final model was never opened, and the F-01 weight checks passed
    for every fit.
  - **Runtime.** BLAS used 1 thread.
  - **Determinism.** BaseFit(T) and the outer M4 monitor refitted to within 1e-12.

## 4. Base model errors and whether the monitors could be trained

**Base errors:**

| Set | Windows | Errors | FP | FN | Errors by C0 / C1 / C2 / C3 |
|---|---|---|---|---|---|
| ErrorTable(T), 9 subjects | 3,686 | 983 (26.7 %) | 873 | 110 | 37 / 2 / 836 / 108 |
| S105 (outer, BaseFit(T)) | 1,102 | 170 (15.4 %) | 127 | 43 | 1 / 2 / 126 / 41 |

- **Where errors come from.** All 9 training subjects contribute errors. S210 (335) and S219 (340) account for 69 %
  of the training errors.
- **Mostly C2.** Most errors are false positives on noise-injected normal windows (C2).
- **Not independent.** The 36 noise variants per event are not independent error samples.
- **Training feasibility.** Every monitor training set contained both classes, so no single-class fallback was
  needed. There were 0 convergence warnings.

## 5. Retrospective selective risk (window-count coverage; S105 only)
Realized coverage 881 / 1,102 = 0.7995 at the 80 % target and 991 / 1,102 = 0.8993 at 90 %. Risk at 100 % equals the
base error rate 0.154.

| Monitor | Selected C | Inner mean R80 (9 subjects) | R80 | R90 | AURC [0.5, 1] | Error AUROC | Error AUPRC |
|---|---|---|---|---|---|---|---|
| M0 margin | — | 0.216 | 0.119 | 0.134 | 0.053 | 0.717 | 0.288 |
| M1 score + SQI + amplitude | 0.1 | 0.175 | 0.078 | 0.106 | 0.040 | 0.826 | 0.469 |
| M2 score + all classical | 1 | 0.195 | 0.085 | 0.118 | 0.042 | 0.792 | 0.362 |
| M3 PH only | 10 | 0.234 | 0.129 | 0.136 | 0.067 | 0.569 | 0.226 |
| M4 M2 + PH | 0.1 | 0.191 | 0.070 | 0.108 | 0.038 | 0.817 | 0.452 |

- **Primary comparison.** Delta_R80 (M4 − M2) = **−0.015** (0.0704 − 0.0851). Delta_R90 = −0.010. There is no CI;
  this is one subject.
- **Simpler monitor.** M1 (score + SQI + amplitude) had the lowest inner-CV risk and an S105 R80 (0.078) close to
  M4.
- **Inner CV.** M4 and M2 were nearly equal (0.191 vs 0.195).
- **PH alone.** M3 was the weakest learned monitor, and was worse than the margin alone.

## 6. Standardized (F-01 weighted) risk on the same accepted sets (80 %)

| Monitor | F-01 weighted risk | F-01 weighted coverage |
|---|---|---|
| M0 | 0.080 | 0.829 |
| M1 | 0.049 | 0.845 |
| M2 | 0.062 | 0.837 |
| M3 | 0.087 | 0.813 |
| M4 | 0.045 | 0.837 |

The direction is the same as the count-based result (M4 − M2 = −0.017 at equal weighted coverage 0.837).

## 7. Class composition of the accepted sets and the operating threshold

**Retrospective 80 % set:**

| Monitor | C0 | C1 | C2 | C3 | Accepted FN / all PVC | FP abstained | FN abstained |
|---|---|---|---|---|---|---|---|
| M2 | 0.79 | 0.97 | 0.70 | 0.89 | 0.031 | 0.54 | 0.60 |
| M4 | 0.93 | 0.83 | 0.77 | 0.82 | 0.011 | 0.56 | 0.86 |

The columns C0–C3 give the coverage of each class.

- **Partly a shift in what is held back.** M4 holds back more PVC windows than M2 (C1 coverage 0.83 vs 0.97: 5 vs 1
  of 29 clean PVCs; C3 0.82 vs 0.89) and accepts more clean normal windows.
- **Missed PVCs.** Fewer missed PVCs are accepted automatically (1.1 % vs 3.1 % of all PVC windows).
- **Interpretation.** The lower risk is partly a change in which classes are held back, so it should not be called
  a general safety gain.

**Operating threshold** (fixed on the training subjects' inner OOF scores, target 80 %, applied unchanged to S105):

| Monitor | S105 coverage | Risk | C1 coverage | C3 coverage |
|---|---|---|---|---|
| M0 | 0.823 | 0.120 | 0.97 | 0.91 |
| M1 | 0.587 | 0.051 | 0.48 | 0.42 |
| M2 | 0.950 | 0.142 | 1.00 | 0.98 |
| M3 | 0.963 | 0.141 | 0.97 | 0.96 |
| M4 | 0.863 | 0.086 | 0.86 | 0.89 |

- **Poor transfer.** The thresholds learned on the training subjects transfer poorly to S105: the realized coverage
  ranges from 0.59 to 0.96.
- **M1.** M1's low risk comes with withholding over half of the PVC windows.
- **M4 vs M2.** The two monitors are not compared at equal coverage here.

## 8. Noise detection versus error detection (within noise type × SNR × class cells)

**C2 cells** (18 cells, 126 errors; mean within-cell error AUROC):

| M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 0.62 | 0.91 | 0.64 | 0.40 | 0.82 |

- **What this shows.** M4 separates errors from correct predictions inside the same noise/SNR cell much better than
  M2, so it is not only a low-SNR detector.
- **Caution.** M1 (SQI and amplitude, no PH) does this even better.
- **Cell-level comparison.** Across the 35 cells with both classes, M4 > M2 in 27 cells.
- **C3 cells** (17 evaluable; 1 cell has no error): M4 0.93, M2 0.89.
- **Clean cells.** C0 and C1 contain 1 and 2 errors only, so the clean-signal error detection cannot be judged.

## 9. Added value of PH over the strong classical monitor (M4 vs M2)

**What the pilot shows:**
- M4 had lower selective risk than M2 on S105, under both count and F-01 weighted aggregation.
- M4 had better within-cell error discrimination.
- M4 accepted fewer missed PVCs.

**Reasons this does not establish an added value of PH:**
1. It is one subject. There is no CI, and the effect is about 13 errors out of 881 accepted windows.
2. In the inner CV over 9 subjects, M4 and M2 were almost equal.
3. M1, which uses SQI and amplitude without PH and with far fewer features, did about as well. M2 may underuse
   signal-quality information among its 101 inputs, so the M4 gain may reflect quality information rather than
   anything specific to topology.
4. Part of the risk reduction comes with more PVC abstention.
5. Training errors are concentrated in two subjects.

The pilot neither confirms nor rules out a benefit.

## 10. Limits of a one-subject pilot
- S105 was fixed in advance. It is the subject with the most events (29), so its errors (mainly C2 false positives)
  may not be typical.
- No subject-level uncertainty can be estimated, and the influence of the subject cannot be assessed.
- The 36 noise variants per event are correlated, so the number of windows overstates the information.
- The Development data have been inspected many times. Even a full 10-fold result would be an adaptive Development
  analysis, not a validation.

## 11. Computation

**Measured, 1 process, 1 BLAS thread:**

| Stage | Wall time (s) |
|---|---|
| Inputs and hashing | 33.0 |
| Base and regression check | 14.8 |
| Error tables | 160.8 |
| Monitors | 170.5 |
| Evaluation and audit | 1.0 |
| **Total** | **380** |

- **Peak memory:** 1,154 MB RSS.
- **Base fit units: 646.** The plan counted 650; the 4 unused C values of the 9-subject set were never needed,
  because T is fitted only at its selected C.
  - The units come from 46 BaseFits.
  - Median unit fit time: 0.019 s.
  - The cache was hit 1,229 times.
- **Monitor fits: 113.** That is 4 monitors × (9 × 3 + 1) = 112, plus 1 determinism refit.
  - Median monitor fit time: 0.64 s; total 102 s.
- **Warnings and fallbacks:** none.

**Estimate for a full 10-fold run** (`plan.json`):
- **Workload:** 175 BaseFits, up to 1,925 units and 1,120 monitor fits.
- **Basis:**
  - The pilot's monitor and error-table stages (331 s for 10 subject sets) scale with the number of subject sets.
    A full run needs about 100 sets (10 T plus 90 U), roughly 10× the pilot, so about 1 hour single-process.
  - Base fitting is minor.
  - Memory should be similar to the pilot (about 1.2 GB), since subject sets are processed one at a time.
- **Not done:** the full run was not executed.

## 12. Open decisions before any full run
- **Approval.** Whether to run the full 10-fold Development analysis (`approved_for_full_run`), and with which paired
  bootstrap reporting. The config reserves stream [20261031, 1] and 1,000 resamples.
- **Threshold transfer.** Whether to keep the operating-threshold rule, given its poor transfer in this pilot. Any
  change would be a new version (v0.2), not an edit of v0.1.
- **Role of M1.** Whether M1 should be reported as a co-primary baseline. Changing the primary comparison after
  this pilot would be a result-driven change, so it must be a new, documented version.
- **External data.** Which unused external subjects or dataset would serve as the eventual independent evaluation.
- **Repository.** At the time of the pilot nothing was committed. The researcher approved committing the code, tests and this pilot summary on 2026-11-04.
