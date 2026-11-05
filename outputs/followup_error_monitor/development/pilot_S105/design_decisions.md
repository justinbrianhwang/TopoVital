# Follow-up v0.1: design decisions E-01 to E-08

Settings are in `configs/followup_error_monitor_v0_1.yaml`. Every item below was fixed before any pilot result
existed. Items marked **[IMPL]** are implementation details chosen for this study; all other items come from the
follow-up prompt v0.1 written by the researcher.

## Boundary between this study and TopoVital
- This is a new, exploratory Development study. The frozen TopoVital protocol (Stage 0, Stage 1 Levels 1–3, G-06)
  and the single Test evaluation are not changed, rerun or reinterpreted.
- The 10 Development subjects have been analysed many times. This LOSO is therefore an adaptively developed
  Development analysis, not an independent validation.
- The independent Test set of TopoVital has been published. It is never read here, and it can never serve as an
  independent test for this new method.
- New code lives in `topovital_followup/`, the new config is `configs/followup_error_monitor_v0_1.yaml`, and new
  outputs go only to `outputs/followup_error_monitor/`. Existing files are hashed before and after the run.

## E-01 Task and scope
- **Question.** When predicting the errors of a fixed classical PVC pipeline, does PH add information beyond model
  margin, classical features and SQI?
- **Data.** Development C0–C3 only: 10 subjects, 126 events, 4,788 windows.
- **Inputs.** Classical features come from the existing Level 3 feature table. PIs are built from the existing 5,040
  diagram files. No waveform is regenerated, NSTDB is not read, and no PH is recomputed.

## E-02 Fixed base pipeline
- The base model is Level 3 `classical_only`, refitted inside every allowed subject set A as BaseFit(A). It keeps
  the Level 3 algorithm and rules: training-median imputation, a standard scaler, L2 logistic regression with the
  C grid {0.001, 0.01, 0.1, 1, 10}, F-01 weights, F-02 inner LOSO selection and the F-03 primary threshold.
- The full-Development final model is never used.
- The no-boundary variant is not added. The alignment-cue limitation of Level 3 still applies.
- **[IMPL] Regression check.** BaseFit(Dev ∖ {S105}) must reproduce the Level 3 `outer_S105` `classical_only` fold
  (selected C, threshold and scores) to within 1e-12.
  - This check only verifies that the implementation is identical. The Level 3 predictions are not used as error
    labels.
  - If the check fails, the run stops before the monitor stage.

## E-03 Monitors and primary comparison
- **Score features.** d = z − logit(θ), |d|, p and ŷ, where z is the decision function and θ the training-derived
  threshold.
  - A θ outside (0, 1) blocks the run. It is never replaced by an epsilon.
- **Monitors.**
  - M0 ranks by margin only: score −|d|.
  - M1 uses the score features, SQI and amplitude.
  - M2 uses the score features and all 97 classical columns. It is the strong classical baseline.
  - M3 uses PI features only.
  - M4 is M2 plus PI features.
- **Primary comparison.** M4 − M2. An advantage over M0 or M1 alone is not evidence for PH.
- **[IMPL] M1 missing indicators.** M1 has none. None of the seven existing indicators relates to SQI or amplitude,
  and neither group has missing values in Development.
- **Forbidden inputs.** Condition codes, labels, SNR, noise type, snippet, scale, subject, record or event ids,
  annotation R peaks, clean counterparts of noisy halves, counterfactual pair differences, and anything from Test.
  These are checked by name.

## E-04 Leakage-free nesting
- **ErrorTable(A).** Rows of subject k are produced by BaseFit(A ∖ {k}).
- **Outer fold.** For held-out subject s and T = Dev ∖ {s}:
  - the evaluation base model is BaseFit(T);
  - the monitor is trained on ErrorTable(T).
- **Inner fold.** For inner held-out subject j and U = T ∖ {j}:
  - the monitor is trained on ErrorTable(U);
  - j is labelled and scored with BaseFit(U).
- **Forbidden shortcut.** Building monitor training data by subsetting ErrorTable(T) or any earlier OOF table.
- **Fit cache.** A fitted base pipeline (training subjects, C) is shared only between identical fits. The key
  includes the sorted training-subject set, C, the feature-table hash, the config hash and the code hash. No earlier
  OOF prediction or model is reused.
- **Fit count for the pilot.** A base pipeline is fitted for each subset of T of size 9, 8, 7 and 6 at all five C
  values: (1 + 9 + 36 + 84) × 5 = 650 fits.

## E-05 Monitor training
- **Model.** L2 logistic regression with C ∈ {0.1, 1, 10}; ties go to the smaller C. Imputation and standardization
  are fitted on training rows only, and F-01 weights are rescaled to mean 1.
- **No rebalancing.** No class weighting and no change to the error/correct ratio.
- **Single-class training labels.** The monitor falls back to a constant score equal to the weighted training error
  rate, and the fallback is recorded.
- **PI.** R = 10, σ = 1 pixel, the four existing representations, pre/post/post − pre. Ranges are fitted on the
  monitor-training subjects' C0/C1 windows only, and out-of-range points are not clipped. These are pilot defaults,
  not claimed optimal.
- **Selection objective.** The subject-equal mean count-based selective risk at 80 % coverage on the inner held-out
  subjects.
- **[IMPL] Seeds.** Ties are broken by sha256(`20261031|tie|` + window id); labels and conditions never decide a
  tie. The bootstrap stream [20261031, 1] is reserved for a possible full run.

## E-06 Endpoint
- **Primary endpoint.** Delta_R80 = R_macro,M4(0.80) − R_macro,M2(0.80) at window-count coverage, averaged with
  equal subject weight.
  - In each subject, the k_s = max(1, ⌊0.8 N_s⌋) lowest-risk windows are accepted.
  - The realized coverage k_s / N_s is reported, not rounded.
- **Separate standardized report.** On the same accepted set, F-01 weighted selective risk and F-01 weighted
  coverage are reported separately. Window-count coverage puts most of the weight on noisy windows (36 of the 38
  per event); the F-01 weights rebalance that.
- **Secondary.**
  - R90 and the risk–coverage curve at 50–100 %;
  - AURC on [0.5, 1.0] by the trapezoid rule over the six grid points (raw integral, not the full-range AURC);
  - error AUROC and AUPRC with the error prevalence;
  - breakdowns by noise type, SNR and class.
- **Pilot only.** No bootstrap is run, and only one subject is evaluated.

## E-07 Operating threshold and safety
- **[IMPL] Threshold rule.** The smallest training OOF monitor score t such that the subject-equal mean of
  window-count coverage (score ≤ t) is at least 0.80, computed on the training subjects' inner OOF scores at the
  selected C.
- The threshold is applied unchanged to S105, with no recalibration.
- **Safety report.**
  - Coverage of C0, C1, C2 and C3.
  - Sensitivity within accepted PVC windows.
  - Accepted false negatives as a share of all PVC windows.
  - The accepted false-positive rate.
  - The abstention rates of false positives and of false negatives.
- An abstention is never counted as correct.

## E-08 Pilot only
- **Pilot fold.** Single outer fold with S105 held out. S105 is fixed and is not changed after results.
- **[IMPL] Determinism check.** BaseFit(T) and the outer M4 monitor are fitted twice, and the scores must agree to
  within 1e-12.
- **Stop.** After the report. No full 10-fold run, final Development monitor, external data or new Test.
