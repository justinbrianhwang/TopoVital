# TopoVital

**Reproducible, CPU-only pipeline for topological analysis of premature ventricular contractions (PVCs) in ECG.**

TopoVital studies whether topological data analysis (persistent homology) can separate a PVC-induced perturbation from
noise, baseline wander, clipping and time-warping artefacts in short ECG windows. This repository implements and
documents **Stage 0** of the experimental protocol: cohort construction, event eligibility, normal-window matching,
guard-interval (`g_star`) selection on Development data only, and a locked Development/Test gate. It also contains
the **Stage 1** topological feature extraction, which is in progress on Development data.

> **Status (2026-11-02):**
> - **Stage 0** is complete and **frozen** with `g_star = 250 ms`. The Test split was processed exactly once with the frozen configuration.
> - **Stage 1 (Development only):**
>   - Clean-window persistent homology (C0/C1) is complete: 504 / 504 jobs succeeded.
>   - Regime 1 artefact injection (C2/C3, NSTDB `bw` / `ma` / `em`) is complete: 4,536 / 4,536 jobs succeeded.
> - **Level 1** (pre/post PH distances) and **Level 2** (persistence images with nested subject-level CV) analyses on Development are complete. The Level 2 C1-vs-C2 result has a known confound (see its limitation note).
> - The **factorial C0–C3** analysis on Development is complete. It reduced the score dependence on artefact, but improved PVC detection was not shown.
> - The **Level 3** classical baseline comparison on Development is complete. Persistent homology did not add PVC-detection value over classical features, and the PH-only model has a documented thread-environment reproducibility exception.
> - Test remains locked for Stage 1. Regime 2/3 and Test evaluation have not started.
> - The authoritative progress record is [`outputs/stage1_status.yaml`](outputs/stage1_status.yaml).

<p align="center">
  <img src="assets/figs/fig1_stage0_pipeline.png" width="900" alt="TopoVital Stage 0 pipeline: PhysioNet data, cohort, PVC candidates, stratified subject split, Development lane with g sweep and freeze, Test lane locked until freeze, Common cohort and Regime 2 subset">
  <br><em>Figure 1. Stage 0 pipeline. The guard interval g* is selected on Development only; Test is locked until the freeze and then processed once.</em>
</p>

---

## Highlights

- **Protocol-first.** Every scientific value lives in [`configs/stage0.yaml`](configs/stage0.yaml), tagged with the protocol clause it comes from (`[P§n]`) or the researcher approval that fixed it (`[APPROVED A-0n]`).
- **No Test leakage.** The guard interval is chosen on Development only. Test aggregation is refused unless a frozen `locked_config.yaml` exists, and the stage-0 config hash and code hash still match. A Test-side `g` sweep is always refused.
- **Branching cohorts.** Template failure never removes an event from the Common cohort (Regimes 1/3); it only excludes the event from the Regime 2 subset.
- **Auditable.** Per-event audit with every gate flag, quality metric and first drop reason. Resource profile (wall time, peak RSS) for every command. SHA256 for all inputs.
- **CPU-only.** No GPU/CUDA/PyTorch/TensorFlow dependency. Record-by-record processing. The Stage 0 pipeline pins BLAS/OpenMP threads to 1. Stage 1 analysis runs before the Level 3 full run used the machine's default BLAS thread count; see the Level 3 reproducibility exception and `outputs/resource_profile_corrections.csv`. Since analysis code version 2.0.0 (`topovital_analysis/runtime.py`), the analysis CLI sets the thread variables before numpy is imported and refuses to run unless threadpoolctl reports 1 thread.

## Stage 0 results (MIT-BIH Arrhythmia Database, MLII, `g = 250 ms`)

| Stage | Dev subjects | Dev events | Test subjects | Test events |
|---|---|---|---|---|
| Subjects in split | 22 | – | 21 | – |
| PVC (`V`) candidates | 16 | 3,293 | 16 | 3,610 |
| Window inside record | 16 | 3,274 | 16 | 3,598 |
| Pre-boundary beats all `N` | 10 | 476 | 12 | 495 |
| No ADC saturation, IQR ≥ 1e-4 mV | 10 | 476 | 12 | 495 |
| Pre-boundary bSQI ≥ 0.8 | 10 | 466 | 11 | 414 |
| Post-boundary single isolated `V`, no saturation / signal loss | 10 | 281 | 11 | 179 |
| After overlap removal (≥ 10 s separation) | 10 | 186 | 11 | 126 |
| **Common cohort** (matched normal window) | **10** | **126** | **11** | **81** |
| **Regime 2** (≥ 10 template beats) | **10** | **126** | **11** | **80** |

<p align="center">
  <img src="assets/figs/fig3_attrition_dev_test.png" width="900" alt="Stage 0 attrition at g* = 250 ms for Development and Test, events with subjects in parentheses, log scale">
  <br><em>Figure 2. Stage 0 attrition at g* = 250 ms: events (subjects) per gate, log scale.</em>
</p>

Cohort: 48 records / 47 subjects → 44 records / 43 subject groups after excluding records 102, 104, 107 and 217. Records 201 and 202 are treated as one subject.
Split: subject-grouped and stratified by step-1 PVC count (strata 0 / 1–88 / > 88), with seed `20261031`.

**Guard-interval selection (Development, valid DWT onsets):**

| g | Coverage of R − onset ≤ g | Meets 95%? |
|---|---|---|
| 200 ms | 83 / 88 = 94.32 % | no |
| **250 ms** | **84 / 87 = 96.55 %** | **yes (smallest)** |
| 300 ms | 86 / 87 = 98.85 % | yes |

<p align="center">
  <img src="assets/figs/fig5_guard_interval_selection.png" width="700" alt="Development coverage of R minus QRS onset within g: 94.32 % at 200 ms, 96.55 % at 250 ms, 98.85 % at 300 ms, with a 95 % threshold line">
  <br><em>Figure 3. Guard-interval selection on Development. 250 ms is the smallest candidate above the pre-specified 95 % threshold.</em>
</p>

**Known limitations** (recorded in [`configs/g_decision_reviewed.yaml`](configs/g_decision_reviewed.yaml)):
- 39 of 129 Development events (30 %) had no valid DWT onset; these are concentrated in S219 and S210.
- Three events exceed 250 ms (116_164994, 116_302281, 219_365144). `g` mitigates QRS boundary cutting but does not guarantee to eliminate it.
- Events are concentrated in a few subjects.
- The Test run wrote its event audit and then failed in the Development-only `g` coverage summary step. The audit was not regenerated. Test attrition tables were aggregated read-only from that audit (deviation F-02). The bug is left unfixed in the frozen code because fixing it would change the frozen `code_hash`.

Full numbers are in [`outputs/attrition_summary.csv`](outputs/attrition_summary.csv) and [`outputs/stage0_report.md`](outputs/stage0_report.md). Every decision and deviation is logged in [`outputs/implementation_decisions.md`](outputs/implementation_decisions.md).

## Stage 1 results (Development, in progress)

Stage 1 uses the 126 Development Common-cohort events at `g = 250 ms`. C1 is the PVC window and C0 its matched normal window. Each window is split at `t0` into a pre half and a post half, and each half is analysed separately.

**Settings** ([`configs/stage1.yaml`](configs/stage1.yaml), approvals B-01 to B-03):

| Setting | Value |
|---|---|
| Normalization | `(x − median(pre)) / IQR(pre)`, using pre-window statistics for both halves |
| Delay embedding | τ = 28 samples (pooled median of AMI first minima), m = 5 (FNN < 5 %). Never crosses `t0` |
| Point cloud | 300 delay vectors per half, uniform stride |
| Vietoris–Rips | GUDHI 3.13.0, full complex (no `r_max`), `max_dimension = 2`, H0 and H1. Built on the 300 subsampled delay vectors of each half, **not** on the raw ECG samples |
| Cubical | Sublevel PH of `x` and superlevel PH of `−x` |

**Clean windows, C0/C1** ([`outputs/stage1_features/development/`](outputs/stage1_features/development/)):

| Item | Result |
|---|---|
| Jobs (126 events × 2 conditions × 2 halves) | 504 |
| Succeeded / failed | 504 / 0 |
| Validation | All jobs pass. VR has 4,500,250 simplices, 1 essential H0 bar and 0 essential H1 bars. Cubical essential births equal `min(x)` and `−max(x)` |
| Wall time (4 workers) | 1,797 s |
| Max child peak RSS | 490 MB |

**Scope and cost of the VR computation.** "Full complex" means that no edge-length truncation is applied to these 300 points. Every complex therefore has 300 vertices, C(300, 2) = 44,850 edges and C(300, 3) = 4,455,100 triangles, for 4,500,250 simplices in total.

The result is exact persistent homology of a 300-point delay-embedding sample of each 5 s half. It is not a topological computation on the complete raw signal: one half holds 1,800 samples, which give 1,688 delay vectors at τ = 28 and m = 5.

Measured cost per VR job over the 504 C0/C1 jobs is a median wall time of 14.2 s (range 7.0–25.6 s, single thread, 4 jobs running at once) and a peak RSS of 487–490 MB. Cost grows roughly as O(N³) in the number of points, so a larger N would need a separate resource assessment.

The embedding diagnostics behind τ and m are in [`outputs/stage1_pilot/`](outputs/stage1_pilot/). That folder's `embedding_selection_PENDING.md` lists the open questions as they stood before approval; decisions B-01 to B-03 in [`outputs/implementation_decisions.md`](outputs/implementation_decisions.md) resolved them.

**Regime 1 artefact injection, C2/C3** ([`configs/stage1_regime1.yaml`](configs/stage1_regime1.yaml), approvals C-01 to C-03).

The run covers 126 events × 2 conditions × 3 noise types × 6 SNRs = 4,536 post-half jobs. The settings are:
- **Noise source:** NSTDB `bw`, `ma` and `em`, channel 0, Development block `[0, 325000)` only.
- **Snippets:** 5 s, mean-removed. Each event gets one snippet per noise type from 180 non-overlapping slots by a seeded permutation (seed `[20261031, noise_index]`). C2 and C3 share that snippet across all SNRs.
- **Onset taper:** 100 ms raised-cosine ramp at `t0`.
- **Scaling:** `a = ‖s_ref‖ / ‖n‖ · 10^(−SNR/20)` with `s_ref = post − median(pre)`, at SNR ∈ {24, 18, 12, 6, 0, −6} dB.
- **Verification:** achieved SNR must be within 1e-3 dB of the target.
- **Pre half:** reused from the C0/C1 results after a SHA-256 check of the input.

Results (run_id `20261031T232737Z_3634d895`):

| Item | Result |
|---|---|
| Jobs (126 events × C2/C3 × 3 noise types × 6 SNRs) | 4,536. Each noise type × condition × SNR cell has 126 jobs |
| Succeeded / failed | 4,536 / 0. VR, cubical and validation checks all pass |
| Wall time (4 workers) | 17,219 s (about 4 h 47 min). Median 14.7 s per VR job; max child peak RSS 490 MB |
| SNR check (all jobs) | 4,536 / 4,536 within tolerance. Max \|SNR error\| 1.2e-14 dB; max relative norm error 1.4e-15 |
| Independent SNR check | Recomputed from raw MIT-BIH and NSTDB on the injected waveform for 60 random jobs, without the Stage 1 code: max \|SNR error\| 7.1e-15 dB; scale factors agree to 4.6e-15 (relative) |
| Snippet allocation | Re-derived from the seed: 378 / 378 match. All slots lie in the Development block. No snippet is used twice. One snippet per event and noise type is shared by C2/C3 and all SNRs. C2 and C3 scale factors differ in all 2,268 pairs |
| Cache reuse | Post half: 108 / 4,536 (2.4 %) reused from the 3-event pre-validation run. Pre half: 4,536 / 4,536 reused from C0/C1 after a SHA-256 input check, with no mismatch |
| Test data | Not read. Neither MIT-BIH Test records nor the NSTDB Test block `[325000, 650000)` were used |

The 108 reused jobs carry a different run-config hash from the rest. A normalized comparison of the two configurations shows that the only differing key is `approved_for_full_run`. That key was false during pre-validation and true for the full run; the details are in [`hashes/regime1_cache_config_diff.json`](outputs/stage1_features/development/hashes/regime1_cache_config_diff.json).

The original Regime 1 ledgers contain local absolute paths and are kept out of the repository. The published copies in [`outputs/stage1_features/development/public/`](outputs/stage1_features/development/public/) replace those paths with repo-relative ones and leave every other value unchanged. Source and public SHA-256 hashes are listed in `PUBLIC_MANIFEST.json`.

Per-job persistence diagrams (504 + 4,536 `.npz` files) are not stored in this repository; they will be archived on Zenodo.

## Stage 1 analysis, Level 1: pre/post PH distances (Development)

Settings are in [`configs/stage1_analysis.yaml`](configs/stage1_analysis.yaml) (approvals L-01 to L-06).

**Method**
- **Pairs:** for each window, the pre-half and post-half diagrams are compared, giving 4,788 pairs: C0 126, C1 126, C2 2,268 and C3 2,268. C2/C3 reuse the C0/C1 pre half.
- **Representations and distances:** four representations are analysed separately: cubical sublevel H0, cubical superlevel H0, VR H0 and VR H1. Each pair gets three distances: bottleneck, W1 and W2. All use the L∞ ground metric and finite pairs only. W1 and W2 are exact, from an assignment problem with diagonal copies.
- **Separation measures:** C1 (PVC) is compared with C2 (artefact on a normal window) in each noise type × SNR cell.
  - **Signed AUROC** fixes the direction (larger distance = PVC), so values below 0.5 mean the direction is reversed.
  - **Direction-free separation** is reported separately: `2·|AUROC − 0.5|` and the KS statistic, without p-values.
  - **LOSO operating point:** the threshold is set on the training subjects' C1 windows for 90 % C1 sensitivity, then applied to the held-out subject.
  - **Uncertainty:** subject-cluster bootstrap with 1,000 resamples.
  - **Aggregation:** results are given at window level and subject level (equal weight per subject).

**Run:** 57,456 distances, 0 failures, 1,500 s with 4 workers.
- **Independent check:** an independent GUDHI/hera recomputation agrees within 6.6e-11 (relative).
- **Exact vs approximate:** the hera approximation differs from the exact values by at most 2.7e-6.

**Signed AUROC, C1 vs C2 (window level, bw / em / ma):**

| Representation, distance | 24 dB | 12 dB | 6 dB | 0 dB | −6 dB |
|---|---|---|---|---|---|
| VR H0, bottleneck | 0.77 / 0.77 / 0.77 | 0.74 / 0.74 / 0.74 | 0.64 / 0.59 / 0.58 | 0.45 / 0.38 / 0.31 | 0.33 / 0.24 / 0.12 |
| Cubical sublevel H0, W1 | 0.51 / 0.50 / 0.47 | 0.44 / 0.45 / 0.26 | 0.32 / 0.31 / 0.09 | 0.15 / 0.15 / 0.02 | 0.04 / 0.05 / 0.00 |

**Reading (descriptive, Development only):**
- **High SNR:** VR H0 bottleneck distance separates PVC from normal windows best, with AUROC about 0.77. Cubical distances barely do (0.51–0.56).
- **Low SNR:** the artefact changes the diagram more than the PVC does, so signed AUROC falls below 0.5. Separation becomes "large" only in the reversed direction.
- **Operating point:** at the LOSO threshold for 90 % C1 sensitivity, the C2 false-positive rate is at best 0.44 (VR H0, 24 dB) and reaches 0.87–1.00 at 0 dB and below.
- **Conclusion:** a single pre/post distance is not sufficient. This result is about this representation only and is not a verdict on PH as a whole.

Outputs are in [`outputs/stage1_analysis/development/level1/`](outputs/stage1_analysis/development/level1/): per-window distances, summaries, C1/C2 separation per cell and per subject, and the hera cross-check.

## Stage 1 analysis, Level 2: persistence images, nested LOSO (Development)

Settings are in [`configs/stage1_analysis.yaml`](configs/stage1_analysis.yaml) (approvals L-01 to L-06).

**Method**
- **Features:** persistence images in (birth, persistence) coordinates for the four representations, built for the pre half, the post half and their difference. VR H0 images are 1-D because all its births are 0. Nine amplitude features (min, max, range for pre, post and post − pre) are also used.
- **PI region:** estimated in every training fold from the training subjects' C0/C1 halves only. Points outside the region are not clipped.
- **Models:** `pi_only`, `amplitude_only` and `pi_plus_amplitude`, each an L2 logistic regression.
- **Training task:** C1 (PVC, clean) against C2 (normal window with artefact, all 18 noise × SNR cells) in one model.
  - Each subject has equal total weight; within a subject, C1 and C2 have equal total weight.
  - The weights passed to the fit are rescaled to mean 1.
- **Cross-validation:** nested leave-one-subject-out with 10 outer folds.
  - The inner LOSO selects the PI resolution (10, 20), σ (1, 2 pixels) and C (0.001–10).
  - The operating threshold gives 90 % subject-equal C1 sensitivity on the inner out-of-fold scores of the training subjects.
  - The held-out subject is never used for the PI region, standardization, selection or threshold.
- **Final Development setting:** chosen by a separate LOSO over all 10 subjects, never from outer-fold results.
- **Uncertainty:** subject-cluster bootstrap with 1,000 resamples. MMD is exploratory only.

**Run:** 10 / 10 outer folds succeeded with 4,533 fits and 0 convergence warnings. The run took 6,080 s with a peak of 870 MB. Code and configuration hashes were identical before and after the run ([`hashes/`](outputs/stage1_analysis/development/hashes/)).

**Results (outer folds, subject-equal, 95 % bootstrap CI, all 18 cells):**

| Model | C1 vs C2 AUROC | C1 sensitivity | C2 FPR | **C0 FPR** | **C3 sensitivity** |
|---|---|---|---|---|---|
| `amplitude_only` | 0.58 [0.41, 0.75] | 0.92 | 0.89 | 0.96 | 0.82 |
| `pi_only` | 0.84 [0.78, 0.90] | 0.90 | 0.45 [0.31, 0.56] | 0.85 [0.64, 0.98] | 0.56 [0.42, 0.68] |
| `pi_plus_amplitude` | **0.92** [0.87, 0.96] | 0.83 [0.60, 0.99] | 0.38 [0.22, 0.53] | **0.68** [0.44, 0.88] | **0.59** [0.42, 0.72] |

- **C2 FPR by SNR:** for `pi_plus_amplitude`, it is 0.52–0.67 at 24–18 dB, 0.17–0.29 at 0 to −6 dB for bw/em, and 0.00–0.03 for ma.
- **C3 sensitivity by SNR:** 0.80–0.85 at 24–18 dB, and 0.02–0.50 at 0 to −6 dB. For ma at 0 to −6 dB it is 0.02–0.09.
- **Final Development setting (separate LOSO):**
  - `pi_only`: R = 10, σ = 2 px, C = 0.1
  - `pi_plus_amplitude`: R = 10, σ = 2 px, C = 0.1
  - `amplitude_only`: C = 10

> **Limitation: the high C1/C2 AUROC is not PVC detection performance.**
> In the C1-vs-C2 training task, PVC presence and artefact presence are perfectly anti-aligned: every positive is a clean PVC, and every negative is a normal window with artefact. A model can therefore score well by detecting "no artefact".
> The other conditions show this:
> - Clean normal windows (C0) are flagged as PVC at a rate of 0.68–0.96.
> - PVCs with artefact (C3) are detected at only 0.59, and almost never at 0 dB or below.
> - AUROC rises as SNR falls (up to 1.00 for ma at −6 dB), the opposite of what a PVC detector should show.
>
> These are descriptive Development results. The factorial C0–C3 analysis below addresses this confound.

Outputs are in [`outputs/stage1_analysis/development/level2/`](outputs/stage1_analysis/development/level2/):
- out-of-fold predictions;
- window-level, subject-level and per-subject metrics;
- bootstrap CIs;
- inner and final selection records;
- PI regions and out-of-range logs;
- exploratory MMD.

The one-fold pilot is in `level2_pilot/`. Fitted model arrays (`final_models/`) are not stored in the repository.

## Stage 1 analysis, factorial C0–C3 training (Development)

This analysis addresses the Level 2 confound. Settings are in [`configs/stage1_factorial.yaml`](configs/stage1_factorial.yaml) (approvals F-01 to F-04). Everything else (inputs, persistence images, PI candidates, C grid, classifier, tie break, bootstrap) is inherited unchanged from Level 2.

**Method**
- **Training task:** PVC (C1, C3) against non-PVC (C0, C2), all four conditions in one model.
- **Weights (F-01):**
  - Each subject has equal total weight, and within a subject each condition has weight 1/4.
  - Within C2 and C3, the 18 noise × SNR cells have equal weight, and events are equal within each cell.
  - The weighted covariance between PVC presence and artefact presence is therefore 0. This was checked in all 111 training folds (|cov| ≤ 1.3e-17).
- **Selection metric (F-02):** stratified AUROC.
  - For each subject: 0.5 · AUROC(C1 vs C0) + 0.5 · mean over the 18 cells of AUROC(C3 vs C2).
  - The subject values are then averaged with equal weight.
  - This compares PVC with non-PVC only under the same artefact condition.
- **Thresholds (F-03):** both are set on the training subjects' inner out-of-fold scores.
  - **Primary:** 90 % subject-equal mean of [0.5 · C1 sensitivity + 0.5 · cell-mean C3 sensitivity].
  - **C1 rule:** 90 % C1 sensitivity. This is the Level 2 rule.
- **Artefact dependence (F-04):**
  - AUROC(C2 vs C0) and AUROC(C3 vs C1).
  - Paired score changes f(C2) − f(C0) and f(C3) − f(C1) on the same event and cell.
  - FPR(C2) − FPR(C0) and sensitivity(C3) − sensitivity(C1).
- **Cross-validation:** nested LOSO with 10 outer folds. PI parameters and C are selected in the inner LOSO only.

**Run:**
- 10 / 10 outer folds succeeded, with 4,533 fits and 0 convergence warnings.
- The run took 9,547 s with a peak of 1,091 MB.
- Code hash `6fd12bd5…` and configuration hashes were identical before and after the run ([`hashes/`](outputs/stage1_analysis/development/hashes/)).
- Level 1 and Level 2 outputs and the NPZ inputs were verified unchanged after the run.
- The bootstrap used 1,000 subject-cluster resamples, with 0 non-evaluable resamples.

**Results (primary threshold, outer folds, subject-equal, 95 % bootstrap CI, all 18 cells):**

| Model | **Stratified AUROC** | C1 sensitivity | **C3 sensitivity** | C0 FPR | **C2 FPR** | Weighted balanced accuracy |
|---|---|---|---|---|---|---|
| `amplitude_only` | 0.62 [0.41, 0.82] | 0.90 [0.70, 1.00] | 0.91 [0.73, 1.00] | 0.95 [0.85, 1.00] | 0.96 [0.87, 1.00] | 0.48 [0.43, 0.50] |
| `pi_only` | 0.52 [0.37, 0.67] | 0.69 [0.43, 0.92] | 0.77 [0.61, 0.93] | 0.64 [0.40, 0.87] | 0.66 [0.45, 0.85] | 0.54 [0.48, 0.62] |
| `pi_plus_amplitude` | **0.70** [0.60, 0.81] | 0.80 [0.60, 0.95] | **0.82** [0.66, 0.94] | 0.79 [0.57, 0.95] | **0.74** [0.55, 0.89] | 0.52 [0.45, 0.62] |

- **`pi_plus_amplitude` components:** AUROC(C1 vs C0) is 0.72 and AUROC(C3 vs C2) is 0.68.
- **Across the 18 cells:** C2 FPR ranges 0.62–0.79 and C3 sensitivity 0.71–0.87, with little trend over SNR.
- **Per subject:** stratified AUROC ranges 0.46–0.95 (SD 0.18). The four subjects with 25–29 events lie between 0.58 and 0.70.
- **Final Development setting (separate LOSO):**
  - `pi_only`: R = 10, σ = 1 px, C = 0.001
  - `pi_plus_amplitude`: R = 10, σ = 1 px, C = 0.01
  - `amplitude_only`: C = 0.01

**Comparison with Level 2 at the same threshold rule.**
Both models use the C1 rule (90 % C1 sensitivity) and are compared on the same subject, event, condition and cell. The CI is from a paired subject-cluster bootstrap that uses identical resamples for both models. Results at the primary threshold are not compared with Level 2.

| `pi_plus_amplitude`, C1 rule, subject-equal | Level 2 | Factorial | Difference [95 % CI] |
|---|---|---|---|
| AUROC(C2 vs C0), artefact dependence | 0.22 | 0.53 | +0.32 [0.26, 0.37] |
| AUROC(C3 vs C1), artefact dependence | 0.23 | 0.50 | +0.27 [0.22, 0.32] |
| Paired score change f(C2) − f(C0) | −0.18 | 0.02 | +0.20 [0.13, 0.28] |
| Paired score change f(C3) − f(C1) | −0.31 | 0.00 | +0.31 [0.24, 0.36] |
| FPR(C2) − FPR(C0) | −0.30 | −0.01 | +0.28 [0.19, 0.39] |
| Sensitivity(C3) − sensitivity(C1) | −0.25 | 0.01 | +0.26 [0.17, 0.34] |
| C0 FPR | 0.680 | 0.720 | +0.04 [−0.12, 0.20] |
| C2 FPR | 0.383 | 0.707 | +0.32 [0.16, 0.47] |
| C1 sensitivity | 0.832 | 0.796 | −0.04 [−0.30, 0.20] |
| C3 sensitivity | 0.585 | 0.809 | +0.22 [0.03, 0.39] |
| Stratified AUROC | 0.78 | 0.70 | −0.08 [−0.16, −0.01] |
| Weighted balanced accuracy | 0.59 | 0.54 | −0.04 [−0.12, 0.04] |

`pi_only` moves the same way. `amplitude_only` remains artefact-dependent: its AUROC(C2 vs C0) changes from 0.31 to 0.72, and |AUROC − 0.5| does not decrease. All three models are in `comparison_with_level2.csv`.

> **Conclusion: score dependence on artefact decreased, but improved PVC detection was not shown.**
> - **Artefact dependence fell.** For the PI models, the scores of the same event with and without artefact now differ by about 0, and artefact no longer moves the FPR or the sensitivity.
> - **PVC detection did not improve.** The required pattern was lower C0/C2 FPR together with maintained or improved C1/C3 sensitivity. Instead:
>   - C3 sensitivity rose, but C2 FPR rose by a similar amount.
>   - C0 FPR did not fall.
>   - Weighted balanced accuracy stayed at about 0.54.
> - **Discrimination within the same artefact condition fell** (stratified AUROC 0.78 → 0.70). Part of the Level 2 signal was genuine PVC information that the artefact-invariant model gave up.
>
> These are descriptive Development results. Robustness is not claimed.

Outputs are in [`outputs/stage1_analysis/development/factorial/`](outputs/stage1_analysis/development/factorial/). Output hashes are in `hashes/factorial_outputs.sha256`.
- out-of-fold predictions;
- window-level, subject-level and per-subject metrics for both threshold rules and all cells;
- bootstrap CIs;
- the Level 2 comparison;
- the weight check for every fold;
- inner and final selection records;
- PI regions and out-of-range logs;
- exploratory MMD.

The one-fold pilot (outer S105) is in `factorial_pilot/`. `final_models/` is not stored in the repository.

## Stage 1 analysis, Level 3: classical baseline comparison (Development)

This analysis asks whether persistent homology adds information to classical ECG features. Settings are in
[`configs/stage1_level3.yaml`](configs/stage1_level3.yaml) (approvals G-01 to G-06). The training task, weights (F-01),
selection metric (F-02), threshold rules (F-03), nested LOSO, classifier, C grid, tie break and bootstrap are inherited
unchanged from the factorial analysis; only the input features differ.

**Classical features (computed for all 4,788 windows).**
- **Waveforms:** regenerated from the raw records and the Regime 1 injection metadata, then checked against every
  `.npz` input hash (5,040 / 5,040 matched).
- **Annotations:** no MIT-BIH annotation is read for any feature. R peaks come from `xqrs` on the 0.5–40 Hz filtered
  half. Labels and events come only from the frozen Stage 0 audit tables.
- **Feature groups:** each value is computed for the pre half, the post half and their difference.
  - rhythm: beat count, median RR, RR coefficient of variation
  - QRS: NeuroKit2 DWT widths and R amplitude
  - template: correlation and normalized RMS difference against the pre-half template
  - amplitude: the 9 Level 2 features
  - DWT: db4, level 5, relative energies
  - PSD: Welch relative band powers and spectral centroid
  - spectral entropy
  - SQI: bSQI, kSQI, sSQI, pSQI, basSQI
- **Boundary features:** scalars computed on the whole window. They are RR across t0, the minimum post-t0 RR ratio
  (no beat is assumed to be the PVC) and the compensatory ratio.
- **Missing values:** detection and delineation failures are kept as NaN and imputed with the training-fold median.
  Seven 0/1 missing indicators are model inputs.
- **Models (5):**
  - `classical_only`
  - `classical_only_no_boundary`
  - `classical_plus_ph`: classical features + the factorial persistence images
  - `classical_plus_ph_no_boundary`
  - `ph_only`: the factorial `pi_only`
- **Fitting:** all five models are fitted in the same nested LOSO with the same weights and resamples, with no
  block scaling.

**Run:**
- 10 / 10 outer folds, 7,055 fits, 0 failures and 0 convergence warnings.
- The run took 17,138 s with a peak of 1,000 MB, using 1 BLAS thread for all models.
- Code hash `2f6a3586…` and configuration hashes were identical before and after the run.
- Stage 0/1 frozen files, Level 1, Level 2 and factorial outputs and the NPZ inputs were verified unchanged.

**Results (primary threshold, outer folds, subject-equal, 95 % bootstrap CI):**

| Model | **Stratified AUROC** | C1 sensitivity | C3 sensitivity | C0 FPR | C2 FPR | AUROC(C2 vs C0) |
|---|---|---|---|---|---|---|
| `classical_only` | **0.925** [0.888, 0.956] | 0.927 [0.827, 0.992] | 0.901 [0.804, 0.970] | 0.218 [0.079, 0.375] | 0.372 [0.266, 0.498] | 0.630 [0.582, 0.675] |
| `classical_only_no_boundary` | 0.886 [0.830, 0.934] | 0.902 [0.783, 0.988] | 0.887 [0.802, 0.955] | 0.239 [0.081, 0.415] | 0.417 [0.308, 0.546] | 0.623 [0.569, 0.679] |
| `classical_plus_ph` | **0.899** [0.799, 0.968] | 0.898 [0.756, 0.985] | 0.887 [0.756, 0.971] | 0.349 [0.158, 0.587] | 0.364 [0.188, 0.591] | 0.551 [0.482, 0.624] |
| `classical_plus_ph_no_boundary` | 0.904 [0.846, 0.956] | 0.911 [0.772, 0.993] | 0.880 [0.719, 0.979] | 0.405 [0.165, 0.679] | 0.474 [0.274, 0.693] | 0.579 [0.516, 0.643] |
| `ph_only` | 0.521 [0.363, 0.666] | 0.689 [0.433, 0.916] | 0.773 [0.611, 0.925] | 0.639 [0.403, 0.867] | 0.660 [0.451, 0.853] | 0.471 [0.427, 0.516] |

**Paired comparisons (same subjects, events and resamples; difference [95 % CI]):**

| Comparison | Stratified AUROC | C0 FPR | C2 FPR | C1 sensitivity | C3 sensitivity |
|---|---|---|---|---|---|
| **Primary: `classical_plus_ph` − `classical_only`** | **−0.026 [−0.091, 0.021]** | +0.132 [−0.132, 0.431] | −0.008 [−0.186, 0.206] | −0.029 [−0.213, 0.133] | −0.013 [−0.194, 0.142] |
| No boundary: `classical_plus_ph_no_boundary` − `classical_only_no_boundary` | +0.018 [−0.0005, 0.040] | +0.166 [−0.106, 0.482] | +0.057 [−0.134, 0.275] | +0.009 [−0.107, 0.134] | −0.007 [−0.154, 0.102] |
| Boundary effect: `classical_only` − `classical_only_no_boundary` | +0.039 [−0.003, 0.085] | −0.022 [−0.086, 0.040] | −0.045 [−0.102, 0.010] | +0.025 [−0.016, 0.100] | +0.013 [−0.023, 0.059] |
| Boundary effect: `classical_plus_ph` − `classical_plus_ph_no_boundary` | −0.005 [−0.096, 0.061] | −0.056 [−0.147, 0.010] | −0.110 [−0.234, −0.009] | −0.013 [−0.196, 0.150] | +0.007 [−0.173, 0.179] |

- **Artefact dependence:** adding PH lowers AUROC(C2 vs C0) by 0.080 [0.009, 0.151]. However, the paired PVC score
  change f(C3) − f(C1) also falls (−0.043 [−0.061, −0.021]), so this is not a robustness gain.
- **Per subject:** stratified AUROC ranges 0.81–1.00 for `classical_only` (SD 0.06) and 0.53–1.00 for
  `classical_plus_ph` (SD 0.15).
- **Boundary-excluded pair:** this comparison is outside the configured list. It was computed afterwards with the
  same function and bootstrap seed (`level3_audit/model_comparison_no_boundary_pair.csv`).

> **Conclusion: persistent homology did not add PVC-detection value over classical features.**
> - **No improvement from PH.** The primary difference is −0.026 [−0.091, 0.021]: the CI includes 0 and C0 FPR tends to rise.
> - **Same result without boundary features.** The difference is +0.018, its CI includes 0, and C0/C2 FPR rise.
> - **Boundary features raise classical-only performance** (+0.039 subject-equal, +0.098 window-level). This gain is read as a possible alignment shortcut, not as morphology information.
>
> These are descriptive Development results.

**Limitations.**
- **Alignment cue.**
  - In C1/C3 the first detected beat after t0 lies at 0.25 s (IQR 0.250–0.256 s), because events are aligned to
    t0 + 250 ms. In C0 it is spread out (0.006–0.79 s).
  - `boundary_missing` also differs by condition (C0 0.22, C1 0.03).
  - The no-boundary models remove only the boundary scalars. Half-defined rhythm features can still carry alignment
    information, so these models mitigate the shortcut but do not remove every alignment cue.
- **QRS missing indicators.** QRS post-half delineation fails less often under heavy noise. The paired rate change
  C2 − C0 is −0.151, and the rate is 0 % at −6 dB. A successful detection therefore does not mean an accurate QRS
  boundary, and the missing indicator can act as a noise marker.
- **Ten subjects.** Five subjects have only 1–6 events but carry equal weight. Their per-subject FPR and sensitivity
  are coarse (often 0 or 1), and the CIs are wide.
- **C at the grid edge.** The lowest C (0.001) was chosen in 9 / 10 outer folds for `classical_only`, 7 / 10 for
  `classical_only_no_boundary`, 5 / 10 for `classical_plus_ph` and 0 / 10 for `classical_plus_ph_no_boundary`.
  `ph_only` chose the lowest C in 3 folds and the highest in 1. The grid was not changed.
- **Thread environment (reproducibility exception).**
  - `ph_only` does not reproduce the factorial `pi_only` exactly. The maximum OOF difference is 0.076, and the
    selected parameters are equal in every fold.
  - The cause is that the factorial ran with 12 OpenBLAS threads and Level 3 with 1. This changes the lbfgs path in
    weakly regularized PI fits.
  - The effect on overall subject-equal metrics is at most 0.006 (stratified AUROC 0.5215 vs 0.5242). Single cells
    differ by up to 0.10.
  - Details are in [`level3_audit/reproducibility_exception.md`](outputs/stage1_analysis/development/level3_audit/reproducibility_exception.md).

**Outputs.**
- **Main run:** [`outputs/stage1_analysis/development/level3/`](outputs/stage1_analysis/development/level3/)
  contains OOF predictions, window-level, subject-level and per-subject metrics, bootstrap CIs, paired model
  comparisons, weight and imputation logs, and the inner and final selection records.
- **Features:** `level3_features/` holds the classical features and the failure audit. The waveform regeneration
  check is published as a path-sanitized copy in `level3_features/public/`; the original holds local absolute paths
  and is kept only locally.
- **Pilot:** `level3_pilot/` holds the outer-fold S105 pilot.
- **Audit:** `level3_audit/` holds the reproducibility exception, the PH-only fold, candidate and aggregate
  differences, and the BLAS-thread evidence.
- **Hashes:** output hashes are in `hashes/level3_outputs.sha256`.
- **Thread-record corrections:** `outputs/resource_profile_corrections.csv` corrects the thread records of earlier
  runs, marking confirmed and inferred values separately.
- **Not stored:** `final_models/` is not stored in the repository.

## Installation

Developed and tested on Python 3.14 (Windows 11). Other Python versions have not been verified.

```bash
git clone https://github.com/justinbrianhwang/TopoVital.git
cd TopoVital
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.lock.txt
# Linux / macOS
.venv/bin/python -m pip install -r requirements.lock.txt
```

[`requirements.txt`](requirements.txt) lists the direct dependencies. [`requirements.lock.txt`](requirements.lock.txt) pins the exact versions that were installed and import-verified, including numpy, scipy, pandas, pyarrow, wfdb 4.3.1, neurokit2 0.2.13, gudhi 3.13.0 and scikit-learn.

## Data

The raw data are **not** included. Download the official ZIP archives from PhysioNet (no credentials required) into `data/`:

| Dataset | Version | Save as |
|---|---|---|
| [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) | 1.0.0 | `data/mitdb-1.0.0.zip` |
| [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/) | 1.0.0 | `data/mit-bih-noise-stress-test-database-1.0.0.zip` |

`python -m topovital fetch` extracts them into `data/raw/`. It never overwrites existing files, and it verifies every file against the distribution's own `SHA256SUMS.txt`. Only the raw NSTDB noise records `bw`, `ma` and `em` are used; the pre-mixed `118e*/119e*` records are not.

## Usage

```bash
python -m topovital env-check                        # environment.json + requirements.lock.txt
python -m topovital fetch                            # extract + SHA256 verification
python -m topovital cohort                           # 48/47 -> 44/43, MLII by header name, step-1 PVC list
python -m topovital split                            # stratified 22/21 subject split (immutable)
python -m topovital stage0 --split development --g-sweep
python -m topovital freeze-stage0 --decision configs/g_decision_reviewed.yaml
python -m topovital stage0 --split test --config outputs/locked_config.yaml
python -m topovital report-stage0
python -m pytest -q
```

A command that is not allowed in the current state exits with code `3` and prints the reason. For example, a Test run before freezing, or any Test `g` sweep. Each command appends a row to `outputs/resource_profile.csv`.
On real data the Development sweep took about 230 s with a peak RSS of about 420 MB on a 10-core / 32 GB CPU machine.

**Stage 1** (Development only; refuses to run unless Stage 0 is frozen with `g_star = 250` and the approved parameters are unchanged):

```bash
python -m topovital_stage1 pilot [--only embedding|ph]   # embedding diagnostics + PH resource pilot
python -m topovital_stage1 features --workers 4          # C0/C1 persistent homology (resumable, cached per job)
python -m topovital_stage1 regime1 --workers 4           # C2/C3 Regime 1 injection (needs approved_for_full_run)
```

```bash
python -m topovital_analysis level1 --workers 4 [--events N]   # Level 1 pre/post PH distances
python -m topovital_analysis level2 [--pilot]                  # Level 2 persistence images, nested LOSO (needs approval)
python -m topovital_analysis factorial [--pilot]               # factorial C0-C3 training, nested LOSO (needs approval)
python -m topovital_analysis level3-features --workers 4       # Level 3 classical features (waveforms regenerated and hash-checked)
python -m topovital_analysis level3 [--pilot]                  # Level 3 classical baseline comparison, nested LOSO (needs approval)
```

`--events N` runs a deterministic subset with one event per subject. Every VR job runs in a monitored child process with a 6 GB abort limit. Each job is cached by the SHA-256 of its normalized input, parameters, GUDHI version and Stage 1 code hash, so an interrupted run resumes without recomputation.

## Method summary (Stage 0)

<p align="center">
  <img src="assets/figs/fig2_event_window_timing.png" width="900" alt="One 10 s analysis window: pre-window 0 to 5 s with all normal beats, analysis boundary t0 at 5 s, guard interval of 250 ms, target PVC R-peak at t0 plus g star, QRS onset between t0 and the R-peak, post-window 5 to 10 s">
  <br><em>Figure 4. Event-window timing. The boundary t0 sits at 5 s and the target PVC R-peak at t0 + g* (250 ms), so the QRS onset falls after t0.</em>
</p>

1. **Candidates:** every `V` beat annotation on the 44 retained records. The 10 s window has its boundary at `t0 = 5 s` and the target R peak at `t0 + g`.
2. **Pre-boundary `[0, t0)`:** all beat annotations are `N`. There must be no run of ≥ 3 samples at the digital ADC limits 0/2047. `IQR ≥ 1e-4 mV`. `bSQI ≥ 0.8`, where bSQI compares `wfdb.processing.xqrs_detect` and NeuroKit2 Pan–Tompkins detections after a 0.5–40 Hz 4th-order Butterworth band-pass (zero-phase), with greedy one-to-one matching within ±150 ms.
3. **Post-boundary `[t0, W)`:** exactly one `V` (the target) and all other beats `N`. No saturation and no annotated signal loss.
4. **Overlap removal:** chronological; the earlier event is kept and the next one needs a separation of at least 10 s. **Normal matching:** a clean 10 s window from the same subject is used once. Its pre-boundary median RR must be within ±15 % of the event's. It may not overlap any PVC window together with that window's 60 s lookback, nor any other selected normal window. The nearest window in time is chosen, and ties go to the earlier window.
5. **Common cohort** = matched pairs. **Regime 2** = the subset with ≥ 10 normal template beats in a 60 s lookback whose `[−250, +450] ms` windows end before `t0`.
6. **Guard interval:** NeuroKit2 DWT QRS onset on Development only. `g_star` is the smallest `g ∈ {200, 250, 300} ms` for which ≥ 95 % of valid onsets satisfy `R − onset ≤ g`, confirmed by visual waveform review.

<p align="center">
  <img src="assets/figs/fig4_normal_window_matching.png" width="900" alt="Normal-window matching on a 30-minute record: PVC windows occupy a 60 s lookback; candidate normal windows overlapping a lookback or a selected normal window are rejected; the matched window has the same subject, pre-window median RR within 15 percent, nearest in time">
  <br><em>Figure 5. Normal-window matching (step 4). Schematic, not to scale.</em>
</p>

## Repository layout

```
assets/figs/  README figures
configs/            stage0.yaml (all Stage 0 settings), g-decision records,
                    stage1.yaml (PH settings), stage1_regime1.yaml (artefact injection)
                    stage1_analysis.yaml (Level 1/2), stage1_factorial.yaml (factorial C0-C3),
                    stage1_level3.yaml (Level 3 classical baseline)
topovital/          Stage 0 pipeline package (frozen)
topovital_stage1/   Stage 1 package: embedding, persistent homology, Regime 1 injection
topovital_analysis/ Stage 1 analysis: Level 1 PH distances, Level 2 persistence images, factorial C0-C3,
                    Level 3 classical features and baseline comparison, nested LOSO
tools/              publication helpers: path-sanitized audit copies, cache config-hash comparison
tests/              unit + real-data reproducibility tests (pytest); tests/stage1/ for Stage 1
outputs/            Stage 0 deliverables: cohort_manifest.csv, event_audit*.parquet, attrition_summary.csv,
                    locked_config.yaml, implementation_decisions.md, stage0_report.md, resource_profile.csv, ...
outputs/stage1_pilot/      embedding diagnostics and PH resource pilot (Development)
outputs/stage1_features/   Stage 1 job ledgers and run summaries (per-job .npz diagrams archived on Zenodo)
outputs/stage1_status.yaml Stage 1 progress record
outputs/stage1_analysis/   Level 1 / Level 2 / factorial / Level 3 analysis outputs (Development)
```

## License and data attribution

The code is licensed under the [Apache License 2.0](LICENSE). Copyright © 2026 Sunjun Hwang.

The MIT-BIH Arrhythmia Database and the MIT-BIH Noise Stress Test Database are distributed by PhysioNet under the Open Data Commons Attribution License v1.0. If you use them, please cite:

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 20(3):45–50 (2001).
- Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. *Computers in Cardiology* 11:381–384 (1984).
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).

