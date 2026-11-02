# Level 3 reproducibility exception: PH-only vs factorial `pi_only`

Recorded 2026-11-02. Approved by the principal investigator, who chose to accept the single-thread Level 3 run
instead of re-running it under the factorial's thread setting.

## Requirement not met

Approval G-04 required the Level 3 `ph_only` model to reproduce the factorial `pi_only` out-of-fold (OOF) results
exactly (`configs/stage1_level3.yaml`, `ph_reproducibility`, tolerance 1e-12). The full Level 3 run
(run_id `20261102T024744Z_59b9f919`) does not meet this requirement:

| Item | Compared | Max abs difference | Equal |
|---|---|---|---|
| OOF score | 4,788 windows | 0.076 | no |
| Primary threshold | 4,788 | 0.0032 | no |
| C1-rule threshold | 4,788 | 0.0016 | no |
| Selected resolution, sigma, C | 10 outer folds | 0 | yes |
| Inner candidate scores | 200 | 0.026 | no |
| Final Development selection record | 22 values | 7.9e-15 | yes |

Source: `level3/ph_reproducibility_check.csv` (written by the run) and `run_summary.json`
(`ph_reproducibility_passed: false`).

## Cause

The cause is the number of OpenBLAS threads. Code and data played no part.

- `topovital_analysis/__main__.py` sets `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_MAXIMUM_THREADS=1` only after numpy has
  been imported, so the setting has no effect on BLAS.
- The factorial run and the Level 3 pilot were launched without these variables in the shell, so they ran with the
  machine default of 12 OpenBLAS threads.
- The full Level 3 run was launched with the variables exported as 1 in the shell, so it ran with 1 thread.
- A different thread count changes the floating-point summation order, so lbfgs follows a different path.
- The effect grows as the PI models (thousands of columns) get weaker regularization:

| Selected C | Folds | Max abs OOF difference |
|---|---|---|
| 0.001 | S116, S219, S221 | ≤ 4e-12 |
| 0.01 | S123, S213 | ≤ 8e-7 |
| 0.1–1 | S105, S215, S233, S234 | 3e-3 to 2e-2 |
| 10 | S210 | 0.076 |

Evidence:

- `s210_blas_thread_check.csv` (script `s210_blas_thread_check.py`) refits the S210 outer-fold `ph_only` model
  (R 10, sigma 2, C 10), two repeats per setting:
  - 12 threads: 884 lbfgs iterations; equals the factorial OOF exactly (max difference 0.0); differs from Level 3
    by 0.076.
  - 1 thread: 832 iterations; equals the Level 3 OOF exactly (0.0); differs from the factorial by 0.076.
  - Each setting is deterministic: repeats are bit-identical. The same result was also obtained in an earlier,
    unsaved check.
- PI-free models do not depend on the thread count. For S105, the pilot (12 threads) and the full run (1 thread)
  give identical OOF scores for `classical_only` and `classical_only_no_boundary` (difference 0). For the
  Classical+PH models the OOF scores differ by at most 4e-11, but their inner candidate scores differ by up to
  0.006 (`thread_environment_check.csv`).

## All five Level 3 models share one environment

All five models come from one process, one run and one `oof_predictions.parquet`, launched with the variables
exported as 1. threadpoolctl reports 1 OpenBLAS and OpenMP thread under these exports, and the 1-thread refit
reproduces Level 3 exactly (`thread_environment_check.csv`). The within-Level-3 comparisons are therefore made
under identical numerics.

## Numerical sensitivity of reported results

Differences between Level 3 `ph_only` (1 thread) and factorial `pi_only` (12 threads); source
`ph_only_aggregate_diff.csv`.

**Overall, subject-equal, primary rule:**

| Metric | Level 3 | Factorial | Difference |
|---|---|---|---|
| Stratified AUROC | 0.5215 | 0.5242 | −0.0027 |
| C3 sensitivity | 0.7726 | 0.7744 | −0.0019 |
| C2 FPR | 0.6598 | 0.6600 | −0.0002 |
| C0 FPR | 0.6395 | 0.6395 | 0 |
| C1 sensitivity | 0.6891 | 0.6891 | 0 |

- **Overall, any metric:** max |difference| 0.0057 for subject-equal aggregation and 0.0005 for window-level
  aggregation.
- **Single cells:** up to 0.10 (AUROC C3 vs C2 in ma 0 dB) and 0.05 (stratified AUROC in ma 0 dB). Per-cell results
  of PI models with weakly regularized fits should be read with this sensitivity in mind.
- Fold-level and candidate-level differences are in `ph_only_fold_oof_diff.csv` and
  `ph_only_inner_candidate_diff.csv`.

## Decision and follow-up

- The single-thread Level 3 results are accepted as the Level 3 results, under this exception. No 12-thread
  re-run was made.
- The thread-variable timing bug will be fixed in a separate code version after the Level 3 results are final.
- The resource-profile thread records of earlier runs are corrected in `outputs/resource_profile_corrections.csv`.
  The original `outputs/resource_profile.csv` is unchanged.
- In that file, confirmed values are kept separate from inferred ones. Confirmed: factorial full run 12 threads;
  Level 3 full run 1 thread; Level 3 pilot not 1 thread.
- No Stage 0/1 frozen file, no Level 1, Level 2 or factorial output, no NPZ file and no audit table was modified.
