# TopoVital

**Reproducible, CPU-only pipeline for topological analysis of premature ventricular contractions (PVCs) in ECG.**

TopoVital studies whether topological data analysis (persistent homology) can separate a PVC-induced perturbation from
noise, baseline wander, clipping and time-warping artefacts in short ECG windows. This repository implements and
documents **Stage 0** of the experimental protocol: cohort construction, event eligibility, normal-window matching,
guard-interval (`g_star`) selection on Development data only, and a locked Development/Test gate. It also contains
the **Stage 1** topological feature extraction, which is in progress on Development data.

> **Status (2026-11-01):**
> - **Stage 0** is complete and **frozen** with `g_star = 250 ms`. The Test split was processed exactly once with the frozen configuration.
> - **Stage 1 (Development only):**
>   - Clean-window persistent homology (C0/C1) is complete: 504 / 504 jobs succeeded.
>   - Regime 1 artefact injection (C2/C3, NSTDB `bw` / `ma` / `em`) is complete: 4,536 / 4,536 jobs succeeded.
> - Test remains locked for Stage 1. Regime 2/3, persistence images, classifiers and statistical evaluation (protocol §7) have not started.
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
- **CPU-only.** No GPU/CUDA/PyTorch/TensorFlow dependency. BLAS/OpenMP threads pinned to 1, record-by-record processing.

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
topovital/          Stage 0 pipeline package (frozen)
topovital_stage1/   Stage 1 package: embedding, persistent homology, Regime 1 injection
tools/              publication helpers: path-sanitized audit copies, cache config-hash comparison
tests/              unit + real-data reproducibility tests (pytest); tests/stage1/ for Stage 1
outputs/            Stage 0 deliverables: cohort_manifest.csv, event_audit*.parquet, attrition_summary.csv,
                    locked_config.yaml, implementation_decisions.md, stage0_report.md, resource_profile.csv, ...
outputs/stage1_pilot/      embedding diagnostics and PH resource pilot (Development)
outputs/stage1_features/   Stage 1 job ledgers and run summaries (per-job .npz diagrams archived on Zenodo)
outputs/stage1_status.yaml Stage 1 progress record
```

## License and data attribution

The code is licensed under the [Apache License 2.0](LICENSE). Copyright © 2026 Sunjun Hwang.

The MIT-BIH Arrhythmia Database and the MIT-BIH Noise Stress Test Database are distributed by PhysioNet under the Open Data Commons Attribution License v1.0. If you use them, please cite:

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol* 20(3):45–50 (2001).
- Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. *Computers in Cardiology* 11:381–384 (1984).
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).

