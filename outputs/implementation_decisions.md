# Implementation decisions — TopoVital v1.3 Stage 0 (CPU)

Status legend: **RESOLVED-brief** = stated in the CPU execution brief and implemented as written ·
**RESOLVED-data** = verified against the official data or documentation · **IMPL** = implementation choice without
scientific effect · **APPROVED** = left unspecified by the protocol and explicitly approved by the principal investigator.

## D-00 (RESOLVED 2026-10-31) — v1.3 protocol received
The protocol document (kept private; not published in this repository)
(SHA256 `9da3011c7bce677882c24b04775f8913e4dab04a2442bf4fda3786a8858612e6`) was received and transferred into
`configs/stage0.yaml`. Values stated in the protocol are tagged `[P§n]`. Items that protocol §9 leaves as
"unspecified, approval required" were approved by the principal investigator as A-01..A-05 below.
The first run, made before the protocol was available, is kept as history at the end of this document.

## Approved items (unspecified in protocol §9 → approved by the principal investigator on 2026-10-31)

| ID | Protocol basis | Approved decision |
|---|---|---|
| A-01 | §9-1 seed unspecified | `split.seed = 20261031`, approved before the split was executed |
| A-02 | §9-2 stratification unspecified | Three strata by subject-level count of step-1 `V` candidates: 0 (11 subjects) / 1–88 (16) / ≥ 89 (16). The median of 88 is computed over the 32 subjects with ≥ 1 candidate. The Development quota per stratum is floor(n·22/43) plus largest-remainder allocation, giving 6/8/8 = 22 (Test 5/8/8 = 21). Only g-independent step-1 information is used |
| A-03 | §9-3 QRS-onset algorithm and g rule unspecified | `ECG_R_Onsets` from `neurokit2.ecg_delineate(method="dwt")`. g rule: the smallest g whose Development common-cohort events satisfy (R − onset) ≤ g in ≥ 95 % of events with a valid onset; 300 ms if none qualifies. **The researcher must complete a visual waveform review on the generated figures before freezing** |
| A-04 | §4 (unspecified details), §9-4 | Bulk approval of the following defaults. **IQR:** numpy linear interpolation (type 7), MLII in mV. **Filter:** Butterworth order 4 as SOS with zero-phase `sosfiltfilt` (effective order 8). **bSQI detector context:** the 5 s pre-window only. **bSQI matching ties:** sorted two-pointer, earliest first. **Saturation:** 0/2047 as written; header mismatches are recorded as diagnostics only. **Overlap removal:** chronological; keep the earlier event and drop the next one if its start < previous end + 10 s. **Normal windows:** same subject, 10 s, 1 s grid, same pre-window criteria plus all-N post-window and no saturation. **Signal loss:** `~` subtype −1 state. **"Other arrhythmia" in the post-window:** every beat other than the single V must be N |
| A-05 | §2-4 interpretation of "(including lookback)" | **Only PVC windows occupy a 60 s lookback** `[start − 60 s, end)`; normal windows occupy only their own `[start, end)`. Approved 2026-10-31. Rationale: the protocol defines the 60 s lookback only for the Regime 2 template of PVC events. Before approval, Development match counts for the three readings were reported (g = 250: lookback on both → 23, PVC only → 126, none → 186). Test had not been accessed |

### Interpretations of protocol wording (fixed during implementation within A-04, reported separately)
- I-01 (superseded by A-05): the first implementation let both PVC and normal windows occupy `[start − 60 s, end)`.
  The first Development sweep showed that this reading caused most of the matching attrition (g = 250: 186 → 23),
  so a decision was requested from the researcher.
- I-02 Scope of the signal-loss criterion: PVC events use the post-window only, as in the protocol; normal windows use the whole window.
- I-03 Cross-record matching for 201/202: allowed only when no candidate exists in the same record; the earliest candidate in the other record is chosen.
- I-04 bSQI with a zero denominator → 0 (protocol §4). The pre-protocol implementation (D-08: NaN) was corrected.
- I-05 Normal-matching criterion: **pre-boundary median RR within ±15 %** as in the protocol (not heart rate), nearest in time,
  earlier on ties. The pre-protocol implementation (D-09: heart-rate difference first) was replaced.

## Resolved items

| ID | Item | Decision | Status |
|---|---|---|---|
| D-01 | Data source | Official PhysioNet ZIPs (mitdb 1.0.0, nstdb 1.0.0) placed in `data/`. Every file is verified against the distribution's own `SHA256SUMS.txt` (mitdb 704, nstdb 97, 0 mismatches). The ZIP SHA256 is recorded as well. No network download was performed | RESOLVED-data |
| D-02 | Subject groups | 201/202 → `S201`; otherwise `S<record>`. Official documentation (mitdbdir/intro.htm): "Records 201 and 202 came from the same male subject" | RESOLVED-brief/data |
| D-03 | ms → samples | `round(g_ms·fs/1000)`, computed exactly on rationals with round-half-up. 200/250/300 ms map exactly to 72/90/108 samples, so the rounding rule has no effect. Realized: 200.0/250.0/300.0 ms | IMPL |
| D-04 | Boundary membership | All intervals are `[start, end)`. Pre-window = `[start, boundary)`, post-window = `[boundary, end)`, so an annotation exactly on the boundary belongs to the **post-window** | IMPL |
| D-05 | Split ordering | Unique subject_ids sorted ascending; one `numpy.random.default_rng(seed)` used across strata in the order zero, low, high | IMPL |
| D-06 | Lead selection | Selected by header `sig_name == "MLII"`. Verified: record 114 has MLII on channel 1; 102/104 have no MLII (both excluded). All 44 cohort records pass the MLII assertion | RESOLVED-data |
| D-07 | Beat vs non-beat | Only the PhysioNet standard beat codes `N L R B A a J S V r F e j n E / f Q ?` are beats. `+ ~ \| ! [ ] x "` and others are non-beat (rhythm, quality, comment) annotations. Observed non-beat symbols: `! " + [ ] x \| ~` | RESOLVED-data |
| D-07b | Signal-quality annotations | Official documentation (mitdbdir/intro.htm): quality change `qq` (first character = upper signal, second = lower signal, c/n); `U` = both signals unreadable. Stored as the WFDB `~` subtype bitmask: bit0 = signal 0, bit1 = signal 1, −1 = U. A state **persists until the next `~`**. Record 114 (MLII on channel 1) uses bit1. Raw subtypes are kept in `signal_quality_annotations.csv`. The absence of an annotation is not treated as ground truth for a noise-free signal | RESOLVED-data |
| D-08 | bSQI edge cases | F1 = 2M/(N_a+N_b); 0 if either or both lists are empty (protocol §4, see I-04) | RESOLVED |
| D-08b | bSQI one-to-one matching | Both detection lists are stably sorted and matched by a two-pointer sweep, earliest first (|Δ| ≤ tol inclusive). This gives a maximum matching for 1-D tolerance windows and never reuses a point. tol = round(150 ms·360/1000) = 54 samples | IMPL |
| D-09 | Normal-matching ties | Replaced by the protocol rule (I-05) | RESOLVED |
| D-10 | Template beats | `N` only; lookback `[boundary − 60 s, boundary)`; `r_j + 162 ≤ boundary_abs` (450 ms); `r_j − 90 ≥ 0` (250 ms) | APPROVED (A-04) |
| D-11 | Saturation diagnostics | Header adc_res = 11 → range [0, 2047]. Record 116 MLII has 663 samples at 0 and 265 at 2047; 0 samples lie outside the range. No automatic correction | RESOLVED-data |
| D-12 | Noise sources | NSTDB `bw`, `ma`, `em` only (360 Hz, 650,000 samples, 2 channels verified). The pre-mixed 118e*/119e* ECGs are not used. Development block = first half, Test block = second half | RESOLVED |
| D-13 | Execution resources | `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_*_THREADS=1`, n_jobs = 1, record-by-record reading. Not a scientific setting | IMPL |
| D-14 | Python environment | The system Python 3.14.4 lacked wfdb, neurokit2 and gudhi, so they were installed into a project `.venv` (system unchanged). Exact versions are in `requirements.lock.txt` | IMPL |
| D-15 | Scope of the step-1 audit | Step 1 (the V candidate list) does not depend on g or the split, so it is built for all 44 records. g-dependent columns and step 2–6 columns stay null there, so no g comparison on Test records can occur before the split | IMPL |
| D-16 | Split stratification value | `null` = pending (blocked); `'none'` only if the protocol specified an unstratified split; `pvc_candidate_count_3strata` = A-02 | IMPL |

## Execution notes
- The first pytest run had one failure. It was a **test bug**: the test read a UTF-8 CSV with the Windows default codec (cp949). The gate itself behaved correctly (exit code 3). After specifying `encoding="utf-8"` all tests passed.
- `fetch` reuses already-extracted files (0 newly extracted) and re-verifies all hashes, so `cache_used=True` appears in the resource profile.
- Local user-profile paths in `environment.json`, `resource_profile.csv` and `stage0_report.md` were redacted to `C:\Users\<user>` before publication. Frozen records were not modified.

## Stage 0 freeze and Test processing (2026-10-31)
- **F-01 Freeze:** g_star = 250 ms, decided by the principal investigator (name entered by the researcher; see
  `configs/g_decision_reviewed.yaml`). Immediately before freezing, `dev_audit_sha256`, `config_hash` and `code_hash`
  were re-verified against the current files. Freeze record: `outputs/stage0_freeze_record.json`; locked configuration:
  `outputs/locked_config.yaml` (frozen).
- **F-02 Single Test pass and failure:** `stage0 --split test` was executed once (run_id `20261031T203554Z_a8af0552`).
  `event_audit_test.parquet` (SHA256 `7634de2d…a31d`, all 3,610 Test step-1 candidates, g = 250 only) was written
  correctly. The summary step that followed then ran Development-only onset-coverage code on Test and stopped with an
  `AttributeError` (exit 1). **Test was not re-run, and neither code nor configuration was changed.**
  `attrition_test.csv` and `per_subject_counts_test.csv` were produced by a read-only aggregation of the saved audit,
  using the same cumulative drop order as the pipeline; the `note` column records this recovery. Fixing the pipeline
  bug (separating the Development-only block) changes `code_hash` and must be handled as a post-freeze change.

## History: first run before the protocol was available
Before the protocol was received, every protocol-only value in `configs/stage0.yaml` was left `null` and the split,
Development sweep, freeze and Test commands refused to run (exit code 3). That run produced only the
protocol-independent outputs: data integrity, the cohort assertions and the step-1 candidate list. All of those were
later re-verified unchanged (event list SHA256 `191f135d…f035`).
- **AM-01 Amendment to the freeze records (2026-10-31, approved by the principal investigator):** the `decided_by`
  value was romanized to "Sunjun Hwang, Principal Investigator / Research Lead", and the absolute local path of the
  decision file was replaced by `configs/g_decision_reviewed.yaml`, in the decision file,
  `outputs/stage0_freeze_record.json` and `outputs/locked_config.yaml`. The decision-file SHA256 changed from
  `d6da7ea0f810…` to `b136ff474c5a…`; both values are recorded in the `amendments` field of the freeze record.
  g_star, the stage-0 settings, `stage0_config_hash`, `code_hash` and all results are unchanged, and the Test gate still validates.

## Stage 1 — Development pilot and approved feature settings (2026-10-31)
The pilot (`python -m topovital_stage1 pilot`, run_id `20261031T212507Z_2dcb6336`) used the 126 Development
common-cohort events at g* = 250 ms only; no Test data were read. Stage 0 code was not modified (`code_hash` unchanged).
Embedding diagnostics over 504 segments: τ (AMI first minimum) was found in 497 segments, with median 28 samples
(IQR 18–51). m (FNN < 5 %) was 5 in 444 segments, 7 in 46 and 3 in 14. VR resource pilot (20 jobs, 300 points,
max_dimension 2, no r_max): all succeeded, 4,500,250 simplices per job, 13.5 s mean, about 488 MB peak RSS per job.

| ID | Protocol basis | Approved decision |
|---|---|---|
| B-01 | §6 τ/m "selected on Development"; aggregation unspecified | τ = 28 samples (pooled median of valid Dev segments), m = 5 (mode). NeuroKit2 `fraser1986` AMI with library defaults; FNN Rtol 15, Atol 2, < 5 %, m ∈ {3, 5, 7} |
| B-02 | §6 "subsample 300 points per half"; method unspecified | Uniform stride `np.linspace(0, n−1, 300).round()` |
| B-03 | §6 "fix maximum filtration radius"; rule unspecified | No truncation (full Rips complex). Cost measured as acceptable; no right-censored bars, and exactly one essential H0 bar |

Settings are recorded in `configs/stage1.yaml`.

## Stage 1 — Regime 1 artefact injection, Development (2026-10-31)
Validation on 3 representative Development event pairs (run_id `20261031T225455Z_3acff931`): 108/108 post-half
jobs succeeded. SNR error was at most 8.9e-15 dB and the relative norm error at most 7.3e-16. Pre-half reuse was
verified by SHA256 in 108/108 jobs, and no snippet was reused. The full run was then approved.

| ID | Item (not specified by the protocol) | Approved decision |
|---|---|---|
| C-01 | NSTDB noise channel | `noise1` (channel 0); the nst convention adds noise signal i to ECG signal i, and MLII is signal 0 for all Development records |
| C-02 | Snippet allocation | Per noise type, 180 non-overlapping 1,800-sample slots in the Development block [0, 325000), permuted with `default_rng([20261031, type_index])` and assigned without replacement to the 126 event pairs sorted by event_id. The same snippet is used for C2/C3 and for all SNR levels of a pair |
| C-03 | Onset | Raised-cosine ramp over 100 ms (36 samples) at t0. The hard-onset (η = 0) variant is excluded from this run |

Principles set by the principal investigator: NSTDB bw/ma/em with the Development/Test block split; scale factors
computed separately for C2 and C3; 5 s snippets with mean removal; SNR verified on the injected waveform
(< 0.001 dB); SNR ∈ {24, 18, 12, 6, 0, −6} dB; pre-half PH reused from Stage 1; only post halves computed;
4 workers; resumable; Test locked; no Regime 2/3 or classifier. Pre-run configuration and code hashes are in
`outputs/stage1_features/development/hashes/regime1_full_run_prerun.json`.

### Regime 1 full Development run and publication (2026-11-01)

- **Result:** run_id `20261031T232737Z_3634d895`, 4,536/4,536 post-half jobs succeeded and validated, 0 failed,
  4,536/4,536 SNR checks passed, 4,536/4,536 pre-half reuses verified by SHA256, 108 jobs reused from the
  pre-validation cache. Wall time 17,218.5 s with 4 workers; max child peak RSS 490.3 MB.
- **Independent checks** (not using `topovital_stage1`): snippet allocation re-derived from the seed, 378/378 match;
  SNR and scale factor recomputed from raw MIT-BIH/NSTDB for 60 random jobs, max |SNR error| 7.1e-15 dB.
- **Cache config hash:** the 108 cached jobs carry run config hash `3acff931…`, the rest `3634d895…`. A normalized
  flattened comparison shows the only differing key is `regime1.approved_for_full_run` (false at validation, true
  for the full run); both hashes are reproduced exactly and all 108 cache keys match their job files
  (`outputs/stage1_features/development/hashes/regime1_cache_config_diff.json`).
- **P-01 Publication copies:** the original `regime1_ledger.csv` and `regime1_injection_log.csv` contain local
  absolute paths (user name) in `job_path` and `pre_reference`. They are kept unchanged locally and are not
  published. Public copies with repo-relative paths, all other values identical, are in
  `outputs/stage1_features/development/public/`, with source and public SHA256 in `PUBLIC_MANIFEST.json`
  (`tools/make_public_copies.py`). Per-job `.npz` files are not published in the repository.
- **Status record:** `configs/stage1.yaml` is unchanged to preserve its recorded hash; current progress is recorded in
  `outputs/stage1_status.yaml` (Regime 2/3, persistence images and classifiers not started; Test locked for Stage 1).
