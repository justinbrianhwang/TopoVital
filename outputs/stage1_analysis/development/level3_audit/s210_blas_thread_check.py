"""Audit evidence for the Level 3 reproducibility exception (see reproducibility_exception.md).

Refits the outer-fold ph_only / pi_only model for one held-out Development subject (default S210, selected
R 10, sigma 2 px, C 10) twice under the BLAS thread setting of the calling shell, and compares the scores with the
Level 3 and factorial out-of-fold predictions.

Usage, from the repository root (OPENBLAS threads come only from the shell environment):
    python outputs/stage1_analysis/development/level3_audit/s210_blas_thread_check.py S210 default_unset out.csv
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python outputs/stage1_analysis/development/level3_audit/s210_blas_thread_check.py S210 exported_1 out.csv
Arguments: held-out subject, a free-text setting label, optional output CSV path.

Requirements: the pinned environment (requirements.lock.txt; threadpoolctl 3.7.0), the raw Development records and the
Stage 1 Development .npz job files (not in the repository), and the Level 3 and factorial OOF files. Produced
s210_blas_thread_check.csv with analysis code hash 2f6a3586... on a 12-logical-CPU machine, where the unset default is
12 OpenBLAS threads. Writes nothing except the optional CSV.
"""
import sys, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from threadpoolctl import threadpool_info
from topovital.util import PROJECT_ROOT, load_yaml
from topovital_stage1.core import load_dev_events
from topovital_analysis.level2 import _load_windows, _feature_matrix, _fit_score
from topovital_analysis.factorial import _arrays, factorial_weights
from topovital_analysis.persistence_images import estimate_regions

print("BLAS", [(i["internal_api"], i["num_threads"]) for i in threadpool_info()], "env", os.environ.get("OPENBLAS_NUM_THREADS"))
root = PROJECT_ROOT
cfg_a = load_yaml(root / "configs/stage1_analysis.yaml")
events = load_dev_events(root / "outputs/development/event_audit_dev.parquet", root / "outputs/split_assignment.csv")
windows = _load_windows(events, cfg_a, root)
held = sys.argv[1] if len(sys.argv) > 1 else "S210"
D = root / "outputs/stage1_analysis/development"
new = pd.read_parquet(D / "level3/oof_predictions.parquet"); new = new[(new.model == "ph_only") & (new.subject_id == held)]
old = pd.read_parquet(D / "factorial/oof_predictions.parquet"); old = old[(old.model == "pi_only") & (old.subject_id == held)]
cand = (int(new.resolution.iloc[0]), float(new.sigma_pixels.iloc[0]), float(new.C.iloc[0]))
reps = list(cfg_a["inputs"]["representations"]); quantiles = cfg_a["level2"]["persistence_image"]["range_quantiles"]
train = [i for i, w in enumerate(windows) if w["subject_id"] != held]
test = [i for i, w in enumerate(windows) if w["subject_id"] == held]
regions = estimate_regions(windows, [i for i in train if windows[i]["condition"] in ("C0", "C1")], reps, quantiles, excluded_subject=held)
y = np.asarray([windows[i]["condition"] in ("C1", "C3") for i in train])
w = factorial_weights(*_arrays(windows, train))
a = _feature_matrix(windows, train, "pi_only", regions, cand, reps); b = _feature_matrix(windows, test, "pi_only", regions, cand, reps)
keys = ["event_id", "condition", "noise_type", "snr_target"]
info = threadpool_info()
rows = []
for rep in range(2):
    s, est = _fit_score(a, y, w, b, cand[2], [])
    f = pd.DataFrame([{**{k: windows[i][k] for k in keys}, "s": v} for i, v in zip(test, s)])
    jn = f.merge(new, on=keys); jo = f.merge(old, on=keys)
    rows.append({"setting": sys.argv[2] if len(sys.argv) > 2 else "", "held_out_subject": held,
                 "resolution": cand[0], "sigma_pixels": cand[1], "C": cand[2], "repeat": rep + 1,
                 "openblas_threads": ";".join(str(i["num_threads"]) for i in info if i["internal_api"] == "openblas"),
                 "openmp_threads": ";".join(str(i["num_threads"]) for i in info if i["internal_api"] == "openmp"),
                 "OPENBLAS_NUM_THREADS_env": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                 "lbfgs_n_iter": int(est[-1].n_iter_[0]), "n_windows": len(f), "n_matched_level3": len(jn), "n_matched_factorial": len(jo),
                 "max_abs_diff_vs_level3_oof": float((jn.s - jn.score).abs().max()),
                 "max_abs_diff_vs_factorial_oof": float((jo.s - jo.score).abs().max())})
    print(rows[-1])
if len(sys.argv) > 3:
    pd.DataFrame(rows).to_csv(sys.argv[3], index=False, lineterminator="\n")
