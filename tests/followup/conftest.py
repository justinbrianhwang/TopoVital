import numpy as np
import pandas as pd
import pytest

from topovital.util import load_yaml, PROJECT_ROOT

# Legacy tests import the top-level conftest as a module. Pytest can resolve this
# directory's conftest first during whole-suite collection, so keep those exports.
MITDB = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"
MITDB_RECORDS = ("100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 "
                 "122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 "
                 "222 223 228 230 231 232 233 234").split()
requires_data = pytest.mark.skipif(not (MITDB / "RECORDS").exists(), reason="real MIT-BIH data not extracted")


def synthetic(n_subjects=5, events_per_subject=2, seed=0, order="sorted"):
    rng = np.random.default_rng(seed)
    windows, rows = [], []
    reps = ("cub_sub_h0", "cub_super_h0", "vr_h0", "vr_h1")
    for s in range(n_subjects):
        for event in range(events_per_subject):
            event_id, subject_id = f"E{s:02d}_{event}", f"S{s:02d}"
            clean = {}
            for condition in ("C0", "C1"):
                label = int(condition == "C1")
                signal = label * 0.85 + rng.normal(0, 0.65)
                def diagram(offset):
                    return {rep: np.array([[0.0 if rep == "vr_h0" else .1 + .02 * s,
                                             .5 + .04 * event + .03 * label + offset],
                                            [0.0 if rep == "vr_h0" else .2 + .01 * event,
                                             .8 + .02 * s + offset]]) for rep in reps}
                clean[condition] = (diagram(0), signal)
            for condition in ("C0", "C1", "C2", "C3"):
                base = "C0" if condition in ("C0", "C2") else "C1"
                cells = [("", None)] if condition in ("C0", "C1") else [(n, snr) for n in ("white", "pink", "brown") for snr in (-10, -5, 0, 5, 10, 15)]
                for noise, snr in cells:
                    pre, signal = clean[base]
                    post = {rep: pairs + np.array([[0, .025], [0, .015]]) for rep, pairs in pre.items()}
                    key = dict(event_id=event_id, subject_id=subject_id, condition=condition,
                               noise_type=noise, snr_target=snr)
                    windows.append({**key, "pre": pre, "post": post,
                                    "amplitude": {"pre": (.1, .8), "post": (.12, .83)}})
                    rows.append({**key, "sqi__signal": signal + rng.normal(0, .04),
                                 "amplitude__range": signal * .3, "qrs__width": signal + rng.normal(0, .1),
                                 "qrs__nullable": np.nan if (s + event) % 3 == 0 else signal,
                                 "qrs__missing": int((s + event) % 3 == 0)})
    classical = pd.DataFrame(rows)
    if order == "shuffled":
        permutation = rng.permutation(len(windows))
        windows = [windows[i] for i in permutation]
        classical = classical.iloc[permutation].reset_index(drop=True)
    elif order != "sorted":
        raise ValueError(order)
    return windows, classical


@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_yaml(PROJECT_ROOT / "configs/followup_error_monitor_v0_1.yaml")
    cfg["base_model"]["c_grid"] = [0.1, 1]
    cfg["monitor_training"]["c_grid"] = [0.1, 1]
    cfg["persistence_image"]["resolution"] = 3
    return cfg


@pytest.fixture(scope="module")
def fold(small_cfg):
    from topovital_followup.error_monitor.pilot import compute_fold
    windows, classical = synthetic()
    result = compute_fold(windows, classical, "S04", small_cfg, threshold_target=.9,
                          hashes=("feature", "config", "code"), log=lambda _: None)
    return windows, classical, result
