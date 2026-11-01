from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path

import gudhi.hera
import numpy as np
import pandas as pd
import pytest

from topovital.cohort import BlockedError
from topovital.util import load_yaml
from topovital_stage1.features import save_job
from topovital_analysis.distances import bottleneck, finite_diagram, wasserstein_exact
from topovital_analysis.level1 import _safe_job, _worker, run_level1, validate_events
from topovital_analysis.stats import (bootstrap_auroc, direction_free, loso_rates,
                                       separation, signed_auroc)


@pytest.mark.parametrize("order", [1, 2])
def test_tiny_distances_and_symmetry(order):
    empty = np.empty((0, 2))
    a = np.array([[0., 2.]])
    b = np.array([[0., 4.]])
    assert wasserstein_exact(empty, empty, order) == 0
    assert wasserstein_exact(a, empty, order) == 1
    assert wasserstein_exact(empty, a, order) == 1
    assert bottleneck(empty, empty) == 0
    assert bottleneck(empty, a) == bottleneck(a, empty) == 1
    # Direct match costs 2; matching both to diagonal costs 1^p + 2^p.
    expected = min(2., (1 + 2 ** order) ** (1 / order))
    assert wasserstein_exact(a, b, order) == pytest.approx(expected)
    assert wasserstein_exact(b, a, order) == pytest.approx(expected)
    assert bottleneck(a, b) == bottleneck(b, a) == 2


def brute_force(x, y, order):
    n, m = len(x), len(y)
    best = float("inf")
    for k in range(min(n, m) + 1):
        for xs in combinations(range(n), k):
            for ys in permutations(range(m), k):
                paired_x, paired_y = set(xs), set(ys)
                total = sum(max(abs(x[i] - y[j])) ** order for i, j in zip(xs, ys))
                total += sum(((x[i, 1] - x[i, 0]) / 2) ** order for i in range(n) if i not in paired_x)
                total += sum(((y[j, 1] - y[j, 0]) / 2) ** order for j in range(m) if j not in paired_y)
                best = min(best, total)
    return best ** (1 / order)


def test_exact_assignment_matches_brute_force_and_hera():
    rng = np.random.default_rng(42)
    for n in (3, 4):
        x = rng.uniform(-1, 1, (n, 2))
        y = rng.uniform(-1, 1, (n, 2))
        x[:, 1] = x[:, 0] + rng.uniform(.1, 2, n)
        y[:, 1] = y[:, 0] + rng.uniform(.1, 2, n)
        for order in (1, 2):
            exact = wasserstein_exact(x, y, order)
            assert exact == pytest.approx(brute_force(x, y, order), rel=1e-12)
            assert exact == pytest.approx(wasserstein_exact(y, x, order), rel=1e-12)
            hera = gudhi.hera.wasserstein_distance(x, y, order=order, internal_p=np.inf, delta=1e-6)
            assert exact == pytest.approx(hera, rel=1e-4)


def test_vr_essential_bar_validation():
    assert finite_diagram([[0, 1], [0, np.inf]], vr_h0=True).tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="one_essential"):
        finite_diagram([[0, 1]], vr_h0=True)
    with pytest.raises(ValueError, match="unexpected_essential"):
        finite_diagram([[0, np.inf]], vr_h1=True)
    with pytest.raises(ValueError, match="birth_after_death"):
        finite_diagram([[2, 1]])


def test_signed_auroc_direction_and_subject_missing_class():
    high = signed_auroc([3, 4], [1, 2])
    low = signed_auroc([1, 2], [3, 4])
    assert high == 1 and low == 0
    assert direction_free(high) == direction_free(low)
    frame = pd.DataFrame({"subject_id": ["A", "A", "B"],
                          "condition": ["C1", "C2", "C1"], "score": [2., 1., 3.]})
    result, detail = separation(frame, "C1", "C2", .9, 20, 7)
    assert result["status"] == "not_evaluable"
    assert detail.loc[detail.subject_id.eq("B"), "status"].item() == "not_evaluable"


def test_loso_threshold_excludes_held_out_and_bootstrap_seed():
    frame = pd.DataFrame({"subject_id": ["A", "A", "B", "B", "C", "C"],
                          "condition": ["C1", "C2"] * 3,
                          "score": [100., 0., 3., 0., 4., 0.]})
    folds = loso_rates(frame, "C1", "C2", .9)
    # Held-out A's score 100 must not set A's training threshold.
    assert folds.loc[folds.subject_id.eq("A"), "threshold"].item() == 3
    first = bootstrap_auroc(frame, "C1", "C2", 100, 17)
    assert first == bootstrap_auroc(frame, "C1", "C2", 100, 17)


def test_test_subject_and_external_job_rejected(tmp_path):
    frame = pd.DataFrame({"event_id": ["x"], "subject_id": ["S1"], "split": ["test"]})
    with pytest.raises(BlockedError, match="Test"):
        validate_events(frame)
    clean = tmp_path / "outputs/stage1_features/development/jobs"
    noise = tmp_path / "outputs/stage1_features/development/regime1_jobs"
    with pytest.raises(BlockedError, match="outside Development"):
        _safe_job(tmp_path / "outputs/test/x.npz", (clean, noise))


def test_worker_reads_only_development_jobs_and_indexes_sha(tmp_path):
    cfg = load_yaml(Path(__file__).parents[2] / "configs/stage1_analysis.yaml")
    clean = tmp_path / cfg["inputs"]["clean_jobs"]
    clean.mkdir(parents=True)
    diagrams = {"vr_h0": np.array([[0., 1.], [0., np.inf]]),
                "vr_h1": np.empty((0, 2)), "cubical_sub_finite": np.array([[0., 1.]]),
                "cubical_super_finite": np.array([[0., 1.]]),
                "cubical_sub_essential_births": np.array([0.]),
                "cubical_super_essential_births": np.array([0.])}
    pre, post = (clean / "x__C1__pre.npz", clean / "x__C1__post.npz")
    for path, half in ((pre, "pre"), (post, "post")):
        save_job(path, diagrams, {"event_id": "x", "condition": "C1", "half": half,
                                  "validation_status": "ok"})
    spec = {"event_id": "x", "subject_id": "S1", "condition": "C1",
            "noise_type": "", "snr_target": None, "pre_path": pre, "post_path": post}
    row, index, _, _ = _worker(spec, cfg, tmp_path, False, 1e-4)
    assert row["status"] == "ok"
    assert len(index) == 2
    assert all(entry["job_path"].startswith("outputs/stage1_features/development/jobs/") for entry in index)
    assert row["vr_h0__wasserstein_1"] == 0


def test_synthetic_level1_writes_complete_output_set(tmp_path):
    source = Path(__file__).parents[2] / "configs/stage1_analysis.yaml"
    cfg = load_yaml(source)
    cfg["scope"]["regime1_cells"] = {"noise_types": ["bw"], "snr_db": [24]}
    cfg["level1"]["separation"]["bootstrap"]["resamples"] = 4
    config_path = tmp_path / "configs/stage1_analysis.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(source.read_bytes())
    clean = tmp_path / cfg["inputs"]["clean_jobs"]
    regime = tmp_path / cfg["inputs"]["regime1_jobs"]
    clean.mkdir(parents=True)
    regime.mkdir(parents=True)
    diagrams = {"vr_h0": np.array([[0., 1.], [0., np.inf]]),
                "vr_h1": np.empty((0, 2)), "cubical_sub_finite": np.array([[0., 1.]]),
                "cubical_super_finite": np.array([[0., 1.]]),
                "cubical_sub_essential_births": np.array([0.]),
                "cubical_super_essential_births": np.array([0.])}
    for condition in ("C0", "C1"):
        for half in ("pre", "post"):
            save_job(clean / f"x__{condition}__{half}.npz", diagrams,
                     {"event_id": "x", "condition": condition, "half": half, "validation_status": "ok"})
    for condition in ("C2", "C3"):
        save_job(regime / f"x__{condition}__bw__24__post.npz", diagrams,
                 {"event_id": "x", "condition": condition, "half": "post", "validation_status": "ok"})
    events = pd.DataFrame({"event_id": ["x"], "subject_id": ["S1"], "split": ["development"]})
    output = tmp_path / "analysis"
    summary = run_level1(events, cfg, tmp_path, output, 1, pilot=True)
    assert summary["windows"] == 4
    assert summary["failures"] == 0
    assert len(pd.read_csv(output / "diagram_index.csv")) == 6
    assert len(pd.read_parquet(output / "ph_distances.parquet")) == 4
    assert pd.read_csv(output / "c1_c2_separation.csv").status.eq("not_evaluable").all()
    assert (output / "hera_crosscheck.csv").is_file()
    assert (output / "run_summary.json").is_file()
