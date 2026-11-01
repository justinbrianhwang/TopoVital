"""Descriptive factorial metrics and a hash-checked, paired Level 2 comparison."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from topovital.cohort import BlockedError
from .level2_metrics import unbiased_mmd2
from .stats import signed_auroc


def _cells(frame):
    return sorted({(r.noise_type, r.snr_target) for r in frame.itertuples()
                   if r.condition in ("C2", "C3")})


def _part(frame, condition, cell=None):
    part = frame.loc[frame.condition.eq(condition)]
    if cell is not None and condition in ("C2", "C3"):
        part = part.loc[part.noise_type.eq(cell[0]) & part.snr_target.eq(cell[1])]
    return part


def stratified_auroc(frame, cells=None):
    cells = _cells(frame) if cells is None else cells
    first = signed_auroc(_part(frame, "C1").score, _part(frame, "C0").score)
    other = [signed_auroc(_part(frame, "C3", c).score, _part(frame, "C2", c).score) for c in cells]
    return float(.5 * first + .5 * np.mean(other)) if len(other) and np.isfinite([first, *other]).all() else np.nan


class _MetricArrays:
    def __init__(self, frame, cells):
        self.cells = tuple(cells)
        self.score = frame.score.to_numpy(dtype=float)
        self.subject = pd.factorize(frame.subject_id, sort=False)[0]
        self.condition = pd.Categorical(frame.condition, categories=["C0", "C1", "C2", "C3"]).codes
        cell_codes = {cell: i for i, cell in enumerate(cells)}
        self.cell = np.fromiter((cell_codes.get((n, s), -1) for n, s in
                                 zip(frame.noise_type, frame.snr_target)), dtype=int, count=len(frame))
        self.event = frame.event_id.to_numpy()
        self.threshold_primary = frame.threshold_primary.to_numpy(dtype=float) if "threshold_primary" in frame else None
        self.threshold_c1 = frame.threshold_c1.to_numpy(dtype=float) if "threshold_c1" in frame else None

    def take(self, indices, subject):
        out = object.__new__(_MetricArrays)
        out.cells = self.cells
        for name in ("score", "condition", "cell", "event", "threshold_primary", "threshold_c1"):
            values = getattr(self, name)
            setattr(out, name, None if values is None else values[indices])
        out.subject = subject
        return out


def _as_arrays(frame, cells):
    return frame if isinstance(frame, _MetricArrays) else _MetricArrays(frame, cells)


def _weighted_auc(frame, cells):
    data = _as_arrays(frame, cells)
    # Equal condition and cell mass within each subject; row mass is equal inside a cell.
    n_subjects = int(data.subject.max()) + 1 if len(data.subject) else 0
    n_cells = len(cells)
    if not n_subjects or not n_cells:
        return np.nan
    group = np.where(data.condition < 2, data.condition,
                     2 + (data.condition - 2) * n_cells + data.cell)
    valid = (data.condition >= 0) & ((data.condition < 2) | (data.cell >= 0))
    width = 2 + 2 * n_cells
    counts = np.bincount(data.subject[valid] * width + group[valid], minlength=n_subjects * width)
    if np.any(counts == 0) or not valid.all():
        return np.nan
    weights = 1 / (n_subjects * 4 * np.where(data.condition < 2, 1, n_cells) *
                   counts[data.subject * width + group])
    pos = np.isin(data.condition, (1, 3))
    score = data.score
    if not np.isfinite(score).all():
        return np.nan
    order = np.argsort(score, kind="mergesort")
    score, pos, weights = score[order], pos[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    positive_weight = np.add.reduceat(np.where(pos, weights, 0), starts)
    negative_weight = np.add.reduceat(np.where(pos, 0, weights), starts)
    negative_before = np.cumsum(negative_weight) - negative_weight
    numerator = np.sum(positive_weight * (negative_before + .5 * negative_weight))
    return float(numerator / (weights[pos].sum() * weights[~pos].sum()))


def _rate(frame, condition, threshold, cell=None):
    data = _as_arrays(frame, [cell] if cell is not None else [])
    code = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}[condition]
    mask = data.condition == code
    if cell is not None and code >= 2:
        mask &= data.cell == data.cells.index(cell)
    scores = data.score[mask]
    thresholds = getattr(data, threshold)
    thresholds = thresholds[mask]
    if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(thresholds).all():
        return np.nan
    return float(np.mean(scores >= thresholds))


def _paired(frame, changed, original, cells):
    data = _as_arrays(frame, cells)
    changed_code = {"C2": 2, "C3": 3}[changed]
    original_code = {"C0": 0, "C1": 1}[original]
    base_indices = np.flatnonzero(data.condition == original_code)
    base = {(data.subject[i], data.event[i]): data.score[i] for i in base_indices}
    values = []
    for cell_code in range(len(cells)):
        altered = np.flatnonzero((data.condition == changed_code) & (data.cell == cell_code))
        if not len(altered) or any((data.subject[i], data.event[i]) not in base for i in altered):
            return np.nan
        values.append(float(np.mean([data.score[i] - base[(data.subject[i], data.event[i])]
                                     for i in altered])))
    return float(np.mean(values)) if values else np.nan


def metric_values(frame, cells, rule="primary"):
    if not cells:
        return {}
    data = _as_arrays(frame, cells)
    scores = data.score
    conditions = data.condition
    cell_codes = data.cell
    clean = [scores[conditions == code] for code in (0, 1)]
    c1 = signed_auroc(clean[1], clean[0])
    injected = [[scores[(conditions == code) & (cell_codes == i)] for i in range(len(cells))]
                for code in (2, 3)]
    c3 = np.array([signed_auroc(a, b) for a, b in zip(injected[1], injected[0])])
    c2_art = np.array([signed_auroc(a, clean[0]) for a in injected[0]])
    c3_art = np.array([signed_auroc(a, clean[1]) for a in injected[1]])
    threshold = "threshold_primary" if rule == "primary" else "threshold_c1"
    rates = {k: _rate(data, k, threshold) if k in ("C0", "C1") else
             float(np.mean([_rate(data, k, threshold, c) for c in cells]))
             for k in ("C0", "C1", "C2", "C3")}
    stratified = float(.5 * c1 + .5 * np.mean(c3)) if np.isfinite(np.r_[c1, c3]).all() else np.nan
    result = {"stratified_auroc": stratified,
              "auroc_c1_vs_c0": c1, "auroc_c3_vs_c2": float(np.mean(c3)),
              "weighted_pooled_auroc": _weighted_auc(data, cells),
              "auroc_c2_vs_c0": float(np.mean(c2_art)), "auroc_c3_vs_c1": float(np.mean(c3_art)),
              "abs_auroc_c2_vs_c0_minus_half": float(np.mean(np.abs(c2_art - .5))),
              "abs_auroc_c3_vs_c1_minus_half": float(np.mean(np.abs(c3_art - .5))),
              "paired_change_normal": _paired(data, "C2", "C0", cells),
              "paired_change_pvc": _paired(data, "C3", "C1", cells),
              "c1_sensitivity": rates["C1"], "c3_sensitivity": rates["C3"],
              "c0_fpr": rates["C0"], "c2_fpr": rates["C2"]}
    result["weighted_balanced_accuracy"] = .25 * (rates["C1"] + rates["C3"] + 2 - rates["C0"] - rates["C2"])
    result["rate_change_fpr"] = rates["C2"] - rates["C0"]
    result["rate_change_sens"] = rates["C3"] - rates["C1"]
    return result


def _scope(frame, cells):
    clean = frame.loc[frame.condition.isin(("C0", "C1"))]
    injected = frame.loc[frame.condition.isin(("C2", "C3")) &
                         pd.Series([(n, s) in cells for n, s in zip(frame.noise_type, frame.snr_target)], index=frame.index)]
    return pd.concat((clean, injected), ignore_index=True)


def metric_tables(predictions, cfg):
    cells = [(n, s) for n in cfg["scope"]["regime1_cells"]["noise_types"]
             for s in cfg["scope"]["regime1_cells"]["snr_db"]]
    scopes = [("overall", "all", cells), ("protocol_9_cell", "12_6_0", [c for c in cells if c[1] in (12, 6, 0)])]
    scopes += [(n, s, [(n, s)]) for n, s in cells]
    resamples = int(cfg["reporting"]["bootstrap"]["resamples"])
    seed = int(cfg["reporting"]["bootstrap"]["seed"])
    window, subject, detail, ci = [], [], [], []
    for model, mf in predictions.groupby("model", sort=True):
        for noise, snr, selected in scopes:
            sf = _scope(mf, selected)
            groups = {s: g for s, g in sf.groupby("subject_id", sort=True)}
            for rule in ("primary", "c1"):
                per = {s: metric_values(g, selected, rule) for s, g in groups.items()}
                pooled = metric_values(sf, selected, rule)
                for metric, value in pooled.items():
                    key = {"model": model, "noise_type": noise, "snr_target": snr, "rule": rule, "metric": metric}
                    values = np.array([v[metric] for v in per.values()], dtype=float)
                    valid = values[np.isfinite(values)]
                    subject_value = float(valid.mean()) if len(valid) else np.nan
                    for name, metrics in per.items():
                        v = metrics[metric]
                        detail.append({**key, "subject_id": name, "value": v, "status": "ok" if np.isfinite(v) else "not_evaluable", "reason": "" if np.isfinite(v) else "missing_class_or_score"})
                    for out, v, count_key, count in ((window, value, "n_windows", len(sf)), (subject, subject_value, "n_subjects", len(valid))):
                        out.append({**key, "value": v, count_key: count, "status": "ok" if np.isfinite(v) else "not_evaluable", "reason": "" if np.isfinite(v) else "missing_class_or_score"})
                    rng = np.random.default_rng(seed)
                    draws = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) if len(values) else np.nan for _ in range(resamples)])
                    good = draws[np.isfinite(draws)]
                    ok = len(draws) > 0 and len(good) / len(draws) >= .95
                    ci.append({**key, "ci_low": float(np.quantile(good, .025)) if ok else np.nan,
                               "ci_high": float(np.quantile(good, .975)) if ok else np.nan,
                               "undefined_resamples": len(draws) - len(good), "status": "ok" if ok else "not_evaluable",
                               "reason": "" if ok else "bootstrap_undefined_over_5_percent"})
    return tuple(map(pd.DataFrame, (window, subject, detail, ci)))


def _verified_level2(root: Path, source: Path):
    manifest = source.parent.parent / "hashes" / "level2_outputs.sha256"
    if not manifest.is_file() or not source.is_file():
        raise BlockedError("Level 2 comparison source or hash manifest missing")
    try:
        relative = source.relative_to(manifest.parent.parent).as_posix()
    except ValueError as exc:
        raise BlockedError("Level 2 comparison source is outside the manifest directory") from exc
    matches = [line.split(maxsplit=1) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [parts[0] for parts in matches if len(parts) == 2 and parts[1].lstrip("*") == relative]
    if len(expected) != 1 or hashlib.sha256(source.read_bytes()).hexdigest().lower() != expected[0].lower():
        raise BlockedError("Level 2 output SHA256 mismatch")
    return pd.read_parquet(source)


def comparison_with_level2(factorial, root, cfg_f):
    source = root / cfg_f["reporting"]["comparison_with_level2"]["source"]
    old = _verified_level2(root, source)
    keys = ["model", "event_id", "subject_id", "condition", "noise_type", "snr_target"]
    old = old.loc[old.subject_id.isin(factorial.subject_id.unique())]
    if factorial.duplicated(keys).any() or old.duplicated(keys).any() or len(old) != len(factorial):
        raise BlockedError("Level 2 comparison keys are not one-to-one")
    joined = factorial.merge(old[keys + ["score", "threshold"]], on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_level2"))
    if not joined._merge.eq("both").all():
        raise BlockedError("Level 2 comparison rows do not match")
    cells = _cells(factorial)
    scopes = [("overall", "all", cells), ("protocol_9_cell", "12_6_0", [c for c in cells if c[1] in (12, 6, 0)])]
    scopes += [(n, s, [(n, s)]) for n, s in cells]
    boot = cfg_f["reporting"]["bootstrap"]
    sentence = cfg_f["reporting"]["comparison_with_level2"]["rule"]
    rows = []
    for model, mf in joined.groupby("model", sort=True):
        for noise, snr, selected in scopes:
            sf = _scope(mf, selected)
            subjects = sorted(sf.subject_id.unique())
            old_frame = sf.assign(score=sf.score_level2, threshold_c1=sf.threshold)
            subject_indices = [np.flatnonzero(sf.subject_id.to_numpy() == s) for s in subjects]
            new_arrays = _MetricArrays(sf, selected)
            old_arrays = _MetricArrays(old_frame, selected)
            for rule in ("c1", "primary"):
                for aggregation in ("window_level", "subject_equal_weight"):
                    new_values = metric_values(sf, selected, rule)
                    old_values = metric_values(old_frame, selected, "c1") if rule == "c1" else {}
                    new_groups = {s: g for s, g in sf.groupby("subject_id")}
                    old_groups = {s: g for s, g in old_frame.groupby("subject_id")} if rule == "c1" else {}
                    new_per = {s: metric_values(g, selected, rule) for s, g in new_groups.items()}
                    old_per = {s: metric_values(g, selected, "c1") for s, g in old_groups.items()} if rule == "c1" else {}
                    draws = []
                    if rule == "c1":
                        rng = np.random.default_rng(int(boot["seed"]))
                        cache = {}
                        for _ in range(int(boot["resamples"])):
                            indices = rng.integers(0, len(subjects), len(subjects))
                            key = tuple(np.bincount(indices, minlength=len(subjects)))
                            if key not in cache:
                                if aggregation == "subject_equal_weight":
                                    cache[key] = {metric: float(np.mean([new_per[subjects[i]][metric] - old_per[subjects[i]][metric] for i in indices])) for metric in new_values}
                                else:
                                    blocks = [subject_indices[i] for i, count in enumerate(key) for _ in range(count)]
                                    draw_rows = np.concatenate(blocks)
                                    draw_subject = np.repeat(np.arange(len(blocks)), [len(block) for block in blocks])
                                    nv = metric_values(new_arrays.take(draw_rows, draw_subject), selected, rule)
                                    ov = metric_values(old_arrays.take(draw_rows, draw_subject), selected, "c1")
                                    cache[key] = {metric: nv[metric] - ov[metric] for metric in new_values}
                            draws.append(cache[key])
                    for metric, fv in new_values.items():
                        lv = old_values.get(metric, np.nan) if rule == "c1" else np.nan
                        if aggregation == "subject_equal_weight":
                            fv = float(np.mean([new_per[s][metric] for s in subjects]))
                            if rule == "c1":
                                lv = float(np.mean([old_per[s][metric] for s in subjects]))
                        good = np.asarray([draw[metric] for draw in draws], dtype=float)
                        good = good[np.isfinite(good)]
                        ok = len(draws) and len(good) / len(draws) >= .95
                        rows.append({"model": model, "noise_type": noise, "snr_target": snr, "rule": rule,
                                     "aggregation": aggregation, "metric": metric, "level2_value": lv,
                                     "factorial_value": fv, "difference": fv - lv,
                                     "ci_low": float(np.quantile(good, .025)) if ok else np.nan,
                                     "ci_high": float(np.quantile(good, .975)) if ok else np.nan,
                                     "robustness_rule": sentence})
    return pd.DataFrame(rows)
