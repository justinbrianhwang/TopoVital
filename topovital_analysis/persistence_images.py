"""Fold-local persistence-image geometry and feature construction.

Feature order: representation insertion order from config; for each representation,
row-major flattened pre, post, then post-minus-pre. VR H0 is one dimensional.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def estimate_regions(windows, indices, representations, quantiles, *, excluded_subject=None):
    """Pool C0/C1 pre/post pairs from training subjects only."""
    selected = [windows[i] for i in indices]
    assert selected and all(w["condition"] in ("C0", "C1") for w in selected)
    if excluded_subject is not None:
        assert all(w["subject_id"] != excluded_subject for w in selected), "held-out subject in PI region"
    result = {}
    for rep in representations:
        pairs = [w[half][rep] for w in selected for half in ("pre", "post")]
        pairs = np.concatenate(pairs, axis=0)
        if not len(pairs):
            raise ValueError(f"no_finite_pairs_{rep}")
        persistence = pairs[:, 1] - pairs[:, 0]
        p0, p1 = quantiles["persistence"]
        if float(p0) != 0.0:
            raise ValueError("persistence_region_must_start_at_zero")
        low_p, high_p = 0.0, float(np.quantile(persistence, p1))
        if rep == "vr_h0":
            low_b = high_b = None
        else:
            b0, b1 = quantiles["birth"]
            low_b, high_b = float(np.quantile(pairs[:, 0], b0)), float(np.quantile(pairs[:, 0], b1))
            if high_b <= low_b:
                high_b = low_b + np.finfo(float).eps * max(1., abs(low_b))
        if high_p <= low_p:
            raise ValueError(f"no_positive_persistence_{rep}")
        result[rep] = (low_b, high_b, low_p, high_p)
    return result


def out_of_range(pairs, region):
    pairs = np.asarray(pairs, dtype=float).reshape(-1, 2)
    b0, b1, p0, p1 = region
    persistence = pairs[:, 1] - pairs[:, 0]
    outside = (persistence < p0) | (persistence > p1)
    if b0 is not None:
        outside |= (pairs[:, 0] < b0) | (pairs[:, 0] > b1)
    return int(outside.sum()), float(persistence[outside].sum())


def persistence_image(pairs, region, resolution, sigma_pixels):
    pairs = np.asarray(pairs, dtype=float).reshape(-1, 2)
    b0, b1, p0, p1 = region
    shape = (resolution,) if b0 is None else (resolution, resolution)
    if not len(pairs):
        return np.zeros(shape, dtype=float)
    persistence = pairs[:, 1] - pairs[:, 0]
    p_edges = np.linspace(p0, p1, resolution + 1)
    sigma_p = sigma_pixels * (p1 - p0) / resolution
    p_mass = np.diff(ndtr((p_edges[None, :] - persistence[:, None]) / sigma_p), axis=1)
    weights = persistence / p1
    if b0 is None:
        return weights @ p_mass
    b_edges = np.linspace(b0, b1, resolution + 1)
    sigma_b = sigma_pixels * (b1 - b0) / resolution
    b_mass = np.diff(ndtr((b_edges[None, :] - pairs[:, 0, None]) / sigma_b), axis=1)
    return (b_mass * weights[:, None]).T @ p_mass


def amplitude_features(window):
    parts = []
    for half in ("pre", "post"):
        low, high = window["amplitude"][half]
        parts.append(np.array([low, high, high - low], dtype=float))
    return np.concatenate((parts[0], parts[1], parts[1] - parts[0]))


def pi_features(window, regions, resolution, sigma_pixels, representations):
    parts = []
    for rep in representations:
        pre = persistence_image(window["pre"][rep], regions[rep], resolution, sigma_pixels).ravel()
        post = persistence_image(window["post"][rep], regions[rep], resolution, sigma_pixels).ravel()
        parts.extend((pre, post, post - pre))
    return np.concatenate(parts)
