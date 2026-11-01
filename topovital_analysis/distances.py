"""Exact diagram distances using the L-infinity ground metric."""
from __future__ import annotations

import gudhi
import numpy as np
from scipy.optimize import linear_sum_assignment


def finite_diagram(values, *, vr_h0: bool = False, vr_h1: bool = False) -> np.ndarray:
    pairs = np.asarray(values, dtype=float).reshape(-1, 2)
    if np.isnan(pairs).any() or not np.isfinite(pairs[:, 0]).all():
        raise ValueError("invalid_diagram_coordinate")
    finite = np.isfinite(pairs[:, 1])
    dropped = int((~finite).sum())
    if vr_h0 and dropped != 1:
        raise ValueError("vr_h0_requires_one_essential_bar")
    if (vr_h1 or not vr_h0) and dropped:
        raise ValueError("unexpected_essential_bar")
    pairs = pairs[finite]
    if (pairs[:, 0] > pairs[:, 1]).any():
        raise ValueError("birth_after_death")
    return pairs


def diagonal_cost(points: np.ndarray) -> np.ndarray:
    return (points[:, 1] - points[:, 0]) / 2


def wasserstein_exact(x, y, order: int) -> float:
    """Optimal matching with a separate diagonal copy for each input point."""
    if order not in (1, 2):
        raise ValueError("unsupported_order")
    x = np.asarray(x, dtype=float).reshape(-1, 2)
    y = np.asarray(y, dtype=float).reshape(-1, 2)
    n, m = len(x), len(y)
    if not n and not m:
        return 0.0
    if not n or not m:
        return float(np.sum(diagonal_cost(x if n else y) ** order) ** (1 / order))
    cost = np.full((n + m, n + m), np.inf)
    cost[:n, :m] = np.max(np.abs(x[:, None, :] - y[None, :, :]), axis=2) ** order
    cost[np.arange(n), m + np.arange(n)] = diagonal_cost(x) ** order
    cost[n + np.arange(m), np.arange(m)] = diagonal_cost(y) ** order
    cost[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    return float(np.sum(cost[rows, cols]) ** (1 / order))


def bottleneck(x, y) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 2)
    y = np.asarray(y, dtype=float).reshape(-1, 2)
    if not len(x) and not len(y):
        return 0.0
    if not len(x) or not len(y):
        return float(np.max(diagonal_cost(x if len(x) else y)))
    return float(gudhi.bottleneck_distance(x, y, e=0))


def diagram_distances(x, y, metrics: dict) -> dict[str, float]:
    results = {}
    for name, settings in metrics.items():
        if settings.get("ground_metric") != "linf":
            raise ValueError(f"unsupported_ground_metric:{name}")
        implementation = settings.get("implementation")
        if implementation == "gudhi.bottleneck_distance(e=0)":
            results[name] = bottleneck(x, y)
        elif implementation == "exact_assignment_with_diagonal" and name in ("wasserstein_1", "wasserstein_2"):
            results[name] = wasserstein_exact(x, y, int(name[-1]))
        else:
            raise ValueError(f"unsupported_metric:{name}")
    return results
