"""Surprisal-matched stimulus subsets.

Validation analysis complementing variance partitioning: select a subset of stimuli whose
surprisal distribution is balanced across a structural manipulation (e.g. high vs low
syntactic complexity), so that any residual difference in brain predictivity cannot be
attributed to surprisal. Two strategies are provided:

  - `match_by_binning`: bin on surprisal, equalize counts per structural group per bin.
  - `match_by_nearest`: greedy nearest-surprisal pairing between two groups.
"""

from __future__ import annotations

import numpy as np


def match_by_binning(
    surprisal: np.ndarray,
    group: np.ndarray,
    n_bins: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Return indices forming a surprisal-balanced subset across `group` labels.

    Within each surprisal bin, downsample every group to the minimum group count so the
    surprisal distribution is matched across groups.
    """
    surprisal = np.asarray(surprisal).ravel()
    group = np.asarray(group).ravel()
    rng = np.random.default_rng(seed)

    edges = np.quantile(surprisal, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(surprisal, edges[1:-1]), 0, n_bins - 1)

    keep: list[int] = []
    groups = np.unique(group)
    for b in range(n_bins):
        in_bin = np.flatnonzero(bin_idx == b)
        if in_bin.size == 0:
            continue
        per_group = {g: in_bin[group[in_bin] == g] for g in groups}
        counts = [v.size for v in per_group.values()]
        if min(counts) == 0:
            continue
        k = min(counts)
        for g in groups:
            chosen = rng.choice(per_group[g], size=k, replace=False)
            keep.extend(chosen.tolist())
    return np.sort(np.asarray(keep, dtype=int))


def match_by_nearest(
    surprisal: np.ndarray,
    group: np.ndarray,
    max_delta: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Greedy nearest-surprisal pairing between exactly two groups.

    Pairs each item of the smaller group with the closest-surprisal unused item of the
    other group (optionally within `max_delta`). Returns the paired indices.
    """
    surprisal = np.asarray(surprisal).ravel()
    group = np.asarray(group).ravel()
    groups = np.unique(group)
    if groups.size != 2:
        raise ValueError("match_by_nearest requires exactly two groups")

    a = np.flatnonzero(group == groups[0])
    b = np.flatnonzero(group == groups[1])
    if a.size > b.size:
        a, b = b, a
    used_b = np.zeros(b.size, dtype=bool)
    keep: list[int] = []
    for i in a:
        d = np.abs(surprisal[b] - surprisal[i])
        d[used_b] = np.inf
        j = int(np.argmin(d))
        if max_delta is not None and d[j] > max_delta:
            continue
        used_b[j] = True
        keep.extend([int(i), int(b[j])])
    return np.sort(np.asarray(keep, dtype=int))


def matched_balance_report(
    surprisal: np.ndarray, group: np.ndarray, idx: np.ndarray
) -> dict:
    """Summarize surprisal balance across groups before/after matching (sanity check)."""
    surprisal = np.asarray(surprisal).ravel()
    group = np.asarray(group).ravel()

    def per_group_mean(mask_idx):
        return {
            str(g): float(np.mean(surprisal[mask_idx][group[mask_idx] == g]))
            for g in np.unique(group[mask_idx])
        }

    full = np.arange(surprisal.size)
    return {
        "n_full": int(full.size),
        "n_matched": int(idx.size),
        "surprisal_mean_full": per_group_mean(full),
        "surprisal_mean_matched": per_group_mean(idx),
    }
