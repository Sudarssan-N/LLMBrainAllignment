"""Bootstrap confidence intervals for per-voxel encoding/variance-partitioning metrics.

Two kinds of CI are provided:
  - `bootstrap_ci_mean`: CI on the mean of a per-voxel quantity by resampling voxels.
  - `paired_diff_ci`: CI on the mean paired difference between two models evaluated on the
    *same* voxels (e.g. base vs instruction-tuned), the right test for E004.

Resampling voxels gives error bars on the across-voxel mean. The stronger stimulus-level
bootstrap (resampling sentences and refitting) is left for the camera-ready; these cheap
voxel-level CIs are sufficient to gate which effects are worth reporting.
"""

from __future__ import annotations

import numpy as np


def _clean(values: np.ndarray, clip: tuple[float, float] | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if clip is not None:
        values = np.clip(values, clip[0], clip[1])
    return values


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
    clip: tuple[float, float] | None = None,
) -> dict:
    """Bootstrap CI for the mean of a per-voxel quantity (resampling voxels)."""
    v = _clean(values, clip)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
    clip: tuple[float, float] | None = None,
) -> dict:
    """Bootstrap CI for the mean paired difference (a - b) over shared voxels.

    Both arrays must be per-voxel quantities aligned to the same voxels (e.g. base vs
    instruct unique_hidden). Voxels where either value is non-finite are dropped. If the CI
    excludes 0 the difference is 'significant' at the given level.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    diff = a[mask] - b[mask]
    if clip is not None:
        diff = np.clip(diff, clip[0], clip[1])
    if diff.size == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "significant": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(diff.size),
        "significant": bool(lo > 0 or hi < 0),
    }
