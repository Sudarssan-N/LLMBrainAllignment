"""Variance partitioning (commonality analysis) — the project's primary analysis.

Given hidden-state features H and surprisal features S, we fit three cross-validated ridge
encoding models and decompose explained variance:

    unique(H) = R²(H ∪ S) - R²(S)        # variance hidden states add beyond surprisal
    unique(S) = R²(H ∪ S) - R²(H)        # variance surprisal adds beyond hidden states
    shared    = R²(H) + R²(S) - R²(H ∪ S)
    total     = R²(H ∪ S)

This directly answers the core question — how much alignment is *uniquely* explained by
hidden representations beyond surprisal — and is robust to the H/S collinearity that makes
surprisal-matched subsets alone underpowered.

Cross-validated R² can be negative for poorly-predicted voxels; we clip the *reported*
unique/shared aggregates at 0 for interpretability but also return raw values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ridge import ridge_encode


@dataclass
class VariancePartition:
    r2_hidden: np.ndarray            # per-voxel R²(H)
    r2_surprisal: np.ndarray         # per-voxel R²(S)
    r2_joint: np.ndarray             # per-voxel R²(H ∪ S)
    unique_hidden: np.ndarray
    unique_surprisal: np.ndarray
    shared: np.ndarray

    def summary(self, clip: bool = True, ci: bool = False, n_boot: int = 1000,
               seed: int = 0) -> dict:
        clip_range = (0.0, None) if clip else None

        def agg(a):
            a = np.clip(a, 0, None) if clip else a
            return float(np.nanmean(a))

        out = {
            "mean_r2_hidden": agg(self.r2_hidden),
            "mean_r2_surprisal": agg(self.r2_surprisal),
            "mean_r2_joint": agg(self.r2_joint),
            "mean_unique_hidden": agg(self.unique_hidden),
            "mean_unique_surprisal": agg(self.unique_surprisal),
            "mean_shared": agg(self.shared),
            "n_voxels": int(self.r2_hidden.shape[0]),
        }
        if ci:
            from .stats import bootstrap_ci_mean

            cr = (0.0, np.inf) if clip else None
            out["ci_unique_hidden"] = bootstrap_ci_mean(
                self.unique_hidden, n_boot=n_boot, seed=seed, clip=cr)
            out["ci_unique_surprisal"] = bootstrap_ci_mean(
                self.unique_surprisal, n_boot=n_boot, seed=seed, clip=cr)
        return out


def variance_partitioning(
    hidden: np.ndarray,
    surprisal: np.ndarray,
    Y: np.ndarray,
    alphas: list[float],
    n_folds: int = 5,
    standardize: bool = True,
    pca_components: int | None = None,
    seed: int = 0,
) -> VariancePartition:
    """Decompose voxel variance into unique/shared contributions of hidden vs surprisal."""
    joint = np.concatenate([np.asarray(hidden), np.asarray(surprisal)], axis=1)

    r2_h = ridge_encode(hidden, Y, alphas, n_folds, standardize, pca_components, seed).voxel_r2
    r2_s = ridge_encode(surprisal, Y, alphas, n_folds, standardize, None, seed).voxel_r2
    r2_j = ridge_encode(joint, Y, alphas, n_folds, standardize, pca_components, seed).voxel_r2

    return VariancePartition(
        r2_hidden=r2_h,
        r2_surprisal=r2_s,
        r2_joint=r2_j,
        unique_hidden=r2_j - r2_s,
        unique_surprisal=r2_j - r2_h,
        shared=r2_h + r2_s - r2_j,
    )
