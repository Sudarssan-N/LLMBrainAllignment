"""Noise-ceiling estimation for Pereira2018.

Pereira voxels are subject-specific and the averaged assembly has no within-subject
stimulus repeats, so a standard within-subject split-half ceiling is undefined. We use a
**cross-subject (brain-to-brain) ceiling**: for each held-out subject, predict their voxels
from the *other* subjects' responses to the same sentences (PCA-reduced, cross-validated
ridge). The resulting per-voxel held-out correlation is a principled upper bound on what any
external model can achieve, since it reflects only the variance shared with independent
neural data. Normalized predictivity = model_r / ceiling, aggregated over voxels whose
ceiling clears a reliability threshold.

For datasets where every subject shares the same voxels and provides a repeat (e.g. the
synthetic fixture), use `split_half_noise_ceiling` in `ridge.py` instead.
"""

from __future__ import annotations

import numpy as np

from .ridge import ridge_encode


def cross_subject_noise_ceiling(
    Y: np.ndarray,
    voxel_subject: np.ndarray,
    alphas: list[float],
    pca_components: int = 100,
    n_folds: int = 5,
    standardize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Per-voxel cross-subject ceiling (held-out subject predicted from the rest).

    Args:
        Y: (n_samples, n_voxels) responses.
        voxel_subject: (n_voxels,) subject id per voxel column.
    Returns:
        (n_voxels,) ceiling correlations; NaN where it can't be estimated.
    """
    voxel_subject = np.asarray(voxel_subject)
    ceiling = np.full(Y.shape[1], np.nan, dtype=np.float64)
    subjects = np.unique(voxel_subject)
    if subjects.size < 2:
        return ceiling

    for t in subjects:
        tgt = voxel_subject == t
        donor = ~tgt
        if donor.sum() == 0 or tgt.sum() == 0:
            continue
        res = ridge_encode(
            X=Y[:, donor],
            Y=Y[:, tgt],
            alphas=alphas,
            n_folds=n_folds,
            standardize=standardize,
            pca_components=min(pca_components, int(donor.sum())),
            seed=seed,
        )
        ceiling[tgt] = res.voxel_r
    return ceiling


def normalized_predictivity(
    voxel_r: np.ndarray,
    ceiling: np.ndarray,
    min_ceiling: float = 0.1,
) -> dict:
    """Aggregate model voxel_r normalized by ceiling over reliably-estimated voxels.

    Only voxels with ceiling >= min_ceiling are included (dividing by a near-zero,
    unreliable ceiling explodes the ratio). Returns mean raw r, mean normalized r, and the
    voxel count used.
    """
    voxel_r = np.asarray(voxel_r, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    valid = np.isfinite(voxel_r) & np.isfinite(ceiling) & (ceiling >= min_ceiling)
    if valid.sum() == 0:
        return {"mean_r": float(np.nanmean(voxel_r)), "normalized_r": None,
                "n_voxels_used": 0, "mean_ceiling": None}
    norm = voxel_r[valid] / ceiling[valid]
    return {
        "mean_r": float(np.nanmean(voxel_r[valid])),
        "normalized_r": float(np.mean(norm)),
        "n_voxels_used": int(valid.sum()),
        "mean_ceiling": float(np.mean(ceiling[valid])),
    }
