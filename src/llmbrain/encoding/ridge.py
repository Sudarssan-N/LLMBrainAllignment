"""Cross-validated ridge encoding model: features -> voxel responses.

Reproduces the Brain-Score linear-predictivity recipe: standardize features, optional PCA,
ridge regression with alpha selected by nested/grid CV, evaluate held-out per-voxel Pearson
r, then normalize by a noise ceiling. Returns both per-voxel scores and cross-validated R²
(used by variance partitioning).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def pearson_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-column (per-voxel) Pearson correlation. Zero-variance columns -> NaN."""
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    den = np.where(den == 0, np.nan, den)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


@dataclass
class EncodingResult:
    n_voxels: int
    n_folds: int
    voxel_r: np.ndarray              # mean held-out Pearson r per voxel
    voxel_r2: np.ndarray             # mean held-out R² per voxel (for variance partitioning)
    mean_r: float
    normalized_r: float | None = None
    noise_ceiling: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    ss_tot[ss_tot == 0] = np.nan
    return 1.0 - ss_res / ss_tot


def _fit_alpha(X_tr, Y_tr, X_va, Y_va, alphas):
    """Pick a single shared alpha by validation R² (averaged over voxels)."""
    best_alpha, best_score = alphas[0], -np.inf
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_tr, Y_tr)
        score = np.nanmean(_r2_per_column(Y_va, model.predict(X_va)))
        if score > best_score:
            best_score, best_alpha = score, a
    return best_alpha


def ridge_encode(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: list[float],
    n_folds: int = 5,
    standardize: bool = True,
    pca_components: int | None = None,
    seed: int = 0,
) -> EncodingResult:
    """Fit a cross-validated ridge encoding model.

    Args:
        X: (n_samples, n_features) stimulus features (activations / surprisal).
        Y: (n_samples, n_voxels) brain responses.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    r_folds, r2_folds = [], []
    for tr, te in kf.split(X):
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]

        if standardize:
            xs = StandardScaler().fit(X_tr)
            X_tr, X_te = xs.transform(X_tr), xs.transform(X_te)

        if pca_components:
            k = min(pca_components, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=k, random_state=seed).fit(X_tr)
            X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)

        # inner split of the training fold to choose alpha
        cut = max(1, int(0.8 * X_tr.shape[0]))
        alpha = _fit_alpha(X_tr[:cut], Y_tr[:cut], X_tr[cut:], Y_tr[cut:], alphas)

        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
        r_folds.append(pearson_per_column(Y_te, pred))
        r2_folds.append(_r2_per_column(Y_te, pred))

    # Some voxels are NaN in every fold (constant in a test split); average over folds
    # ignoring NaN, then over voxels ignoring any that are NaN across all folds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        voxel_r = np.nanmean(np.stack(r_folds), axis=0)
        voxel_r2 = np.nanmean(np.stack(r2_folds), axis=0)
        mean_r = float(np.nanmean(voxel_r)) if np.isfinite(voxel_r).any() else float("nan")
    return EncodingResult(
        n_voxels=Y.shape[1],
        n_folds=n_folds,
        voxel_r=voxel_r,
        voxel_r2=voxel_r2,
        mean_r=mean_r,
    )


def split_half_noise_ceiling(
    responses_by_subject: list[np.ndarray],
    n_splits: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Estimate a per-voxel noise ceiling via split-half reliability across subjects/repeats.

    Args:
        responses_by_subject: list of (n_samples, n_voxels) arrays, one per subject/repeat.
    Returns:
        (n_voxels,) Spearman-Brown corrected split-half reliability (the ceiling on r).
    """
    arr = np.stack(responses_by_subject)  # (S, n_samples, n_voxels)
    S = arr.shape[0]
    rng = np.random.default_rng(seed)
    rels = []
    for _ in range(n_splits):
        perm = rng.permutation(S)
        h1 = arr[perm[: S // 2]].mean(axis=0)
        h2 = arr[perm[S // 2 :]].mean(axis=0)
        r = pearson_per_column(h1, h2)
        # Spearman-Brown correction for halving
        sb = (2 * r) / (1 + r)
        rels.append(sb)
    ceiling = np.nanmean(np.stack(rels), axis=0)
    return np.clip(ceiling, 1e-6, 1.0)
