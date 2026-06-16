"""Representational Similarity Analysis (RSA).

Compares the *geometry* of representations rather than linear decodability: build a
Representational Dissimilarity Matrix (RDM) for the model layer and for the brain ROI over
the same stimuli, then correlate their upper triangles. A clean complement to ridge
predictivity for claims about shared linguistic structure (per the proposal's Step 5).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, pearsonr, spearmanr


def compute_rdm(features: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Condensed RDM (upper triangle) over rows (stimuli) of `features`."""
    features = np.asarray(features, dtype=np.float64)
    if metric == "correlation":
        return pdist(features, metric="correlation")
    if metric == "euclidean":
        return pdist(features, metric="euclidean")
    if metric == "cosine":
        return pdist(features, metric="cosine")
    raise ValueError(f"unknown RDM metric: {metric}")


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    """Correlate two condensed RDMs."""
    a = np.asarray(rdm_a, dtype=np.float64)
    b = np.asarray(rdm_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"RDM shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    if method == "pearson":
        return float(pearsonr(a, b).statistic)
    if method == "kendall":
        return float(kendalltau(a, b).statistic)
    raise ValueError(f"unknown comparison method: {method}")


def rsa_score(
    model_features: np.ndarray,
    brain_responses: np.ndarray,
    rdm_metric: str = "correlation",
    compare: str = "spearman",
) -> float:
    """Full RSA: RDM(model) vs RDM(brain) correlation over shared stimuli."""
    rdm_model = compute_rdm(model_features, rdm_metric)
    rdm_brain = compute_rdm(brain_responses, rdm_metric)
    return compare_rdms(rdm_model, rdm_brain, compare)
