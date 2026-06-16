"""End-to-end pipeline tests on synthetic data (no models, no real fMRI download).

Validates the analytical core: ridge encoding recovers signal, variance partitioning
recovers a positive unique-hidden contribution, RSA is well-formed, pooling and
surprisal-matching behave. Run with: pytest -q
"""

import numpy as np

from llmbrain.data.synthetic import make_synthetic_dataset
from llmbrain.encoding.ridge import ridge_encode, split_half_noise_ceiling, pearson_per_column
from llmbrain.encoding.variance_partitioning import variance_partitioning
from llmbrain.encoding.rsa import compute_rdm, compare_rdms, rsa_score
from llmbrain.analysis.matched_subsets import match_by_binning, matched_balance_report
from llmbrain.utils.pooling import pool_tokens, build_valid_mask

ALPHAS = [1.0, 10.0, 100.0, 1000.0]


def _hidden_and_surprisal(ds):
    """Use the synthetic latent factors as stand-ins for hidden + surprisal features."""
    hidden = ds.meta["latent_semantic"]
    surprisal = ds.meta["latent_surprisal"]
    return hidden, surprisal


def test_synthetic_dataset_shapes():
    ds = make_synthetic_dataset(n_samples=50, n_voxels=30, n_subjects=5, seed=1)
    assert ds.is_synthetic
    assert ds.responses.shape == (50, 30)
    assert len(ds.sentences) == 50
    assert len(ds.responses_by_subject) == 5


def test_ridge_recovers_signal():
    ds = make_synthetic_dataset(seed=2)
    hidden, _ = _hidden_and_surprisal(ds)
    res = ridge_encode(hidden, ds.responses, ALPHAS, n_folds=5, seed=0)
    # hidden states drive the response -> mean held-out r should be clearly positive
    assert res.mean_r > 0.1, res.mean_r
    assert res.voxel_r.shape[0] == ds.responses.shape[1]


def test_variance_partitioning_unique_hidden_positive():
    ds = make_synthetic_dataset(seed=3, unique_hidden_strength=1.5)
    hidden, surprisal = _hidden_and_surprisal(ds)
    vp = variance_partitioning(hidden, surprisal, ds.responses, ALPHAS, n_folds=5, seed=0)
    s = vp.summary()
    # by construction, hidden carries variance beyond the single surprisal factor
    assert s["mean_unique_hidden"] > s["mean_unique_surprisal"]
    assert s["mean_unique_hidden"] > 0.0


def test_noise_ceiling_in_range():
    ds = make_synthetic_dataset(seed=4)
    nc = split_half_noise_ceiling(ds.responses_by_subject, n_splits=5, seed=0)
    assert nc.shape[0] == ds.n_voxels
    assert np.all(nc > 0) and np.all(nc <= 1.0)


def test_rsa_self_is_one():
    ds = make_synthetic_dataset(seed=5)
    rdm = compute_rdm(ds.responses, metric="correlation")
    assert abs(compare_rdms(rdm, rdm, method="spearman") - 1.0) < 1e-6


def test_rsa_score_runs():
    ds = make_synthetic_dataset(seed=6)
    hidden, _ = _hidden_and_surprisal(ds)
    score = rsa_score(hidden, ds.responses)
    assert -1.0 <= score <= 1.0


def test_pooling_methods():
    x = np.arange(12, dtype=float).reshape(4, 3)
    assert np.allclose(pool_tokens(x, "mean"), x.mean(0))
    assert np.allclose(pool_tokens(x, "last"), x[-1])
    mask = np.array([True, False, True, False])
    assert np.allclose(pool_tokens(x, "mean", mask=mask), x[[0, 2]].mean(0))


def test_build_valid_mask_drops_specials():
    mask = build_valid_mask(4, special_tokens_mask=[1, 0, 0, 1], include_bos=False)
    assert mask.tolist() == [False, True, True, False]
    mask2 = build_valid_mask(4, special_tokens_mask=[1, 0, 0, 1], include_bos=True)
    assert mask2.tolist() == [True, True, True, False]


def test_matched_subset_balances_surprisal():
    rng = np.random.default_rng(0)
    surprisal = rng.normal(size=200)
    group = (surprisal + rng.normal(scale=0.5, size=200) > 0).astype(int)  # collinear
    idx = match_by_binning(surprisal, group, n_bins=8, seed=0)
    rep = matched_balance_report(surprisal, group, idx)
    means = list(rep["surprisal_mean_matched"].values())
    # after matching, group surprisal means should be much closer than before
    full_gap = abs(rep["surprisal_mean_full"]["0"] - rep["surprisal_mean_full"]["1"])
    matched_gap = abs(means[0] - means[1])
    assert matched_gap < full_gap
