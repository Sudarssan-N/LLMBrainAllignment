"""Synthetic dataset — a labelled stand-in so the full pipeline runs end-to-end before
real Pereira2018 access is wired up. Responses are generated as a linear function of a
hidden "semantic" factor plus a surprisal-correlated factor plus noise, so variance
partitioning has a known-positive unique-hidden signal to recover in tests.

NEVER report synthetic numbers as results: `BrainDataset.is_synthetic` is always True.
"""

from __future__ import annotations

import numpy as np

from .dataset import BrainDataset

_WORDS = (
    "the cat sat on a mat by the river while birds sang softly above green trees and "
    "children played near the old stone bridge under a bright morning sky".split()
)


def _fake_sentence(rng: np.random.Generator) -> str:
    n = rng.integers(5, 12)
    return " ".join(rng.choice(_WORDS, size=n)) + "."


def make_synthetic_dataset(
    n_samples: int = 60,
    n_voxels: int = 40,
    n_subjects: int = 6,
    hidden_dim: int = 16,
    seed: int = 0,
    unique_hidden_strength: float = 1.0,
) -> BrainDataset:
    """Generate a synthetic BrainDataset with a recoverable unique-hidden signal."""
    rng = np.random.default_rng(seed)
    sentences = [_fake_sentence(rng) for _ in range(n_samples)]

    # latent factors
    semantic = rng.normal(size=(n_samples, hidden_dim))          # drives hidden states
    surprisal_latent = rng.normal(size=(n_samples, 1))           # drives surprisal feature
    # correlate them partially (collinearity, as in real data)
    semantic[:, 0] += 0.7 * surprisal_latent[:, 0]

    # voxel weights
    W_sem = rng.normal(size=(hidden_dim, n_voxels)) * unique_hidden_strength
    w_surp = rng.normal(size=(1, n_voxels))

    base = semantic @ W_sem + surprisal_latent @ w_surp

    responses_by_subject = []
    for _ in range(n_subjects):
        noise = rng.normal(scale=2.0, size=(n_samples, n_voxels))
        responses_by_subject.append((base + noise).astype(np.float32))
    responses = np.mean(responses_by_subject, axis=0).astype(np.float32)

    roi = rng.integers(0, 3, size=n_voxels)  # 3 fake ROIs
    position = (np.arange(n_samples) % 4).astype(np.float64)  # fake passages of 4 sentences

    return BrainDataset(
        sentences=sentences,
        responses=responses,
        responses_by_subject=responses_by_subject,
        roi=roi,
        subject_ids=[f"S{i:02d}" for i in range(n_subjects)],
        position=position,
        name="synthetic",
        is_synthetic=True,
        meta={"latent_semantic": semantic, "latent_surprisal": surprisal_latent},
    )


def synthetic_activations(
    ds: BrainDataset, n_layers: int = 6, hidden_dim: int = 32, seed: int = 0
) -> dict[int, np.ndarray]:
    """Fake per-layer activations derived from the dataset's latent semantic factor.

    Lets the script pipeline (stages 1-5) run end-to-end without torch/transformers.
    Middle layers carry the strongest signal, mimicking the intermediate-layer advantage.
    """
    rng = np.random.default_rng(seed + 100)
    semantic = ds.meta["latent_semantic"]  # (n_samples, k)
    n = semantic.shape[0]
    acts = {}
    for l in range(n_layers):
        # bell-shaped signal weight peaking at the middle layer
        w = 1.0 - abs(l - (n_layers - 1) / 2) / ((n_layers - 1) / 2 + 1e-9)
        proj = rng.normal(size=(semantic.shape[1], hidden_dim))
        signal = (semantic @ proj) * (0.3 + 0.7 * w)
        noise = rng.normal(scale=0.5, size=(n, hidden_dim))
        acts[l] = (signal + noise).astype(np.float32)
    return acts


def synthetic_surprisal_features(
    ds: BrainDataset, reducers: list[str], seed: int = 0,
    entropy_reducers: list[str] | None = None,
) -> np.ndarray:
    """Fake per-sentence surprisal (+ entropy) features from the latent surprisal factor.

    Entropy columns are generated as a noisier correlate of the same latent so the
    synthetic pipeline exercises the wider nuisance space without claiming realism.
    """
    rng = np.random.default_rng(seed + 200)
    base = ds.meta["latent_surprisal"]  # (n_samples, 1)
    n_extra = len(entropy_reducers or [])
    cols = [base[:, 0] + rng.normal(scale=0.1, size=base.shape[0]) for _ in reducers]
    cols += [base[:, 0] + rng.normal(scale=0.4, size=base.shape[0]) for _ in range(n_extra)]
    return np.stack(cols, axis=1).astype(np.float32)
