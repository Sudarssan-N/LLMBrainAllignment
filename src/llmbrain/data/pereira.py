"""Pereira et al. (2018) language-network fMRI loader.

Primary path: the `brain-score/language` harness, which bundles Pereira2018 as a
`NeuroidAssembly` (sentences × neuroids, with subject/experiment/atlas metadata). We adapt
that assembly into our `BrainDataset`. The harness is heavy to install, so this loader
imports it lazily and raises a clear, actionable error if it's missing.

If `allow_synthetic=True` and the real data can't be loaded, returns the labelled synthetic
dataset so downstream code is still exercisable — but `is_synthetic` will be True.

References:
    Pereira et al. 2018, Nat. Commun. 9:963 (OSF: crwz7).
    Schrimpf et al. 2021, PNAS 118(45):e2105646118 (brain-score/language).
"""

from __future__ import annotations

import numpy as np

from .dataset import BrainDataset
from .synthetic import make_synthetic_dataset


def load_pereira2018(
    experiments: list[int] | None = None,
    atlas: str = "language",
    allow_synthetic: bool = False,
    seed: int = 0,
) -> BrainDataset:
    """Load Pereira2018 into a BrainDataset.

    Args:
        experiments: which sentence sets to include (243, 384). None = all.
        atlas: voxel atlas/network to keep (e.g. 'language').
        allow_synthetic: if True, fall back to synthetic data instead of raising.
    """
    try:
        return _load_via_brainscore(experiments, atlas)
    except Exception as e:  # noqa: BLE001 - we want to surface the reason
        if allow_synthetic:
            ds = make_synthetic_dataset(seed=seed)
            ds.meta["fallback_reason"] = str(e)
            return ds
        raise RuntimeError(
            "Could not load real Pereira2018 via brain-score/language.\n"
            f"Underlying error: {e}\n"
            "Install the harness in Colab with:\n"
            "    pip install git+https://github.com/brain-score/language\n"
            "or pass allow_synthetic=True to run the pipeline on labelled synthetic data."
        ) from e


def _load_via_brainscore(experiments, atlas) -> BrainDataset:
    """Adapt the brain-score Pereira2018 assembly to a BrainDataset.

    The exact accessor in brain-score/language has shifted across versions; we try the
    documented benchmark data packager and fall back to the assembly registry. This is the
    one place to update if the harness API changes.
    """
    assembly = None
    errors = []

    # Preferred: data packaging accessor.
    try:
        from brainscore_language import load_dataset  # type: ignore

        assembly = load_dataset("Pereira2018.language")
    except Exception as e:  # noqa: BLE001
        errors.append(f"load_dataset: {e}")

    if assembly is None:
        try:
            from brainscore_language.data_packaging.pereira2018 import (  # type: ignore
                upload_pereira2018,
            )

            assembly = upload_pereira2018()
        except Exception as e:  # noqa: BLE001
            errors.append(f"data_packaging: {e}")

    if assembly is None:
        raise ImportError("; ".join(errors) or "brainscore_language not importable")

    return _assembly_to_dataset(assembly, experiments, atlas)


def _assembly_to_dataset(assembly, experiments, atlas) -> BrainDataset:
    """Convert an xarray NeuroidAssembly (presentation × neuroid) to BrainDataset.

    Expected coords (names vary slightly by version):
      - presentation: 'stimulus'/'sentence', 'experiment'
      - neuroid: 'subject', 'atlas'/'roi'
    """
    import numpy as np  # local to keep import errors scoped

    a = assembly
    # Filter by atlas/network if available.
    neuroid_dim = "neuroid"
    if atlas and "atlas" in a.coords:
        a = a.sel(neuroid=a["atlas"].values == atlas) if neuroid_dim in a.dims else a
    # Filter by experiment if available.
    if experiments and "experiment" in a.coords:
        keep = np.isin(a["experiment"].values, np.asarray(experiments).astype(str)) | np.isin(
            a["experiment"].values, np.asarray(experiments)
        )
        a = a.isel(presentation=np.flatnonzero(keep))

    responses = np.asarray(a.values, dtype=np.float32)  # (presentation, neuroid)

    # sentence text
    sent_coord = next(
        (c for c in ("stimulus", "sentence", "stimulus_sentence", "word") if c in a.coords),
        None,
    )
    sentences = (
        [str(s) for s in a[sent_coord].values]
        if sent_coord
        else [f"stim_{i}" for i in range(responses.shape[0])]
    )

    roi = None
    for c in ("roi", "atlas", "region"):
        if c in a.coords:
            roi = np.asarray(a[c].values)
            break

    subjects = None
    responses_by_subject = None
    if "subject" in a.coords:
        subj_vals = np.asarray(a["subject"].values)
        subjects = sorted(set(subj_vals.tolist()))
        # group neuroids by subject so each subject's voxels form a repeat estimate is
        # not directly possible (voxels differ per subject); instead expose per-subject
        # response matrices padded to a common voxel count for noise-ceiling estimation.
        responses_by_subject = _per_subject_matrices(responses, subj_vals, subjects)

    return BrainDataset(
        sentences=sentences,
        responses=responses,
        responses_by_subject=responses_by_subject,
        roi=roi,
        subject_ids=subjects,
        name="pereira2018",
        is_synthetic=False,
        meta={"atlas": atlas, "experiments": experiments},
    )


def _per_subject_matrices(responses, subj_vals, subjects):
    """Split a (samples, neuroids) matrix into per-subject (samples, neuroids_subj) blocks.

    For Pereira, voxels are subject-specific, so a true cross-subject split-half ceiling
    requires a shared stimulus space. We return per-subject blocks; the noise-ceiling
    routine should be applied within a shared-voxel projection (left to the analysis step).
    """
    blocks = []
    for s in subjects:
        cols = np.flatnonzero(subj_vals == s)
        blocks.append(responses[:, cols])
    return blocks
