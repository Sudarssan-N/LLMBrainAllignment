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


def _patch_importlib_entry_points() -> None:
    """Restore the dict-style ``EntryPoints.get`` that brain-score's plugin loader needs.

    brain-score-core calls ``importlib.metadata.entry_points().get(group, default)``. Up to
    Python 3.9 ``entry_points()`` returned a ``dict`` (with ``.get``); since Python 3.12 it
    returns a flat ``EntryPoints`` object that has no ``.get`` (use ``.select(group=...)``).
    On Colab (py3.12) this raises ``'EntryPoints' object has no attribute 'get'`` for every
    plugin access. We add a compatible ``.get`` that delegates to ``.select``. Idempotent.
    """
    import importlib.metadata as ilm

    ep_cls = getattr(ilm, "EntryPoints", None)
    if ep_cls is None or hasattr(ep_cls, "get"):
        return

    def _get(self, group, default=None):  # type: ignore[no-untyped-def]
        selected = self.select(group=group)
        return selected if len(selected) else (default if default is not None else selected)

    ep_cls.get = _get  # type: ignore[attr-defined]


def _load_via_brainscore(experiments, atlas) -> BrainDataset:
    """Adapt the brain-score Pereira2018 assembly to a BrainDataset.

    The exact accessor in brain-score/language has shifted across versions; we try the
    documented benchmark data packager and fall back to the assembly registry. This is the
    one place to update if the harness API changes.
    """
    import importlib

    _patch_importlib_entry_points()

    assembly = None
    errors = []
    identifier = "Pereira2018.language"

    # Attempt 1: official API. Note this routes through brainscore's entry-point plugin
    # loader, which is broken on Python 3.12 ('EntryPoints' object has no attribute 'get').
    try:
        from brainscore_language import load_dataset  # type: ignore

        assembly = load_dataset(identifier)
    except Exception as e:  # noqa: BLE001
        errors.append(f"load_dataset: {e}")

    # Attempt 2: bypass the entry-point loader. Importing the plugin module runs its
    # registration code, populating data_registry; then we call the registered loader
    # directly. This sidesteps the py3.12 importlib.metadata incompatibility.
    if assembly is None:
        try:
            from brainscore_language import data_registry  # type: ignore

            if identifier not in data_registry:
                importlib.import_module("brainscore_language.data.pereira2018")
            assembly = data_registry[identifier]()
        except Exception as e:  # noqa: BLE001
            errors.append(f"data_registry: {e}")

    if assembly is None:
        raise ImportError("; ".join(errors) or "brainscore_language not importable")

    return _assembly_to_dataset(assembly, experiments, atlas)


def _coord_dim(a, coord):
    """Return the single dimension a coord is indexed along, or None."""
    try:
        dims = a[coord].dims
        return dims[0] if len(dims) == 1 else None
    except Exception:  # noqa: BLE001
        return None


def _safe_filter(a, coord, wanted, dim_name):
    """Filter assembly `a` along `dim_name` by membership of `coord` in `wanted`.

    Tolerant: matches exact values, their string forms, and substrings (so int 384
    matches the string '384sentences'). If the filter would keep zero rows, it is a
    no-op and a warning is printed — we never silently return an empty assembly.
    """
    import numpy as np

    if coord not in a.coords or not wanted:
        return a
    vals = np.asarray(a[coord].values)
    vals_str = vals.astype(str)
    wanted_str = [str(w) for w in wanted]
    keep = np.isin(vals, np.asarray(wanted)) | np.isin(vals_str, wanted_str)
    for w in wanted_str:  # substring match (384 -> '384sentences')
        keep |= np.char.find(vals_str, w) >= 0
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        print(f"[pereira] WARNING: filter {coord}∈{wanted} matched 0 of "
              f"{vals.size} (values e.g. {sorted(set(vals_str))[:4]}); keeping all.")
        return a
    return a.isel({dim_name: idx})


def _assembly_to_dataset(assembly, experiments, atlas) -> BrainDataset:
    """Convert an xarray NeuroidAssembly (presentation × neuroid) to BrainDataset.

    Tolerant to coord-name/version drift. Prints the discovered schema on load so any
    remaining mismatch is diagnosable from the run output.
    """
    import numpy as np  # local to keep import errors scoped

    a = assembly
    print(f"[pereira] assembly dims={a.dims} shape={a.shape}")
    print(f"[pereira] coords={list(a.coords)}")

    pres_dim = a.dims[0] if a.ndim >= 1 else "presentation"
    neur_dim = a.dims[1] if a.ndim >= 2 else "neuroid"

    # Filter by atlas/network (neuroid dim) and experiment (presentation dim), defensively.
    if atlas:
        atlas_coord = next((c for c in ("atlas", "roi", "region") if c in a.coords), None)
        if atlas_coord:
            a = _safe_filter(a, atlas_coord, [atlas], _coord_dim(a, atlas_coord) or neur_dim)
    if experiments and "experiment" in a.coords:
        a = _safe_filter(a, "experiment", experiments,
                         _coord_dim(a, "experiment") or pres_dim)

    responses = np.asarray(a.values, dtype=np.float32)  # (presentation, neuroid)

    # sentence text
    sent_coord = next(
        (c for c in ("stimulus", "sentence", "stimulus_sentence", "sentence_text",
                     "stimulus_sentence_text", "word") if c in a.coords),
        None,
    )
    sentences = (
        [str(s) for s in np.asarray(a[sent_coord].values)]
        if sent_coord
        else [f"stim_{i}" for i in range(responses.shape[0])]
    )
    print(f"[pereira] using sentence coord={sent_coord!r}; n_sentences={len(sentences)} "
          f"n_neuroids={responses.shape[1]}")

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
