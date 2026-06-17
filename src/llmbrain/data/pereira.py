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


def _dim_level_names(a, dim) -> list[str]:
    """All coordinate names addressable on `dim`, including MultiIndex levels.

    brain-score assemblies pack stimulus/experiment/subject/atlas as *levels* of the
    'presentation' and 'neuroid' MultiIndexes, so they don't show up in `list(a.coords)`.
    """
    names: list[str] = []
    idx = a.indexes.get(dim)
    for n in getattr(idx, "names", []) or []:
        if n is not None and n != dim:
            names.append(n)
    for c in a.coords:  # plain (non-index) coords attached to this dim
        if c != dim and getattr(a[c], "dims", ()) == (dim,) and c not in names:
            names.append(c)
    return names


def _level_values(a, dim, name):
    """Values of a coord/level on `dim`, whether plain coord or MultiIndex level."""
    import numpy as np

    if name in a.coords:
        return np.asarray(a[name].values)
    idx = a.indexes.get(dim)
    if idx is not None and name in (getattr(idx, "names", []) or []):
        return np.asarray(idx.get_level_values(name))
    raise KeyError(name)


def _pick_sentence_level(a, dim, names) -> str | None:
    """Choose the level holding sentence text: prefer name match, else most whitespace-y."""
    import numpy as np

    preferred = ("sentence", "stimulus_sentence", "sentence_text", "stimulus", "word")
    for p in preferred:
        if p in names:
            return p
    best, best_score = None, -1.0
    for n in names:
        try:
            vals = _level_values(a, dim, n)
        except Exception:  # noqa: BLE001
            continue
        if vals.dtype.kind not in ("U", "S", "O"):
            continue
        sv = vals.astype(str)
        score = float(np.mean([" " in s for s in sv]))  # sentences contain spaces
        if score > best_score:
            best, best_score = n, score
    return best if best_score > 0 else None


def _filter_dim(a, dim, name, wanted):
    """Boolean-filter `a` along `dim` where level `name` ∈ wanted (tolerant; substring)."""
    import numpy as np

    vals = _level_values(a, dim, name)
    vals_str = vals.astype(str)
    wanted_str = [str(w) for w in wanted]
    keep = np.isin(vals, np.asarray(wanted, dtype=object)) | np.isin(vals_str, wanted_str)
    for w in wanted_str:
        keep |= np.char.find(vals_str, w) >= 0
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        print(f"[pereira] WARNING: filter {name}∈{wanted} matched 0 of {vals.size} "
              f"(e.g. {sorted(set(vals_str))[:4]}); keeping all.")
        return a
    return a.isel({dim: idx})


def _assembly_to_dataset(assembly, experiments, atlas) -> BrainDataset:
    """Convert an xarray NeuroidAssembly (presentation × neuroid) to BrainDataset.

    Reads MultiIndex levels, filters to the requested experiment(s) and atlas, then drops
    NaN voxels (Pereira voxels are subject/experiment-specific, so pooling leaves NaN
    blocks). Prints the discovered schema for diagnosability.
    """
    import numpy as np

    a = assembly
    pres_dim = a.dims[0]
    neur_dim = a.dims[1]
    pres_levels = _dim_level_names(a, pres_dim)
    neur_levels = _dim_level_names(a, neur_dim)
    print(f"[pereira] dims={a.dims} shape={a.shape}")
    print(f"[pereira] presentation levels={pres_levels}")
    print(f"[pereira] neuroid levels={neur_levels}")

    # --- atlas filter (neuroid dim) ---
    if atlas:
        atlas_lvl = next((n for n in neur_levels if n in ("atlas", "roi", "region")), None)
        if atlas_lvl:
            a = _filter_dim(a, neur_dim, atlas_lvl, [atlas])

    # --- experiment filter (presentation dim) ---
    exp_lvl = next((n for n in pres_levels if "experiment" in n.lower()), None)
    if experiments and exp_lvl:
        a = _filter_dim(a, pres_dim, exp_lvl, experiments)
    elif experiments and not exp_lvl:
        print(f"[pereira] WARNING: no experiment level found among {pres_levels}; "
              "cannot filter — NaN voxels likely if multiple experiments are pooled.")

    # --- sentences ---
    sent_lvl = _pick_sentence_level(a, pres_dim, pres_levels)
    if sent_lvl:
        sentences = [str(s) for s in _level_values(a, pres_dim, sent_lvl)]
    else:
        sentences = [f"stim_{i}" for i in range(a.shape[0])]
        print(f"[pereira] WARNING: no sentence-text level found among {pres_levels}; "
              "using placeholders (activations would be meaningless!).")

    responses = np.asarray(a.values, dtype=np.float32)  # (presentation, neuroid)

    # --- drop NaN voxels (columns), then any residual NaN rows ---
    col_ok = ~np.isnan(responses).any(axis=0)
    n_drop = int((~col_ok).sum())
    if n_drop:
        print(f"[pereira] dropping {n_drop}/{responses.shape[1]} voxels with NaN "
              f"(subject/experiment-specific coverage)")
    responses = responses[:, col_ok]
    row_ok = ~np.isnan(responses).any(axis=1)
    if (~row_ok).any():
        print(f"[pereira] dropping {int((~row_ok).sum())} presentation rows with residual NaN")
        responses = responses[row_ok]
        sentences = [s for s, keep in zip(sentences, row_ok) if keep]

    if responses.shape[1] == 0:
        raise ValueError(
            "All voxels were dropped as NaN. This usually means multiple experiments were "
            "pooled (disjoint subjects/voxels). Set dataset.experiments to a single "
            "experiment (e.g. [384]) in the config and re-run."
        )

    # --- subject / roi metadata (post-filter, post-voxel-drop) ---
    subj_lvl = next((n for n in neur_levels if "subject" in n.lower()), None)
    roi_lvl = next((n for n in neur_levels if n in ("roi", "atlas", "region")), None)
    roi = _level_values(a, neur_dim, roi_lvl)[col_ok] if roi_lvl else None
    subjects = responses_by_subject = None
    if subj_lvl:
        subj_vals = _level_values(a, neur_dim, subj_lvl)[col_ok]
        subjects = sorted(set(subj_vals.tolist()))
        responses_by_subject = _per_subject_matrices(responses, subj_vals, subjects)

    print(f"[pereira] final: {responses.shape[0]} sentences x {responses.shape[1]} voxels"
          f"; sentence_level={sent_lvl!r}; example={sentences[0][:60]!r}")

    return BrainDataset(
        sentences=sentences,
        responses=responses,
        responses_by_subject=responses_by_subject,
        roi=roi,
        subject_ids=subjects,
        name="pereira2018",
        is_synthetic=False,
        meta={"atlas": atlas, "experiments": experiments,
              "sentence_level": sent_lvl, "experiment_level": exp_lvl},
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
