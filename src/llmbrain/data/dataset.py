"""Common dataset container shared by real and synthetic loaders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrainDataset:
    """A set of sentence stimuli aligned to language-network fMRI responses.

    Attributes:
        sentences: list of sentence strings (length n_samples).
        responses: (n_samples, n_voxels) averaged BOLD responses.
        responses_by_subject: optional list of (n_samples, n_voxels) per subject/repeat,
            used to estimate the noise ceiling.
        roi: optional (n_voxels,) ROI/region label per voxel.
        subject_ids: optional list of subject identifiers.
        name: dataset name.
        is_synthetic: True if this is the labelled fallback (never a real result).
    """

    sentences: list[str]
    responses: np.ndarray
    responses_by_subject: list[np.ndarray] | None = None
    roi: np.ndarray | None = None
    subject_ids: list[str] | None = None
    voxel_subject: np.ndarray | None = None  # (n_voxels,) subject id per voxel column
    position: np.ndarray | None = None  # (n_samples,) sentence position within its passage
    name: str = "unknown"
    is_synthetic: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.sentences) != self.responses.shape[0]:
            raise ValueError(
                f"sentences ({len(self.sentences)}) != responses rows "
                f"({self.responses.shape[0]})"
            )

    @property
    def n_samples(self) -> int:
        return self.responses.shape[0]

    @property
    def n_voxels(self) -> int:
        return self.responses.shape[1]
