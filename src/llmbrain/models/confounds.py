"""Low-level confound features for nuisance-controlled variance partitioning.

Recent work (Hadidi, Feghhi, Song, Blank & Kao, Nat. Commun. 2026, "Spurious alignment
in LLM-brain comparisons") shows that simple confounds — chiefly **positional signals**
and **word rate** — are competitive with trained LLMs on Pereira2018 and fully account for
the predictivity of *untrained* models. Surprisal alone is therefore an insufficient
nuisance control: a hidden-state representation could appear to "uniquely" explain neural
variance simply by carrying sentence length / position information that a 3-D surprisal
summary does not absorb.

This module builds a small, interpretable confound feature block from the stimulus text
(and, where available, sentence position within its passage). Concatenated with surprisal,
it forms the nuisance space N that variance partitioning controls for, so the reported
unique-hidden contribution is variance *beyond surprisal and these low-level confounds*.

All features are text-derivable except `position`, which the dataset loader supplies from
the assembly metadata when present. `wordfreq` is an optional dependency: if it is not
installed, the frequency feature is silently dropped (the rest still compute).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class ConfoundResult:
    names: list[str]
    # features: (n_sentences, n_names) in the order of `names`
    features: np.ndarray


def _words(sentence: str) -> list[str]:
    return _WORD_RE.findall(sentence)


def _try_word_frequency():
    """Return a callable freq(word)->float in [0,1], or None if wordfreq is unavailable."""
    try:
        from wordfreq import word_frequency  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        return None

    def freq(w: str) -> float:
        return word_frequency(w.lower(), "en")

    return freq


def compute_confounds(
    sentences: list[str],
    positions: np.ndarray | None = None,
    include_frequency: bool = True,
) -> ConfoundResult:
    """Build the low-level confound feature block.

    Args:
        sentences: stimulus strings (length n).
        positions: optional (n,) sentence position within passage (or a running index).
            When provided it is z-scored and added as a `position` feature; the squared
            term `position_sq` captures non-linear primacy/recency drift.
        include_frequency: add a mean log word-frequency feature if `wordfreq` is installed.

    Returns:
        ConfoundResult with `features` (n, k) and matching `names`.

    Features (always): n_words, n_chars, mean_word_len.
    Optional: log_freq (if wordfreq), position + position_sq (if positions given).
    """
    n_words = np.empty(len(sentences), dtype=np.float64)
    n_chars = np.empty(len(sentences), dtype=np.float64)
    mean_word_len = np.empty(len(sentences), dtype=np.float64)

    freq_fn = _try_word_frequency() if include_frequency else None
    log_freq = np.empty(len(sentences), dtype=np.float64) if freq_fn else None

    for i, sent in enumerate(sentences):
        words = _words(sent)
        n_words[i] = len(words)
        n_chars[i] = len(sent)
        mean_word_len[i] = float(np.mean([len(w) for w in words])) if words else 0.0
        if freq_fn is not None:
            # mean over words of log10(freq); unseen/zero-freq words floored to avoid -inf
            fs = [freq_fn(w) for w in words]
            fs = [f for f in fs if f > 0]
            log_freq[i] = float(np.mean(np.log10(fs))) if fs else -8.0

    cols = [n_words, n_chars, mean_word_len]
    names = ["n_words", "n_chars", "mean_word_len"]

    if log_freq is not None:
        cols.append(log_freq)
        names.append("log_freq")

    if positions is not None:
        pos = np.asarray(positions, dtype=np.float64)
        std = pos.std()
        posz = (pos - pos.mean()) / std if std > 0 else np.zeros_like(pos)
        cols.append(posz)
        names.append("position")
        cols.append(posz**2)
        names.append("position_sq")

    features = np.stack(cols, axis=1).astype(np.float32)
    return ConfoundResult(names=names, features=features)
