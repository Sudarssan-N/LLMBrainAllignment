"""Subword-token pooling.

Pereira (2018) responses are per-sentence (or per-word). LLM tokenizers (BPE for
GPT-2/Pythia, etc.) split words into multiple subword tokens, so we must pool token-level
hidden states back to the unit the brain data is defined over. This module makes the
strategy explicit and testable, per the tokenization-mismatch critique in the proposal.
"""

from __future__ import annotations

import numpy as np


def pool_tokens(
    token_states: np.ndarray,
    method: str = "mean",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pool a (n_tokens, hidden) array to a single (hidden,) vector.

    Args:
        token_states: per-token hidden states for one sentence.
        method: 'mean', 'sum', or 'last' (last *valid* token).
        mask: optional boolean array (n_tokens,); True = include token.
    """
    if token_states.ndim != 2:
        raise ValueError(f"expected (n_tokens, hidden), got {token_states.shape}")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != token_states.shape[0]:
            raise ValueError("mask length must match number of tokens")
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError("mask excludes all tokens")
    else:
        idx = np.arange(token_states.shape[0])

    sel = token_states[idx]
    if method == "mean":
        return sel.mean(axis=0)
    if method == "sum":
        return sel.sum(axis=0)
    if method == "last":
        return sel[-1]
    raise ValueError(f"unknown pooling method: {method}")


def word_spans_from_offsets(
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int] | None = None,
) -> list[list[int]]:
    """Group token indices into word spans using character offset mappings.

    A new word starts at a token whose character span begins on a non-continuation
    boundary (offset start follows whitespace). Tokens with a zero-length offset
    (special tokens like BOS/EOS) are dropped.

    Returns a list of words, each a list of token indices belonging to that word.
    This is a heuristic; for strict alignment prefer the model's
    `tokenizer(..., return_offsets_mapping=True)` together with the original text.
    """
    words: list[list[int]] = []
    prev_end = None
    for i, (start, end) in enumerate(offsets):
        if special_tokens_mask is not None and special_tokens_mask[i]:
            continue
        if start == end:  # special / empty token
            continue
        if prev_end is None or start > prev_end:
            words.append([i])
        else:
            words[-1].append(i)
        prev_end = end
    return words


def build_valid_mask(
    n_tokens: int,
    special_tokens_mask: list[int] | None,
    include_bos: bool,
) -> np.ndarray:
    """Boolean mask over tokens, dropping special tokens unless include_bos is set."""
    mask = np.ones(n_tokens, dtype=bool)
    if special_tokens_mask is not None:
        special = np.asarray(special_tokens_mask, dtype=bool)
        mask &= ~special
        if include_bos and special.any():
            mask[np.flatnonzero(special)[0]] = True
    return mask
