"""Per-token and per-sentence surprisal from a HuggingFace causal LM.

Surprisal of token t = -log P(t | context). We compute it from the model's shifted
logits and summarize per sentence (mean / sum / last token) to form the low-dimensional
"surprisal" predictor that variance partitioning controls for.

The reference surprisal model is configurable but defaults to the same model whose
activations are being probed; the proposal specifically suggests GPT-2 surprisal as a
canonical control, which can be selected via `surprisal_model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .activations import load_model_and_tokenizer, _select_device


@dataclass
class SurprisalResult:
    hf_id: str
    reducers: list[str]
    # feature_names: column labels for `features`, e.g.
    # ["surprisal_mean", "surprisal_sum", "surprisal_last", "entropy_mean", "entropy_last"]
    feature_names: list[str]
    # features: (n_sentences, n_features) — surprisal reducers then entropy reducers
    features: np.ndarray
    per_sentence: list[np.ndarray]  # ragged per-token surprisal


def _reduce(values: np.ndarray, reducer: str) -> float:
    if values.size == 0:
        return 0.0
    if reducer == "mean":
        return float(values.mean())
    if reducer == "sum":
        return float(values.sum())
    if reducer == "last":
        return float(values[-1])
    if reducer == "max":
        return float(values.max())
    raise ValueError(f"unknown reducer: {reducer}")


def compute_surprisal(
    sentences: list[str],
    hf_id: str,
    device: str = "cpu",
    reducers: list[str] | None = None,
    base: float = 2.0,
    cache_dir: str | None = None,
    max_length: int = 512,
    entropy_reducers: list[str] | None = None,
) -> SurprisalResult:
    """Compute per-sentence surprisal (and optional predictive-entropy) features.

    base=2 gives surprisal in bits; base=e (math.e) gives nats.

    `entropy_reducers` (e.g. ["mean", "last"]) appends per-sentence summaries of the
    next-token predictive entropy H_t = -sum_v p(v) log p(v) at each position. Entropy is a
    distribution-level uncertainty signal distinct from the realized surprisal of the
    observed token, and enriches the controlled nuisance space toward a fairer
    dimensionality match with the hidden states. Pass None/[] to disable.
    """
    import torch

    reducers = reducers or ["mean", "sum", "last"]
    entropy_reducers = entropy_reducers or []
    model, tok = load_model_and_tokenizer(hf_id, device, cache_dir)
    dev = _select_device(device)
    log_base = float(np.log(base))

    per_sentence: list[np.ndarray] = []
    feats: list[list[float]] = []

    for sent in tqdm(sentences, desc=f"surprisal:{hf_id}"):
        enc = tok(sent, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits  # (1, T, V)
        # log P(token_t | < t): align logits[:-1] with input_ids[1:]
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (T, V)
        ids = enc["input_ids"][0]  # (T,)
        tgt = ids[1:]
        chosen = log_probs[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)  # (T-1,)
        surprisal = (-chosen.cpu().numpy()) / log_base  # convert nats -> chosen base
        per_sentence.append(surprisal.astype(np.float32))
        row = [_reduce(surprisal, r) for r in reducers]
        if entropy_reducers:
            # H_t over the predictive distribution at each context position (same alignment
            # as surprisal: positions 0..T-2 predict tokens 1..T-1).
            ent = (-(log_probs[:-1].exp() * log_probs[:-1]).sum(dim=-1)).cpu().numpy()
            ent = ent / log_base
            row += [_reduce(ent, r) for r in entropy_reducers]
        feats.append(row)

    feature_names = [f"surprisal_{r}" for r in reducers] + \
        [f"entropy_{r}" for r in entropy_reducers]

    return SurprisalResult(
        hf_id=hf_id,
        reducers=reducers,
        feature_names=feature_names,
        features=np.asarray(feats, dtype=np.float32),
        per_sentence=per_sentence,
    )
