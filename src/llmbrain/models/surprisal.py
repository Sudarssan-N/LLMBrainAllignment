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
    # features: (n_sentences, n_reducers) in the order of `reducers`
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
) -> SurprisalResult:
    """Compute per-sentence surprisal features.

    base=2 gives surprisal in bits; base=e (math.e) gives nats.
    """
    import torch

    reducers = reducers or ["mean", "sum", "last"]
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
        feats.append([_reduce(surprisal, r) for r in reducers])

    return SurprisalResult(
        hf_id=hf_id,
        reducers=reducers,
        features=np.asarray(feats, dtype=np.float32),
        per_sentence=per_sentence,
    )
