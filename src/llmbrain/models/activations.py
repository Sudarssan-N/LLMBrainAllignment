"""Layer-wise activation extraction from HuggingFace causal LMs.

For each sentence we run a forward pass with `output_hidden_states=True`, then pool the
token-level hidden states of every layer to a single per-sentence vector (see
`utils.pooling`). Output: dict {layer_index: (n_sentences, hidden_dim)}.

`hidden_states` from HF has length n_layers+1: index 0 is the embedding output, indices
1..n are the transformer blocks. We keep all of them and label layer 0 as the embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from ..utils.pooling import build_valid_mask, pool_tokens


@dataclass
class ActivationResult:
    model_key: str
    hf_id: str
    layer_indices: list[int]
    # activations[layer] -> (n_sentences, hidden)
    activations: dict[int, np.ndarray]
    n_sentences: int


def _select_device(device: str):
    import torch

    return torch.device(device)


def load_model_and_tokenizer(hf_id: str, device: str, cache_dir: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, cache_dir=cache_dir, output_hidden_states=True, torch_dtype=torch.float32
    )
    model.to(_select_device(device))
    model.eval()
    return model, tok


def extract_activations(
    sentences: list[str],
    hf_id: str,
    model_key: str = "model",
    device: str = "cpu",
    layers: list[int] | None = None,
    pooling_method: str = "mean",
    include_bos: bool = False,
    cache_dir: str | None = None,
    batch_size: int = 1,
    max_length: int = 512,
) -> ActivationResult:
    """Extract pooled per-sentence activations for every (selected) layer.

    batch_size>1 pads sentences; padding tokens are masked out before pooling.
    """
    import torch

    model, tok = load_model_and_tokenizer(hf_id, device, cache_dir)
    dev = _select_device(device)

    # Determine number of layers from a probe forward pass.
    with torch.no_grad():
        probe = tok("hello world", return_tensors="pt").to(dev)
        out = model(**probe)
    n_hidden = len(out.hidden_states)  # n_layers + 1 (embeddings + blocks)
    all_layers = list(range(n_hidden))
    sel_layers = all_layers if layers is None else [l for l in layers if l in all_layers]

    acc: dict[int, list[np.ndarray]] = {l: [] for l in sel_layers}

    for start in tqdm(range(0, len(sentences), batch_size), desc=f"activations:{model_key}"):
        batch = sentences[start : start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        special = enc.pop("special_tokens_mask")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)

        attn = enc["attention_mask"].cpu().numpy()
        special_np = special.cpu().numpy()
        for layer in sel_layers:
            hs = out.hidden_states[layer].float().cpu().numpy()  # (B, T, H)
            for b in range(hs.shape[0]):
                n_tok = int(attn[b].sum())
                valid = build_valid_mask(
                    n_tok,
                    special_np[b, :n_tok].tolist(),
                    include_bos=include_bos,
                )
                vec = pool_tokens(hs[b, :n_tok], method=pooling_method, mask=valid)
                acc[layer].append(vec)

    activations = {l: np.stack(v).astype(np.float32) for l, v in acc.items()}
    return ActivationResult(
        model_key=model_key,
        hf_id=hf_id,
        layer_indices=sel_layers,
        activations=activations,
        n_sentences=len(sentences),
    )
