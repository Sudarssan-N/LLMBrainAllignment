"""Stage 1: extract layer-wise pooled activations for a model over the dataset stimuli.

Output: data/derived/<model>/activations.npz  (one array per layer: layer_<i>)
        data/derived/<model>/sentences.json    (the stimulus order they correspond to)

Usage:
    python scripts/01_extract_activations.py --model gpt2
    python scripts/01_extract_activations.py --model gpt2 --synthetic   # toy stimuli
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.data.pereira import load_pereira2018
from llmbrain.data.synthetic import synthetic_activations
from llmbrain.utils.io import save_arrays, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="model key from config")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic stimuli")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.model_spec(args.model)
    device = cfg.resolve_device()

    ds = load_pereira2018(
        experiments=cfg.raw["dataset"].get("experiments"),
        atlas=cfg.raw["dataset"].get("atlas", "language"),
        allow_synthetic=args.synthetic,
        seed=cfg.seed,
    )
    print(f"[stage1] model={spec.key} device={device} n_sentences={ds.n_samples} "
          f"synthetic={ds.is_synthetic}")

    pool = cfg.raw["pooling"]
    out_dir = cfg.derived_dir(spec.key)

    if ds.is_synthetic:
        # No-torch path: fabricate layer activations so the pipeline runs end-to-end.
        activations = synthetic_activations(ds, seed=cfg.seed)
        layer_indices = sorted(activations)
        print("[stage1] SYNTHETIC activations (no model loaded)")
    else:
        from llmbrain.models.activations import extract_activations

        res = extract_activations(
            sentences=ds.sentences,
            hf_id=spec.hf_id,
            model_key=spec.key,
            device=device,
            layers=spec.layers,
            pooling_method=pool.get("method", "mean"),
            include_bos=pool.get("include_bos", False),
            cache_dir=cfg.hf_cache(),
            batch_size=args.batch_size,
        )
        activations, layer_indices = res.activations, res.layer_indices

    arrays = {f"layer_{l}": activations[l] for l in layer_indices}
    save_arrays(out_dir / "activations.npz", **arrays)
    save_json(out_dir / "sentences.json", {"sentences": ds.sentences,
                                           "is_synthetic": ds.is_synthetic})
    save_json(out_dir / "activations_meta.json", {
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "layers": layer_indices,
        "pooling": pool,
        "n_sentences": ds.n_samples,
        "is_synthetic": ds.is_synthetic,
    })
    print(f"[stage1] wrote {out_dir/'activations.npz'} ({len(layer_indices)} layers)")


if __name__ == "__main__":
    main()
