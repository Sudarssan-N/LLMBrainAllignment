"""Stage 2: compute per-sentence surprisal features for the dataset stimuli.

Output: data/derived/<model>/surprisal.npz  (features: (n_sentences, n_reducers))

By default the surprisal model is the probed model itself; pass --surprisal-model to use a
canonical control (e.g. gpt2) as suggested in the proposal.

Usage:
    python scripts/02_compute_surprisal.py --model gpt2
    python scripts/02_compute_surprisal.py --model pythia-410m --surprisal-model gpt2
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.data.pereira import load_pereira2018
from llmbrain.data.synthetic import synthetic_surprisal_features
from llmbrain.utils.io import load_json, save_arrays, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--surprisal-model", default=None,
                    help="model key to source surprisal from (default: same as --model)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.model_spec(args.model)
    surp_spec = cfg.model_spec(args.surprisal_model) if args.surprisal_model else spec
    device = cfg.resolve_device()

    out_dir = cfg.derived_dir(spec.key)
    scfg = cfg.raw["surprisal"]
    reducers = scfg.get("reducer", ["mean", "sum", "last"])
    entropy_reducers = scfg.get("entropy_reducer", []) or []

    is_synthetic = args.synthetic
    meta_path = out_dir / "activations_meta.json"
    if meta_path.exists():
        is_synthetic = load_json(meta_path).get("is_synthetic", is_synthetic)

    if is_synthetic:
        ds = load_pereira2018(allow_synthetic=True, seed=cfg.seed)
        features = synthetic_surprisal_features(ds, reducers, seed=cfg.seed,
                                                entropy_reducers=entropy_reducers)
        used_reducers = reducers
        feature_names = [f"surprisal_{r}" for r in reducers] + \
            [f"entropy_{r}" for r in entropy_reducers]
        print(f"[stage2] SYNTHETIC surprisal (no model loaded) n={ds.n_samples}")
    else:
        from llmbrain.models.surprisal import compute_surprisal

        sent_path = out_dir / "sentences.json"
        if sent_path.exists():
            sentences = load_json(sent_path)["sentences"]
        else:
            sentences = load_pereira2018(
                experiments=cfg.raw["dataset"].get("experiments"),
                atlas=cfg.raw["dataset"].get("atlas", "language"),
                seed=cfg.seed,
            ).sentences
        print(f"[stage2] surprisal-from={surp_spec.key} n_sentences={len(sentences)} "
              f"entropy_reducers={entropy_reducers}")
        res = compute_surprisal(
            sentences=sentences,
            hf_id=surp_spec.hf_id,
            device=device,
            reducers=reducers,
            base=scfg.get("base", 2.0),
            cache_dir=cfg.hf_cache(),
            entropy_reducers=entropy_reducers,
        )
        features, used_reducers, feature_names = res.features, res.reducers, res.feature_names

    save_arrays(out_dir / "surprisal.npz", features=features)
    save_json(out_dir / "surprisal_meta.json", {
        "probed_model": spec.key,
        "surprisal_model": surp_spec.key,
        "reducers": used_reducers,
        "entropy_reducers": entropy_reducers,
        "feature_names": feature_names,
        "base": scfg.get("base", 2.0),
        "shape": list(features.shape),
        "is_synthetic": is_synthetic,
    })
    print(f"[stage2] wrote {out_dir/'surprisal.npz'} shape={features.shape}")


if __name__ == "__main__":
    main()
