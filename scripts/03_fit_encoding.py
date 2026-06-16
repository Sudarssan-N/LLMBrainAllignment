"""Stage 3: fit ridge encoding models (per layer) and report normalized predictivity.

Reproduces the Brain-Score linear-predictivity curve over layers. Requires Stage 1
activations. Loads brain responses from the dataset (deterministic, so synthetic responses
align with synthetic activations by seed).

Output: data/derived/<model>/encoding.json   (per-layer mean r + normalized r)

Usage:
    python scripts/03_fit_encoding.py --model gpt2
    python scripts/03_fit_encoding.py --synthetic     # toy end-to-end
"""

from __future__ import annotations

import argparse

import numpy as np
import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.data.pereira import load_pereira2018
from llmbrain.encoding.ridge import ridge_encode, split_half_noise_ceiling
from llmbrain.utils.io import load_arrays, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.model_spec(args.model)
    out_dir = cfg.derived_dir(spec.key)

    acts = load_arrays(out_dir / "activations.npz")
    layers = sorted(int(k.split("_")[1]) for k in acts)

    ds = load_pereira2018(
        experiments=cfg.raw["dataset"].get("experiments"),
        atlas=cfg.raw["dataset"].get("atlas", "language"),
        allow_synthetic=args.synthetic,
        seed=cfg.seed,
    )
    Y = ds.responses

    nc = None
    if cfg.raw["noise_ceiling"]["method"] == "split_half" and ds.responses_by_subject:
        try:
            nc = split_half_noise_ceiling(
                ds.responses_by_subject,
                n_splits=cfg.raw["noise_ceiling"]["n_splits"],
                seed=cfg.seed,
            )
        except Exception as e:  # noqa: BLE001 - voxel mismatch across subjects, etc.
            print(f"[stage3] noise ceiling skipped: {e}")

    enc = cfg.raw["encoding"]
    results = {}
    for l in layers:
        X = acts[f"layer_{l}"]
        res = ridge_encode(
            X, Y,
            alphas=enc["alphas"],
            n_folds=enc["n_folds"],
            standardize=enc["standardize"],
            pca_components=enc.get("pca_components"),
            seed=cfg.seed,
        )
        norm = None
        if nc is not None and nc.shape[0] == res.voxel_r.shape[0]:
            norm = float(np.nanmean(res.voxel_r / nc))
        results[l] = {"mean_r": res.mean_r, "normalized_r": norm}
        print(f"[stage3] layer {l:2d}  mean_r={res.mean_r:.4f}"
              + (f"  norm_r={norm:.4f}" if norm is not None else ""))

    best = max(results, key=lambda l: results[l]["mean_r"])
    payload = {
        "model_key": spec.key,
        "is_synthetic": ds.is_synthetic,
        "per_layer": results,
        "best_layer": best,
        "best_mean_r": results[best]["mean_r"],
        "n_voxels": int(Y.shape[1]),
    }
    save_json(out_dir / "encoding.json", payload)
    print(f"[stage3] best layer={best} mean_r={results[best]['mean_r']:.4f} "
          f"-> {out_dir/'encoding.json'}")


if __name__ == "__main__":
    main()
