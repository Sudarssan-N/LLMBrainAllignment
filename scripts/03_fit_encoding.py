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
from llmbrain.encoding.noise import cross_subject_noise_ceiling, normalized_predictivity
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
    enc = cfg.raw["encoding"]

    # --- noise ceiling -------------------------------------------------------------
    # Real Pereira: voxels are subject-specific -> cross-subject (brain-to-brain) ceiling.
    # Synthetic: shared voxels with repeats -> split-half reliability.
    nc = None
    if ds.voxel_subject is not None and np.unique(ds.voxel_subject).size >= 2:
        print(f"[stage3] computing cross-subject noise ceiling over "
              f"{np.unique(ds.voxel_subject).size} subjects ...")
        nc = cross_subject_noise_ceiling(
            Y, ds.voxel_subject, alphas=enc["alphas"],
            pca_components=cfg.raw["noise_ceiling"].get("pca_components", 100),
            n_folds=enc["n_folds"], standardize=enc["standardize"], seed=cfg.seed,
        )
        print(f"[stage3] ceiling: mean={np.nanmean(nc):.3f} "
              f"valid_voxels={int(np.isfinite(nc).sum())}")
    elif ds.responses_by_subject:
        try:
            nc = split_half_noise_ceiling(
                ds.responses_by_subject,
                n_splits=cfg.raw["noise_ceiling"]["n_splits"], seed=cfg.seed,
            )
        except Exception as e:  # noqa: BLE001 - voxel mismatch across subjects, etc.
            print(f"[stage3] split-half ceiling skipped: {e}")

    min_ceiling = float(cfg.raw["noise_ceiling"].get("min_ceiling", 0.1))
    results = {}
    for l in layers:
        res = ridge_encode(
            acts[f"layer_{l}"], Y,
            alphas=enc["alphas"], n_folds=enc["n_folds"],
            standardize=enc["standardize"], pca_components=enc.get("pca_components"),
            seed=cfg.seed,
        )
        if nc is not None and nc.shape[0] == res.voxel_r.shape[0]:
            agg = normalized_predictivity(res.voxel_r, nc, min_ceiling=min_ceiling)
            results[l] = {"mean_r": res.mean_r, "normalized_r": agg["normalized_r"],
                          "n_voxels_used": agg["n_voxels_used"]}
            print(f"[stage3] layer {l:2d}  mean_r={res.mean_r:.4f}  "
                  f"norm_r={agg['normalized_r']:.4f}  (n={agg['n_voxels_used']})")
        else:
            results[l] = {"mean_r": res.mean_r, "normalized_r": None}
            print(f"[stage3] layer {l:2d}  mean_r={res.mean_r:.4f}")

    # Rank by normalized predictivity when available, else raw r.
    def score(l):
        v = results[l]["normalized_r"]
        return v if v is not None else results[l]["mean_r"]

    best = max(results, key=score)
    payload = {
        "model_key": spec.key,
        "is_synthetic": ds.is_synthetic,
        "per_layer": results,
        "best_layer": best,
        "best_mean_r": results[best]["mean_r"],
        "best_normalized_r": results[best]["normalized_r"],
        "n_voxels": int(Y.shape[1]),
        "has_noise_ceiling": nc is not None,
    }
    save_json(out_dir / "encoding.json", payload)
    nr = results[best]["normalized_r"]
    print(f"[stage3] best layer={best} mean_r={results[best]['mean_r']:.4f}"
          + (f" norm_r={nr:.4f}" if nr is not None else "")
          + f" -> {out_dir/'encoding.json'}")


if __name__ == "__main__":
    main()
