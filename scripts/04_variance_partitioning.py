"""Stage 4: variance partitioning — the project's core analysis.

Decomposes per-voxel explained variance into unique(hidden), unique(surprisal), shared.
Requires Stage 1 (activations) and Stage 2 (surprisal).

Output: data/derived/<model>/varpart.json   (per-layer summary of unique/shared R²)

Usage:
    python scripts/04_variance_partitioning.py --model gpt2
    python scripts/04_variance_partitioning.py --synthetic --layers best
"""

from __future__ import annotations

import argparse

import numpy as np
import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.data.pereira import load_pereira2018
from llmbrain.encoding.variance_partitioning import variance_partitioning
from llmbrain.utils.io import load_arrays, load_json, save_arrays, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layers", default="all",
                    help="'all', 'best' (from encoding.json), or comma list e.g. 4,6,8")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.model_spec(args.model)
    out_dir = cfg.derived_dir(spec.key)

    acts = load_arrays(out_dir / "activations.npz")
    surp = load_arrays(out_dir / "surprisal.npz")["features"]
    all_layers = sorted(int(k.split("_")[1]) for k in acts)

    if args.layers == "all":
        layers = all_layers
    elif args.layers == "best":
        enc_path = out_dir / "encoding.json"
        layers = [load_json(enc_path)["best_layer"]] if enc_path.exists() else all_layers
    else:
        layers = [int(x) for x in args.layers.split(",")]

    ds = load_pereira2018(
        experiments=cfg.raw["dataset"].get("experiments"),
        atlas=cfg.raw["dataset"].get("atlas", "language"),
        allow_synthetic=args.synthetic,
        seed=cfg.seed,
    )
    Y = ds.responses
    enc = cfg.raw["encoding"]

    results = {}
    vps = {}
    for l in layers:
        vp = variance_partitioning(
            hidden=acts[f"layer_{l}"],
            surprisal=surp,
            Y=Y,
            alphas=enc["alphas"],
            n_folds=enc["n_folds"],
            standardize=enc["standardize"],
            pca_components=enc.get("pca_components"),
            seed=cfg.seed,
        )
        results[l] = vp.summary()
        vps[l] = vp
        s = results[l]
        print(f"[stage4] layer {l:2d}  unique_hidden={s['mean_unique_hidden']:.4f}  "
              f"unique_surprisal={s['mean_unique_surprisal']:.4f}  "
              f"shared={s['mean_shared']:.4f}")

    # Peak layer (by clipped mean unique-hidden) gets bootstrap CIs + per-voxel arrays
    # saved, so base-vs-instruct can be compared with a paired test downstream.
    peak = max(results, key=lambda l: results[l]["mean_unique_hidden"])
    results[peak] = vps[peak].summary(ci=True, seed=cfg.seed)
    ci = results[peak]["ci_unique_hidden"]
    print(f"[stage4] peak layer={peak} unique_hidden={ci['mean']:.4f} "
          f"[{ci['lo']:.4f}, {ci['hi']:.4f}] (95% CI, bootstrap over voxels)")
    save_arrays(out_dir / "varpart_peak_voxels.npz",
                unique_hidden=vps[peak].unique_hidden,
                unique_surprisal=vps[peak].unique_surprisal,
                layer=np.array([peak]))

    save_json(out_dir / "varpart.json", {
        "model_key": spec.key,
        "is_synthetic": ds.is_synthetic,
        "peak_layer": peak,
        "per_layer": results,
    })
    print(f"[stage4] wrote {out_dir/'varpart.json'}")


if __name__ == "__main__":
    main()
