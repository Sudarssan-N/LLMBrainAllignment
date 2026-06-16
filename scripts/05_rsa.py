"""Stage 5: RSA — compare model-layer representational geometry to brain geometry.

Requires Stage 1 activations. Builds RDMs per layer and correlates with the brain RDM.

Output: data/derived/<model>/rsa.json   (per-layer RSA score)

Usage:
    python scripts/05_rsa.py --model gpt2
    python scripts/05_rsa.py --synthetic
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.data.pereira import load_pereira2018
from llmbrain.encoding.rsa import compute_rdm, compare_rdms
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

    rcfg = cfg.raw["rsa"]
    brain_rdm = compute_rdm(ds.responses, metric=rcfg["rdm_metric"])

    results = {}
    for l in layers:
        model_rdm = compute_rdm(acts[f"layer_{l}"], metric=rcfg["rdm_metric"])
        score = compare_rdms(model_rdm, brain_rdm, method=rcfg["compare"])
        results[l] = score
        print(f"[stage5] layer {l:2d}  rsa={score:.4f}")

    best = max(results, key=results.get)
    save_json(out_dir / "rsa.json", {
        "model_key": spec.key,
        "is_synthetic": ds.is_synthetic,
        "rdm_metric": rcfg["rdm_metric"],
        "compare": rcfg["compare"],
        "per_layer": results,
        "best_layer": best,
        "best_rsa": results[best],
    })
    print(f"[stage5] best layer={best} rsa={results[best]:.4f} -> {out_dir/'rsa.json'}")


if __name__ == "__main__":
    main()
