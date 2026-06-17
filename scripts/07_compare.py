"""Stage 7: paired comparison of two models' peak-layer unique-hidden (e.g. E004).

Loads the per-voxel unique-hidden arrays saved by Stage 4 (varpart_peak_voxels.npz) for two
models and runs a paired bootstrap test over shared voxels. Use for base vs instruction-tuned.

Usage:
    python scripts/07_compare.py --a qwen2.5-0.5b --b qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from llmbrain.config import load_config
from llmbrain.encoding.stats import paired_diff_ci
from llmbrain.utils.io import load_arrays, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--a", required=True, help="model A key (e.g. base)")
    ap.add_argument("--b", required=True, help="model B key (e.g. instruct)")
    ap.add_argument("--metric", default="unique_hidden",
                    choices=["unique_hidden", "unique_surprisal"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = cfg.derived_dir()
    a = load_arrays(base / args.a / "varpart_peak_voxels.npz")
    b = load_arrays(base / args.b / "varpart_peak_voxels.npz")

    res = paired_diff_ci(a[args.metric], b[args.metric], seed=cfg.seed)
    res.update({"a": args.a, "b": args.b, "metric": args.metric,
                "a_peak_layer": int(a["layer"][0]), "b_peak_layer": int(b["layer"][0])})

    sig = "SIGNIFICANT" if res["significant"] else "n.s."
    print(f"[stage7] {args.metric}: {args.a} - {args.b} = {res['mean_diff']:+.5f} "
          f"[{res['lo']:+.5f}, {res['hi']:+.5f}] 95% CI  ->  {sig}  (n={res['n']} voxels)")
    out = save_json(base / f"compare_{args.a}_vs_{args.b}.json", res)
    print(f"[stage7] wrote {out}")


if __name__ == "__main__":
    main()
