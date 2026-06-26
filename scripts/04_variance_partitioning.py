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
from llmbrain.models.confounds import compute_confounds
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

    # --- low-level confound block (word rate/length/frequency + position) ---
    # Folded into the controlled nuisance space so unique-hidden is reported as variance
    # beyond surprisal AND these confounds (Hadidi et al. 2026). We also run a
    # surprisal-only partition so the paper can report whether unique-hidden *survives*.
    ccfg = cfg.raw.get("confounds", {}) or {}
    confounds = None
    confound_names: list[str] = []
    if ccfg.get("enabled", False):
        positions = ds.position if ccfg.get("include_position", True) else None
        cres = compute_confounds(
            ds.sentences,
            positions=positions,
            include_frequency=ccfg.get("include_frequency", True),
        )
        confounds, confound_names = cres.features, cres.names
        print(f"[stage4] confound control ON: {confound_names} (shape {confounds.shape})")
    else:
        print("[stage4] confound control OFF (surprisal-only nuisance space)")

    def run_vp(layer, extra):
        return variance_partitioning(
            hidden=acts[f"layer_{layer}"],
            surprisal=surp,
            Y=Y,
            alphas=enc["alphas"],
            n_folds=enc["n_folds"],
            standardize=enc["standardize"],
            pca_components=enc.get("pca_components"),
            seed=cfg.seed,
            extra_nuisance=extra,
        )

    results = {}            # confound-controlled (headline when confounds on)
    results_surp_only = {}  # surprisal-only (original; for the survival comparison)
    vps = {}
    for l in layers:
        vp = run_vp(l, confounds)
        results[l] = vp.summary()
        vps[l] = vp
        s = results[l]
        if confounds is not None:
            vp_so = run_vp(l, None)
            results_surp_only[l] = vp_so.summary()
            print(f"[stage4] layer {l:2d}  unique_hidden={s['mean_unique_hidden']:.4f} "
                  f"(surp-only={results_surp_only[l]['mean_unique_hidden']:.4f})  "
                  f"unique_nuisance={s['mean_unique_surprisal']:.4f}  "
                  f"shared={s['mean_shared']:.4f}")
        else:
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

    # --- per-subject aggregation at the peak layer (W6) ---
    # ridge_encode fits each voxel column independently, so pooling voxels across subjects
    # never mixes subject signal in the *fit*; pooled vs per-subject differs ONLY in how
    # voxels are weighted when averaged (pooled weights by voxel count; per-subject gives
    # each subject equal weight). We therefore re-aggregate the existing per-voxel
    # unique-hidden by subject — no refit needed — to show the result is not a pooling
    # artifact and the magnitude is stable across participants.
    per_subject = None
    if ds.voxel_subject is not None:
        uh = np.clip(vps[peak].unique_hidden, 0, None)
        vs = np.asarray(ds.voxel_subject)
        subj_means = {}
        for s in (ds.subject_ids or sorted(set(vs.tolist()))):
            m = vs == s
            if m.any():
                subj_means[str(s)] = float(np.nanmean(uh[m]))
        if subj_means:
            vals = np.array(list(subj_means.values()))
            per_subject = {
                "peak_layer": int(peak),
                "subject_unique_hidden": subj_means,
                "mean_across_subjects": float(vals.mean()),
                "sem_across_subjects": float(vals.std(ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1 else 0.0,
                "n_subjects": int(len(vals)),
                "pooled_unique_hidden": float(np.nanmean(uh)),
            }
            print(f"[stage4] per-subject (n={per_subject['n_subjects']}): "
                  f"mean={per_subject['mean_across_subjects']:.4f} "
                  f"±{per_subject['sem_across_subjects']:.4f} (SEM) vs "
                  f"pooled={per_subject['pooled_unique_hidden']:.4f}")

    save_json(out_dir / "varpart.json", {
        "model_key": spec.key,
        "is_synthetic": ds.is_synthetic,
        "peak_layer": peak,
        "confound_control": confounds is not None,
        "confound_names": confound_names,
        "per_layer": results,
        "per_layer_surprisal_only": results_surp_only,
        "per_subject": per_subject,
    })
    print(f"[stage4] wrote {out_dir/'varpart.json'}")


if __name__ == "__main__":
    main()
