"""Stage 8: render publication figures into paper/figures/ from data/derived/.

Figures:
  fig_layercurve  - normalized predictivity vs relative depth (all models)
  fig_varpart     - variance-partitioning stackplot (unique-H / shared / unique-S) per model
  fig_sweep       - peak unique-hidden per model (with 95% CI) + size axis
  fig_rsa         - RSA vs relative depth (all models)

Usage:
    python scripts/08_figures.py                 # all models found
    python scripts/08_figures.py --models gpt2,qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmbrain.config import REPO_ROOT, load_config
from llmbrain.utils.io import load_json

# Approx parameter counts (millions) for the size axis.
PARAMS_M = {
    "gpt2": 124, "gpt2-medium": 355, "opt-125m": 125,
    "pythia-160m": 160, "pythia-410m": 410,
    "qwen2.5-0.5b": 500, "qwen2.5-0.5b-instruct": 500,
}


def _layers(per_layer: dict) -> list[int]:
    return sorted(int(k) for k in per_layer)


def _rel_depth(layers: list[int]) -> np.ndarray:
    layers = np.asarray(layers, dtype=float)
    return layers / layers.max() if layers.max() > 0 else layers


def fig_layercurve(models, derived, out):
    plt.figure(figsize=(6, 4))
    for m in models:
        p = derived / m / "encoding.json"
        if not p.exists():
            continue
        e = load_json(p)
        pl = e["per_layer"]
        layers = _layers(pl)
        key = "normalized_r" if e.get("has_noise_ceiling") else "mean_r"
        y = [pl[str(l)].get(key) or pl[str(l)].get("mean_r") for l in layers]
        plt.plot(_rel_depth(layers), y, "o-", ms=3, label=m)
    plt.axhline(1.0, ls="--", c="gray", lw=0.8)
    plt.xlabel("relative layer depth")
    plt.ylabel("normalized predictivity (r / noise ceiling)")
    plt.title("Layer-depth alignment")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_layercurve.pdf")
    plt.close()


def fig_varpart(models, derived, out):
    # one panel per model (up to 6)
    models = [m for m in models if (derived / m / "varpart.json").exists()]
    n = len(models)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, m in zip(axes.flat, models):
        v = load_json(derived / m / "varpart.json")["per_layer"]
        layers = _layers(v)
        uh = [v[str(l)]["mean_unique_hidden"] for l in layers]
        sh = [v[str(l)]["mean_shared"] for l in layers]
        us = [v[str(l)]["mean_unique_surprisal"] for l in layers]
        ax.stackplot(_rel_depth(layers), uh, sh, us,
                     labels=["unique hidden", "shared", "unique surprisal"],
                     colors=["#2c7fb8", "#7fcdbb", "#edf8b1"])
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("rel. depth")
        ax.set_ylabel("R$^2$")
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Variance partitioning: hidden vs surprisal")
    fig.tight_layout()
    fig.savefig(out / "fig_varpart.pdf")
    plt.close(fig)


def fig_sweep(models, derived, out):
    rows = []
    for m in models:
        p = derived / m / "varpart.json"
        if not p.exists():
            continue
        vj = load_json(p)
        peak = vj.get("peak_layer")
        pl = vj["per_layer"]
        if peak is None:
            peak = max(pl, key=lambda l: pl[l]["mean_unique_hidden"])
        d = pl[str(peak)]
        ci = d.get("ci_unique_hidden")
        rows.append((m, PARAMS_M.get(m, np.nan), d["mean_unique_hidden"],
                     ci["lo"] if ci else None, ci["hi"] if ci else None))
    if not rows:
        return
    rows.sort(key=lambda r: (r[1], r[0]))
    names = [r[0] for r in rows]
    vals = np.array([r[2] for r in rows])
    lo = np.array([r[3] if r[3] is not None else r[2] for r in rows])
    hi = np.array([r[4] if r[4] is not None else r[2] for r in rows])
    err = np.vstack([vals - lo, hi - vals])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, vals, yerr=err, capsize=3, color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("peak unique-hidden R$^2$ (95% CI)")
    ax.set_title("Unique-beyond-surprisal contribution across models")
    fig.tight_layout()
    fig.savefig(out / "fig_sweep.pdf")
    plt.close(fig)


def fig_rsa(models, derived, out):
    plt.figure(figsize=(6, 4))
    any_plotted = False
    for m in models:
        p = derived / m / "rsa.json"
        if not p.exists():
            continue
        r = load_json(p)["per_layer"]
        layers = _layers(r)
        plt.plot(_rel_depth(layers), [r[str(l)] for l in layers], "o-", ms=3, label=m)
        any_plotted = True
    if not any_plotted:
        return
    plt.axhline(0, ls="--", c="gray", lw=0.8)
    plt.xlabel("relative layer depth")
    plt.ylabel("RSA (Spearman $\\rho$)")
    plt.title("Representational geometry alignment")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_rsa.pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    derived = cfg.derived_dir()
    out = REPO_ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    models = (args.models.split(",") if args.models
              else sorted(p.name for p in derived.iterdir() if p.is_dir()))
    print(f"[stage8] models: {models}")
    fig_layercurve(models, derived, out)
    fig_varpart(models, derived, out)
    fig_sweep(models, derived, out)
    fig_rsa(models, derived, out)
    print(f"[stage8] wrote figures to {out}")


if __name__ == "__main__":
    main()
