"""Stage 8: render publication figures into paper/figures/ from data/derived/.

Regenerates the five figures used in the paper as PDFs (sharper than PNG in LaTeX):
  fig_scaling           - (L) normalized predictivity vs depth, all models;
                          (R) peak unique-hidden vs params, Pythia line + other families
  fig_confound_survival - unique-hidden: full confound control vs surprisal-only (focus model)
  fig_per_subject       - per-subject vs pooled peak unique-hidden (focus model)
  fig_instruct_pairs    - base vs instruct peak unique-hidden, two Qwen pairs
  fig_geometry_all      - RSA vs relative depth, all models

These read the confound-controlled artifacts written by stage 4 (per_layer is the controlled
partition; per_layer_surprisal_only and per_subject are also stored there).

Usage:
    python scripts/08_figures.py                 # all models found
    python scripts/08_figures.py --focus gpt2    # which model drives the single-model panels
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
    "pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410,
    "pythia-1b": 1000, "pythia-1.4b": 1400,
    "qwen2.5-0.5b": 500, "qwen2.5-0.5b-instruct": 500,
    "qwen2.5-1.5b": 1500, "qwen2.5-1.5b-instruct": 1500,
}

# Pythia suite in scale order: the confound-free scaling axis (same data + step order).
PYTHIA_ORDER = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]

# base/instruct pairs for the instruction-tuning panel.
INSTRUCT_PAIRS = [
    ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct"),
    ("qwen2.5-1.5b", "qwen2.5-1.5b-instruct"),
]


def _layers(per_layer: dict) -> list[int]:
    return sorted(int(k) for k in per_layer)


def _rel_depth(layers: list[int]) -> np.ndarray:
    layers = np.asarray(layers, dtype=float)
    return layers / layers.max() if layers.max() > 0 else layers


def _peak_uh(vj: dict) -> float | None:
    pl = vj["per_layer"]
    peak = vj.get("peak_layer")
    if peak is None:
        peak = max(pl, key=lambda l: pl[l]["mean_unique_hidden"])
    return pl[str(peak)]["mean_unique_hidden"]


def fig_scaling(models, derived, out):
    """Two panels: layer-depth predictivity (all models) + Pythia scaling line."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in models:
        p = derived / m / "encoding.json"
        if not p.exists():
            continue
        e = load_json(p)
        pl = e["per_layer"]
        layers = _layers(pl)
        key = "normalized_r" if e.get("has_noise_ceiling") else "mean_r"
        y = [pl[str(l)].get(key) or pl[str(l)].get("mean_r") for l in layers]
        ax[0].plot(_rel_depth(layers), y, "o-", ms=3, label=m)
    ax[0].axhline(1.0, ls="--", c="gray", lw=0.8)
    ax[0].set(xlabel="relative depth", ylabel="normalized predictivity",
              title="Layer-depth alignment")
    ax[0].legend(fontsize=7)

    def peak(m):
        p = derived / m / "varpart.json"
        return _peak_uh(load_json(p)) if p.exists() else None

    px = [PARAMS_M[m] for m in PYTHIA_ORDER if peak(m) is not None]
    py = [peak(m) for m in PYTHIA_ORDER if peak(m) is not None]
    if px:
        ax[1].plot(px, py, "o-", c="#2c7fb8", lw=2, ms=6,
                   label="Pythia (clean scaling axis)", zorder=3)
    for m in models:
        if m in PYTHIA_ORDER:
            continue
        y = peak(m)
        if y is None:
            continue
        ax[1].scatter(PARAMS_M.get(m, np.nan), y, c="gray", alpha=0.6, zorder=2)
        ax[1].annotate(m, (PARAMS_M.get(m, np.nan), y), fontsize=6,
                       xytext=(3, 3), textcoords="offset points", color="gray")
    ax[1].set(xscale="log", xlabel="params (M, log)", ylabel="peak unique-hidden R$^2$",
              title="Scaling: Pythia suite (line) vs other families (points)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_scaling.pdf")
    plt.close(fig)


def fig_confound_survival(focus, derived, out):
    """Unique-hidden under full confound control vs surprisal-only (focus model)."""
    p = derived / focus / "varpart.json"
    if not p.exists():
        return
    vj = load_json(p)
    v = vj["per_layer"]
    so = vj.get("per_layer_surprisal_only") or {}
    layers = _layers(v)
    plt.figure(figsize=(7, 4))
    plt.plot(layers, [v[str(l)]["mean_unique_hidden"] for l in layers], "o-",
             c="#2c7fb8", label="controlled (surprisal+confounds)")
    if so:
        plt.plot(layers, [so[str(l)]["mean_unique_hidden"] for l in layers], "s--",
                 c="#de2d26", label="surprisal only")
    plt.xlabel("layer")
    plt.ylabel("unique-hidden R$^2$")
    plt.title(f"{focus}: unique-hidden survival under confound control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_confound_survival.pdf")
    plt.close()


def fig_per_subject(focus, derived, out):
    """Per-subject vs pooled peak unique-hidden (focus model)."""
    p = derived / focus / "varpart.json"
    if not p.exists():
        return
    ps = load_json(p).get("per_subject")
    if not ps:
        print(f"[stage8] no per_subject info for {focus}; skipping fig_per_subject")
        return
    sm = ps["subject_unique_hidden"]
    names = list(sm)
    vals = [sm[n] for n in names]
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(vals)), vals, c="#2c7fb8", zorder=3, label="per subject")
    plt.axhline(ps["mean_across_subjects"], c="#2c7fb8", ls="-",
                label=f"across-subject mean={ps['mean_across_subjects']:.3f}")
    plt.axhline(ps["pooled_unique_hidden"], c="#de2d26", ls="--",
                label=f"pooled={ps['pooled_unique_hidden']:.3f}")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
    plt.ylabel("peak unique-hidden R$^2$")
    plt.title(f"{focus}: per-subject vs pooled")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_per_subject.pdf")
    plt.close()


def fig_instruct_pairs(derived, out):
    """Base vs instruct peak unique-hidden for each configured pair."""
    pairs = [(b, i) for b, i in INSTRUCT_PAIRS
             if (derived / b / "varpart.json").exists()
             and (derived / i / "varpart.json").exists()]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4), squeeze=False)
    for ax, (base, inst) in zip(axes[0], pairs):
        vals, errs = [], [[], []]
        for m in (base, inst):
            vj = load_json(derived / m / "varpart.json")
            peak = vj["peak_layer"]
            d = vj["per_layer"][str(peak)]
            ci = d.get("ci_unique_hidden")
            vals.append(d["mean_unique_hidden"])
            errs[0].append(d["mean_unique_hidden"] - ci["lo"] if ci else 0)
            errs[1].append(ci["hi"] - d["mean_unique_hidden"] if ci else 0)
        ax.bar(["base", "instruct"], vals, yerr=errs, capsize=4,
               color=["#2c7fb8", "#de2d26"])
        ax.set_ylabel("peak unique-hidden R$^2$ (95% CI)")
        ax.set_title(base.replace("-instruct", ""))
    fig.tight_layout()
    fig.savefig(out / "fig_instruct_pairs.pdf")
    plt.close(fig)


def fig_geometry_all(models, derived, out):
    plt.figure(figsize=(6.5, 4))
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
    plt.xlabel("relative depth")
    plt.ylabel("RSA (Spearman $\\rho$)")
    plt.title("Geometry alignment")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "fig_geometry_all.pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--focus", default="gpt2",
                    help="model that drives the single-model panels (survival, per-subject)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    derived = cfg.derived_dir()
    out = REPO_ROOT / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    models = (args.models.split(",") if args.models
              else sorted(p.name for p in derived.iterdir() if p.is_dir()))
    print(f"[stage8] models: {models} | focus={args.focus}")
    fig_scaling(models, derived, out)
    fig_confound_survival(args.focus, derived, out)
    fig_per_subject(args.focus, derived, out)
    fig_instruct_pairs(derived, out)
    fig_geometry_all(models, derived, out)
    print(f"[stage8] wrote PDF figures to {out}")


if __name__ == "__main__":
    main()
