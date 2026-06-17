# Paper — Disentangling Surprisal from Structure in LLM–Brain Alignment

NeurIPS-format draft. Venue is not final (we may retarget to CCN / NeurIPS UniReps / CMCL);
the format is easy to swap later.

## Build

1. Download the official NeurIPS 2024 style file and place it here:
   - `neurips_2024.sty` from https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
   - (The `[preprint]` option in `main.tex` shows author names while drafting.)
2. Build:
   ```bash
   make            # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
   ```

If you don't have the style file yet, you can temporarily switch the documentclass line to
plain `article` to preview content (formatting will differ).

## Status / conventions

- `\prelim{...}` (blue) marks **preliminary numbers** to refresh after the
  stabilized-ridge + CI re-run.
- `\todo{...}` (red) marks gaps to fill. Strip both macros before submission.
- Numbers currently reflect the first full sweep (E001b–E004). After re-running the sweep
  with the RidgeCV fix + Stage 7 paired test, update Table 1 and Sections 4.2–4.4, and fill
  the Pythia-410M `unique(S)` cell (was contaminated by the old alpha-selection artifact).

## Figures (to add)

Generate from `data/derived/` after a sweep:
- `fig_layercurve.pdf` — normalized predictivity vs layer (per model).
- `fig_varpart.pdf` — variance-partitioning stackplot (unique-H / shared / unique-S).
- `fig_sweep.pdf` — peak unique-hidden vs model size + base-vs-instruct.

A plotting cell exists in `notebooks/run_experiments.ipynb`; we will add a `scripts/08_figures.py`
to render these to `paper/figures/` directly.
