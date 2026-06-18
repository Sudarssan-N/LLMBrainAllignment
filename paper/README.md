# Paper — Disentangling Surprisal from Structure in LLM–Brain Alignment

NeurIPS-format draft. Venue is not final (we may retarget to CCN / NeurIPS UniReps / CMCL);
the format is easy to swap later.

## Build

The official NeurIPS 2024 style file (`neurips_2024.sty`) is **bundled in this folder**, so the
paper compiles with no manual download.

- **Overleaf:** upload the whole `paper/` folder (including `neurips_2024.sty` and `figures/`)
  and compile `main.tex`. The `[preprint]` option shows author names while drafting.
- **Local:**
  ```bash
  make            # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```

## Status / conventions

- The draft is finalized to the clean re-run (stabilized LOO-GCV ridge + voxel-bootstrap CIs +
  Stage-7 paired test). Table 1 and Sections 4.1–4.5 reflect E003b / E004 / E006.
- `\prelim{...}` (blue) / `\todo{...}` (red) macros remain *defined* for future drafting but are
  no longer used in the body. Strip the definitions before camera-ready if desired.

## Figures

Rendered into `figures/` by `python scripts/08_figures.py` from `data/derived/`:
- `fig_layercurve.pdf` — normalized predictivity vs layer (per model).
- `fig_varpart.pdf` — variance-partitioning stackplot (unique-H / shared / unique-S).
- `fig_sweep.pdf` — peak unique-hidden vs model size (with 95% CIs).
- `fig_rsa.pdf` — RSA (Spearman ρ) vs layer (per model).

All four are wired into `main.tex` via `\includegraphics{figures/...}`.
