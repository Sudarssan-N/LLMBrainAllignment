# Disentangling Statistical Surprisal from Hierarchical Structure in LLM–Brain Alignment

A probing study of **why** LLM hidden states predict human language-network fMRI responses.
Core question:

> **How much of LLM–brain alignment on Pereira2018 is *uniquely* explained by hidden
> representations beyond next-word surprisal — and how does that residual depend on
> model size, layer depth, and instruction tuning?**

This repo reproduces the Brain-Score linear-predictivity pipeline on Pereira (2018) and then
adds the novel contributions:

1. **Surprisal-controlled encoding** — joint variance partitioning of voxel responses into
   variance *unique* to hidden states, *unique* to surprisal, and *shared*.
2. **Surprisal-matched stimulus subsets** — test whether residual alignment still tracks
   syntactic/compositional structure once surprisal is held fixed.
3. **Model-family comparisons** — base vs instruction-tuned, small vs large; layer-depth
   curves and the intermediate-layer-advantage / "bigger ≠ more brain-like" hypotheses.
4. **RSA** — representational geometry comparison (RDM-to-RDM) alongside linear predictivity.

See `LLMBrainAllignment.md` (proposal + critiques) and
`llm_brain_alignment_novelty_report.md` (novelty assessment) for the full framing.

---

## Repository layout

```
.
├── config/default.yaml          # models, layers, paths, CV settings
├── src/llmbrain/
│   ├── config.py                # config loading (dataclasses)
│   ├── data/
│   │   ├── pereira.py           # load Pereira2018 fMRI + sentence stimuli
│   │   └── synthetic.py         # synthetic data for end-to-end pipeline tests
│   ├── models/
│   │   ├── activations.py       # layer-wise activation extraction (HF + hooks)
│   │   └── surprisal.py         # per-token / per-sentence surprisal
│   ├── encoding/
│   │   ├── ridge.py             # cross-validated ridge encoding + noise ceiling
│   │   ├── variance_partitioning.py  # commonality analysis (unique/shared R²)
│   │   └── rsa.py               # representational similarity analysis
│   ├── analysis/
│   │   └── matched_subsets.py   # surprisal-matched stimulus subsets
│   └── utils/
│       ├── pooling.py           # subword-token → word/sentence pooling
│       └── io.py                # cached save/load of arrays
├── scripts/                     # CLI stages, each writes artifacts to data/derived/
│   ├── 01_extract_activations.py
│   ├── 02_compute_surprisal.py
│   ├── 03_fit_encoding.py
│   ├── 04_variance_partitioning.py
│   └── 05_rsa.py
├── notebooks/run_experiments.ipynb   # Colab driver (git clone → run → plot)
├── tests/test_pipeline.py            # runs the full pipeline on synthetic data
├── LOG.md                            # chronological research log
└── TRACKER.md                        # experiment tracker (one row per run)
```

## Workflow (Colab)

Experiments run in Colab via a fresh `git clone`. The notebook drives everything:

```bash
!git clone https://github.com/<you>/LLM-BrainAlignment.git
%cd LLM-BrainAlignment
!pip install -r requirements.txt
```

Then run the stages (notebook cells call the `scripts/`). Code changes are pushed to
`origin/master` so the next Colab clone picks them up.

## Quick local smoke test (no real data, no large models)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # full pipeline on synthetic data
python scripts/03_fit_encoding.py --synthetic   # encoding on toy data
```

## Pipeline stages

| Stage | Script | Output |
|-------|--------|--------|
| 1 | `01_extract_activations.py` | `data/derived/<model>/activations.npz` (layer × sentence × hidden) |
| 2 | `02_compute_surprisal.py` | `data/derived/<model>/surprisal.npz` (per-sentence surprisal features) |
| 3 | `03_fit_encoding.py` | `data/derived/<model>/encoding_<layer>.json` (per-voxel/per-ROI predictivity) |
| 4 | `04_variance_partitioning.py` | `data/derived/<model>/varpart_<layer>.json` (unique/shared R²) |
| 5 | `05_rsa.py` | `data/derived/<model>/rsa_<layer>.json` (RDM correlations) |

## Data

Pereira et al. 2018 (Nat. Commun. 9:963) language-network fMRI. Accessed via the
`brain-score/language` harness when available; `src/llmbrain/data/pereira.py` documents
the loader and falls back to a clearly-labelled synthetic assembly so the pipeline is
runnable before data access is wired up.

## Status

Initial scaffold — see `LOG.md` and `TRACKER.md`.
