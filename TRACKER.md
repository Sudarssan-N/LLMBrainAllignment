# Experiment Tracker

One row per experiment run. Fill in as runs complete. Keep results reproducible: every row
should map to a config + git commit so it can be re-run from a fresh Colab clone.

**Status legend:** ⬜ planned · 🟡 running · ✅ done · ❌ failed/abandoned · 🔁 needs rerun

---

## Active experiment matrix

The full design is the cross product of {model} × {layer} × {analysis}, evaluated on
Pereira2018 (per ROI / per voxel, normalized by noise ceiling).

**Models**

| Key | HF id | Params | Family | Type | Role |
|-----|-------|--------|--------|------|------|
| gpt2 | `gpt2` | 124M | GPT-2 | base | baseline / reproduction |
| gpt2-medium | `gpt2-medium` | 355M | GPT-2 | base | size scaling |
| pythia-160m | `EleutherAI/pythia-160m` | 160M | Pythia | base | scaling family (small) |
| pythia-410m | `EleutherAI/pythia-410m` | 410M | Pythia | base | scaling family (large) |
| opt-125m | `facebook/opt-125m` | 125M | OPT | base | cross-family check |
| pythia-410m-sft | _tbd_ | 410M | Pythia | instruct | base-vs-instruct contrast |

**Analyses**

- `predict` — layer-wise ridge encoding, normalized predictivity (reproduction).
- `varpart` — variance partitioning: unique-hidden / unique-surprisal / shared R².
- `matched` — surprisal-matched-subset predictivity (structure sensitivity check).
- `rsa` — RDM-to-RDM correlation (model layer vs brain ROI).

---

## Runs

| ID | Date | Model | Layers | Analysis | Dataset | Config / commit | Status | Headline result | Notes |
|----|------|-------|--------|----------|---------|-----------------|--------|-----------------|-------|
| E000 | 2026-06-16 | — | — | scaffold | synthetic | `config/default.yaml` | ✅ | full pipeline runs; 9/9 tests pass; varpart shows unique_hidden≫unique_surprisal on synthetic | smoke-test passes before real data |
| E001 | 2026-06-17 | gpt2 | all (0–12) | predict | Pereira2018 exp384 (384 sent × 12155 vox) | `experiments:[384]`, pooling=mean, no noise ceiling | 🟡 | raw mean_r flat ~0.20–0.22, best=L12 (0.221); L0 embeddings≈0.215 | flat curve ⇒ mean-pooling washout; needs last-token + noise ceiling. Re-run as E001b w/ `--pooling last` |
| E002 | 2026-06-17 | gpt2 | all (0–12) | varpart | Pereira2018 exp384 | same as E001 | 🟡 | **unique_hidden≈0.04 ≫ unique_surprisal≈0.0006** (~70×); shared≈0.0015 | core result direction confirmed even at baseline; magnitudes low pending normalization |
| E003 | | pythia-160m / 410m | all | predict+varpart | Pereira2018 | | ⬜ | size × residual alignment | "bigger ≠ more brain-like" |
| E004 | | base vs instruct | best | varpart | Pereira2018 | | ⬜ | instruction-tuning effect | novelty axis |
| E005 | | gpt2 | best | matched | Pereira2018 | | ⬜ | structure survives surprisal match | validation |
| E006 | | gpt2 | all | rsa | Pereira2018 | | ⬜ | geometry alignment | RSA cross-check |

---

## Metrics dictionary

- **Normalized predictivity** = mean voxel Pearson r (held-out) ÷ noise ceiling.
- **R²_unique(hidden)** = R²(hidden ∪ surprisal) − R²(surprisal).
- **R²_unique(surprisal)** = R²(hidden ∪ surprisal) − R²(hidden).
- **R²_shared** = R²(hidden) + R²(surprisal) − R²(hidden ∪ surprisal).
- **RSA score** = Spearman ρ between model-layer RDM and brain-ROI RDM.

Report per-ROI and per-layer with cross-validation folds + CIs (bootstrap over sentences).
