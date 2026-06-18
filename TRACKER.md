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
| E001 | 2026-06-17 | gpt2 | all (0–12) | predict | Pereira2018 exp384 (384 sent × 12155 vox) | `experiments:[384]`, pooling=mean, no ceiling | ❌ | raw mean_r flat ~0.20–0.22; L0≈best | superseded: mean-pooling washout. See E001b |
| E001b | 2026-06-17 | gpt2 | all (0–12) | predict | Pereira2018 exp384 | pooling=**last**, cross-subject noise ceiling (9 subj, mean ceiling 0.219, n=10664 vox) | ✅ | **norm_r 0.14→1.02 monotonic, ~100% ceiling at L12**; mean_r 0.032→0.225 | reproduces Schrimpf "~100% of ceiling"; monotonic peak-at-last expected for 124M |
| E002 | 2026-06-17 | gpt2 | all (0–12) | varpart | Pereira2018 exp384 | pooling=last | ✅ | **unique_hidden→0.039 @L12 ≫ unique_surprisal ~0.0005 (~70×); gap grows with depth**; shared~0.0016 | CORE RESULT. (L3 unique_surprisal=0.0048 outlier — recheck fold/alpha) |
| E003 | 2026-06-17 | gpt2/opt-125m/pythia-160m/pythia-410m/qwen0.5b | all | predict+varpart | Pereira2018 exp384 | pooling=last, cross-subj ceiling | 🔁→✅ | (superseded by E003b) | original numbers used unstable 80/20-split ridge; p410m unique(H) spuriously low (.026) |
| E003b | 2026-06-19 | gpt2/opt-125m/pythia-160m/pythia-410m/qwen0.5b | all | predict+varpart | Pereira2018 exp384 | pooling=last, **stabilized LOO-GCV ridge** + 95% voxel-bootstrap CI | ✅ | **peak unique(H) RISES with scale within Pythia (.039→.057, 160M→410M); qwen0.5b highest (.066)**. norm_r: qwen 1.13 > p410m 1.06 > opt 1.01 > gpt2 0.98 > p160m 0.75. H:S ratio 16–19× at peak | **CORRECTS E003**: p410m artifact gone (.026→.057, ratio→19×). Cross-family not monotonic (gpt2/opt > p160m): architecture/training matter. CIs: gpt2 [.051,.053], opt [.058,.060], p160m [.038,.040], p410m [.056,.058], qwen [.065,.067] |
| E004 | 2026-06-19 | qwen2.5-0.5b base vs instruct | all | predict+varpart | Pereira2018 exp384 | pooling=last, stabilized ridge + CI | ✅ | **instruct SIGNIFICANTLY LOWER**: norm_r 1.103 vs 1.130; peak unique(H) 0.0605 [.0597,.0614] vs 0.0661 [.0652,.0670] — **non-overlapping 95% CIs** (~8.5% rel reduction). **Stage-7 paired test (n=12137 voxels): mean diff +0.0068 [+0.0065,+0.0071], SIGNIFICANT** | instruction tuning reliably reduces alignment+unique-hidden; per-voxel paired CI strictly positive |
| E005 | | gpt2 | best | matched | Pereira2018 | | ⬜ | structure survives surprisal match | validation |
| E006 | 2026-06-19 | gpt2/qwen2.5-0.5b | all | predict+varpart | Pereira2018 **exp243** (243 sent × 8031 vox, 6 subj) | pooling=last, stabilized ridge + CI, ceiling mean 0.272 | ✅ | **core result replicates**: gpt2 peak unique(H) 0.0488 [.0476,.0500] @L11; qwen 0.0457 [.0444,.0471] @L22; hidden≫surprisal | independent-experiment robustness for §4.2. Size/instruct contrasts on 243 await full sweep |
| E007 | | gpt2 | best | rsa | Pereira2018 | | ⬜ | geometry alignment | RSA cross-check (RSA already emitted by Stage 5 in sweep; formalize) |

---

## Metrics dictionary

- **Normalized predictivity** = mean voxel Pearson r (held-out) ÷ noise ceiling.
- **R²_unique(hidden)** = R²(hidden ∪ surprisal) − R²(surprisal).
- **R²_unique(surprisal)** = R²(hidden ∪ surprisal) − R²(hidden).
- **R²_shared** = R²(hidden) + R²(surprisal) − R²(hidden ∪ surprisal).
- **RSA score** = Spearman ρ between model-layer RDM and brain-ROI RDM.

Report per-ROI and per-layer with cross-validation folds + CIs (bootstrap over sentences).
