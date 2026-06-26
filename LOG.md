# Research Log — LLM–Brain Alignment (Surprisal vs Structure)

Chronological notes. Newest entries at the top. Keep entries short: what was done,
what was decided, what's blocked, what's next. Link runs to `TRACKER.md` IDs.

---

## 2026-06-27 — Second review round: responsive edits (no new runs)

Implemented the reviewer's new asks in `paper/main.tex` + `references.bib`. No experiments
re-run; all numbers traced to existing TRACKER runs.

- **W1 (closest prior art now cited).** Added `lepori2026` (Interpreting Brain Responses with
  Sparse Features / Augmented Sparse Encoding Models, arXiv:2606.06857 — verified real, authors
  Lepori/Kay/Tuckute) and `zhao2026` (Beyond next-word prediction…, bioRxiv 10.64898/2026.05.15.725490
  — verified, authors Zhao & Brennan). New Related-Work paragraph states our 3-point delta
  (Feghhi confounds in N, Pythia scaling axis, two-scale IT).
- **W2 (Table-1-vs-prose tension on IT).** §4.5 now states plainly that norm r ALSO dips for both
  instruct pairs (1.13→1.10, 1.22→1.19), so we do not claim total predictivity rises in our data;
  reframed the Aw et al. discrepancy as dataset/metric scope, not a within-pipeline contradiction.
- **W3 (43×/16× ratios).** Explained as a near-floor unique-nuisance denominator at those Pythia
  peak layers (not a large numerator); foreground absolute unique(H)/unique(N)/% retained. Caption
  note added.
- **W4 (raw r).** Removed the unbacked "report raw r alongside" promise. No per-model raw-r summary
  exists in our artifacts — the only recoverable E001b output is normalized_r 0.9774 @ L12 (≈ the
  0.98 in Table 1); the earlier TRACKER note of mean_r 0.225 is not reproducible from saved output,
  so I dropped the "raw r ≈0.22" anchor from §4.1. Now anchor cross-model comparison on the absolute
  unique(H) R² column; the pipeline recomputes per-voxel raw r on re-run. *Open:* drop in a literal
  raw-r column once per-model mean raw r is pulled from a fresh Colab run.
- **W5.** Plain statement in Discussion that "survives confound control" = THESE controls, not
  confound-completeness.
- **Misc.** Fixed `superhuman2025` author (Anonymous → Oh & Linzen). Added a Methods hyperparameter
  table (`tab:hyper`: α-grid, LOO-GCV, 5-fold, ceiling PCA=100, min-ceiling 0.1, 1000-resample
  bootstrap, seed 0). Reinstated 243-set numbers in Limitation (i) from E006.

Static checks pass (all \cite keys resolve, labels/refs/envs balanced); no local TeX toolchain,
so compile on Overleaf.

---

## 2026-06-19 — Clean re-run with stabilized ridge + CIs; paper finalized to current data

Re-ran the full sweep (E003b, E004) and the 243-set robustness (E006) in Colab with the
LOO-GCV ridge + voxel-bootstrap CIs. Three things changed, one of them a headline:

- **The "bigger ≠ more uniquely structured" claim FLIPS.** It was largely an artifact of the
  unstable ridge. Clean peak unique(H): gpt2 .052, opt .059, p160m .039, **p410m .057** (was a
  contaminated .026), qwen .066. Within the controlled Pythia family unique(H) now RISES with
  scale (.039→.057, 160M→410M, +45%) and the largest model (qwen) is highest. Across families
  it's not monotonic (gpt2/opt > p160m) → architecture/training matter, not params alone.
- **p410m artifact gone**: unique(S)@peak .003, ratio 19× (was 2.7×).
- **Instruction tuning now SIGNIFICANT**: qwen base unique(H) 0.0661 [.0652,.0670] vs instruct
  0.0605 [.0597,.0614] — non-overlapping CIs, ~8.5% rel reduction (was a vague ~4–5%).
- **Ratio reframed**: ~17× at the peak-hidden layer (where unique(S) is largest), up to ~70× at
  mid-depth. Abstract no longer claims a flat "50–70×".
- **E006 robustness**: core hidden≫surprisal replicates on the independent 243-set (gpt2 .049,
  qwen .046). Ceiling mean 0.272 (243) / 0.269 (384).

Folded all of this into `paper/main.tex`: abstract, contributions, §4.1–4.4, Table 1, limitations,
conclusion; stripped all `\prelim`/`\todo` uses (macros kept defined); set ceiling = 0.269.

Stage 7 paired per-voxel test done: mean diff +0.0068 [+0.0065,+0.0071] over 12,137 voxels,
strictly positive → instruct effect significant. Folded into §4.4.

**NEXT:** regenerate figures (Stage 8) from the clean artifacts; decide co-authors/affiliations
for the author block.

## 2026-06-18 — Ridge stabilization, CIs, and paper draft

- **Fixed the varpart artifact**: replaced the single 80/20 alpha split with deterministic
  `RidgeCV` LOO-GCV; this removes the spurious unique-surprisal spikes (incl. p410m's
  contaminated ratio). Added `encoding/stats.py` (bootstrap CI over voxels + paired diff),
  Stage 4 now emits 95% CI on peak unique-hidden and saves per-voxel arrays, new Stage 7
  paired base-vs-instruct test, aggregate uses the Stage-4 peak layer + shows CI.
- **Started the paper**: `paper/main.tex` in NeurIPS 2024 format (preprint), `references.bib`,
  Makefile, README. Numbers marked `\prelim{}`/`\todo{}` pending the re-run.

**NEXT (re-run in Colab with fixed code):** re-run the sweep (Section 5), then per pair run
`python scripts/07_compare.py --a qwen2.5-0.5b --b qwen2.5-0.5b-instruct`; refresh Table 1 +
Sections 4.2–4.4 and fill the p410m unique(S) cell with the now-stable number.

## 2026-06-17 (cont. 4) — Full sweep done (E003 size, E004 instruct)

6 models, last-token, noise-ceiling-normalized. Findings:
- **Core thesis holds across all models**: unique_hidden ≫ unique_surprisal (~50–70× at
  clean layers).
- **Size (E003): bigger ≠ more brain-like in unique-hidden.** peak_unique_hidden is NOT
  monotonic in params (gpt2 124M=.039, pythia-160m=.029, pythia-410m=.026, qwen-0.5b=.049).
  Raw norm_r does rise for qwen (1.14). So predictivity ↑ with size/quality but *unique
  representational* contribution does not.
- **Instruction tuning (E004): marginally REDUCES alignment + unique-hidden** (qwen base
  norm_r 1.145 / uh 0.0489 vs instruct 1.127 / 0.0467; both peak L23). ~4–5% relative.

**Two things to fix before these are publication-final:**
1. **Variance-partitioning alpha instability** → deterministic unique_surprisal spikes on
   isolated layers (gpt2 L3, p410m L19/21, opt L10/11, qwen L0/4/20/24 ≈ 0.005–0.018 vs
   ~0.0005 elsewhere). Root cause: alpha picked by a single 80/20 split per ridge call, so
   joint vs hidden models sometimes pick different alphas. Contaminates the p410m headline
   ratio (shows 2.7× instead of ~50×). Fix: stabilize alpha selection (k-fold or shared alpha).
2. **No CIs yet.** The E004 instruction-tuning effect (~5%) needs bootstrap CIs over
   sentences before it can be claimed. Also add experiment 243 for robustness.

## 2026-06-17 (cont. 3) — Clean baseline achieved (E001b/E002)

last-token pooling + cross-subject noise ceiling gave the publishable baseline:
- **GPT-2 reaches ~100% of noise ceiling at L12** (norm_r 0.14→1.02, monotonic). Reproduces
  Schrimpf "~100% of ceiling". Monotonic peak-at-last is expected for a 124M model.
- **Variance partitioning (core result): unique_hidden→0.039 @L12 ≫ unique_surprisal ~0.0005
  (~70×), and the gap grows with depth.** This is the paper's contribution, on real data.
- Anomaly to revisit: L3 unique_surprisal=0.0048 (≈10× neighbours) — likely a fold/alpha
  quirk; check CV stability.

**Set up the model sweep** for the novelty axes: added Qwen2.5-0.5B base/instruct pair to
config (open, ungated) for instruction-tuning; `scripts/06_aggregate.py` builds a cross-model
comparison table + summary.csv; notebook sweep cell runs Pythia-160m/410m, OPT-125m, Qwen
pair with `--pooling last` then aggregates. Next: run sweep (E003 size, E004 instruct).

## 2026-06-17 (cont. 2) — Noise ceiling + pooling override

**Built the cross-subject (brain-to-brain) noise ceiling** (`encoding/noise.py`). Pereira
voxels are subject-specific with no within-subject repeats, so split-half is undefined;
instead, per held-out subject, predict their voxels from the *other* subjects' responses to
the same sentences (PCA-reduced CV ridge). Normalized predictivity = model_r / ceiling over
voxels with ceiling ≥ 0.1. Stage 3 now auto-selects: cross-subject for real Pereira
(`voxel_subject` set), split-half for the synthetic fixture. Added `voxel_subject` to
`BrainDataset` + loader. Tests added (ceiling positive on shared-latent disjoint voxels).

**Pooling washout diagnosis:** baseline used mean-pooling → flat layer curve, embeddings
tie best layer. Added `--pooling {mean,last,sum}` to Stage 1; `last` (final token, has
attended to whole sentence) is the causal-LM standard and should recover the
intermediate-layer advantage. To compare next.

## 2026-06-17 (cont.) — Real Pereira2018 loaded; MultiIndex + NaN handling

Data now loads: assembly is `(presentation=627, neuroid=13553)`. Two structural fixes:
1. **MultiIndex levels:** stimulus text, `experiment`, `subject`, `atlas` are *levels* of
   the presentation/neuroid MultiIndexes, so `list(a.coords)` showed only dim names →
   sentence coord came back `None` and Stage 1 silently ran GPT-2 on placeholder strings
   `stim_0..`. Rewrote the adapter to read levels via `a.indexes[dim].get_level_values`,
   pick the sentence level by whitespace heuristic, and filter experiment/atlas on levels.
2. **NaN voxels:** Pereira voxels are subject/experiment-specific; pooling both experiments
   (627 rows) leaves NaN blocks → sklearn `Input y contains NaN`. Adapter now filters to ONE
   experiment and drops NaN voxels → dense matrix. Config default `experiments: [384]`.
   Verified end-to-end on a simulated MultiIndex assembly (drops 4/8 NaN voxels correctly).

**ACTION for next Colab run:** re-run Stages 1–2 (cached activations were on placeholder
sentences and are invalid; also row count changes 627→384 after experiment filter).

## 2026-06-17 — Colab brain-score data access debugging

**Context:** wiring real Pereira2018 via `brain-score/language` v2.2.1 on Colab (Python 3.12).

**Findings (in order encountered)**
1. `brainscore_language` not installed → loader raised by design. Made the notebook's
   harness-install cell active (install-if-missing) instead of commented out.
2. After install, all plugin access failed with `'EntryPoints' object has no attribute
   'get'`. **Root cause:** brain-score-core calls `entry_points().get(group, default)`;
   that dict-style API was removed in Python 3.12 (now returns a flat `EntryPoints`; use
   `.select(group=...)`). Affects *every* plugin path, incl. direct plugin-module import
   (brain-score's `data/__init__.py` runs plugin discovery on import).
   **Fix:** monkeypatch `importlib.metadata.EntryPoints.get` to delegate to `.select`,
   applied before importing brainscore (`_patch_importlib_entry_points` in `data/pereira.py`).
3. pip dependency-conflict warnings (numpy<2, xarray<2022.6 vs jax/tifffile/arviz) are
   harmless — brain-score intentionally pins numpy 1.26.4; not our blocker.

**Next:** confirm the monkeypatch lets `load_dataset('Pereira2018.language')` download from
S3; capture the printed `[pereira] dims/coords` schema to finalize coord mapping.

---

## 2026-06-16 — Project kickoff & initial scaffold

**Done**
- Read the two framing docs (`LLMBrainAllignment.md` proposal+critiques,
  `llm_brain_alignment_novelty_report.md` novelty assessment).
- Locked the sharpened research question (the *defensible* framing from the novelty report):
  > How much of LLM–brain alignment on Pereira2018 is **uniquely** explained by hidden
  > representations **beyond surprisal**, and how does that residual depend on model size,
  > layer, and instruction tuning?
- Built the initial modular codebase: activation extraction, surprisal, ridge encoding,
  variance partitioning, RSA, surprisal-matched subsets, synthetic data fallback, and a
  5-stage CLI pipeline + Colab notebook driver.

**Key design decisions**
- **Variance partitioning (commonality analysis) is the primary analysis**, not surprisal-
  matched subsets — directly addresses the "collinearity trap" critique (surprisal and
  structure are collinear; matched subsets alone would shrink statistical power).
- **Pooling:** subword tokens → word/sentence via mean or last-token, using tokenizer
  offset mapping. Documented explicitly (Pereira shows whole words; BPE/SentencePiece split).
- **Noise ceiling:** normalize predictivity by split-half / cross-subject reliability so
  numbers are comparable to Schrimpf et al. 2021.
- **Data access:** wrap `brain-score/language` Pereira2018; synthetic fallback so the full
  pipeline runs before real-data access is wired (clearly labelled, never silently used).
- **Models (configurable):** GPT-2 (base baseline), Pythia-160M/410M (scaling),
  OPT-125M, plus instruction-tuned counterparts for the base-vs-instruct contrast.

**Open questions / risks**
- Collinearity between surprisal and hidden states → rely on *unique* variance, report CIs.
- Tokenization mismatch → strict, documented pooling; sanity-check word alignment counts.
- Real Pereira2018 access path in Colab (brain-score install can be heavy) — verify early.
- Git repo currently rooted at `$HOME` (accidental). Need a clean repo in this folder with a
  GitHub remote for the Colab clone workflow.

**Next**
- [ ] Verify `requirements.txt` installs clean in Colab.
- [ ] Wire real Pereira2018 loader via brain-score; confirm assembly shape + ROI/subject metadata.
- [ ] Run Stage 1–3 on GPT-2 to reproduce a layer-depth predictivity curve (baseline sanity check).
- [ ] Then Stage 4 variance partitioning on GPT-2; confirm unique-hidden > 0 above surprisal.
