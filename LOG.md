# Research Log — LLM–Brain Alignment (Surprisal vs Structure)

Chronological notes. Newest entries at the top. Keep entries short: what was done,
what was decided, what's blocked, what's next. Link runs to `TRACKER.md` IDs.

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
