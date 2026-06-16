This is an exceptionally well-framed, highly feasible, and timely research proposal. You have correctly identified a critical bottleneck in the current LLM-brain alignment literature: the field has largely saturated on "predictivity" leaderboards (like the Brain-Score language benchmarks) and is now demanding mechanistic explanations for *why* these alignments exist.

Disentangling structural processing from mere statistical prediction (surprisal) is exactly the right target for 2026, especially as larger models begin diverging from human-like reading behaviors.

Here is a breakdown of your proposal's strengths, along with a few methodological critiques and refinements to ensure it stands up to rigorous peer review at venues like CCN or NeurIPS UniReps.

### **What Makes This Proposal Strong**

* **Exceptional Scoping (Compute & Timeline):** Targeting the M2 Mac environment with ≤410M parameter models is a very smart constraint. By avoiding training/fine-tuning and sticking to inference, activation extraction, and ridge regression, a 2–3 month timeline is highly realistic.
* **Leveraging Existing Infrastructure:** Using the `brain-score/language` GitHub repo rather than building your own fMRI mapping pipeline from scratch saves weeks of work and instantly makes your results directly comparable to Schrimpf et al. (2021) and others.
* **Clear Novelty Axes:** Testing the "bigger ≠ more brain-like" hypothesis (using the Pythia suite or OPT) and comparing base vs. instruction-tuned models adds crucial modern context to older GPT-2 baseline studies.

---

### **Critiques and Areas for Refinement**

While the roadmap is solid, Step 3 and Step 4 will introduce some specific statistical and methodological challenges you should prepare for:

**1. The Collinearity Trap in Step 3**
Syntactic complexity and statistical surprisal are notoriously collinear (rare words or complex structures are almost always statistically surprising to an LLM).

* *The Risk:* If you rely solely on building "surprisal-matched stimulus subsets," you might end up with a very small, highly unnatural subset of the Pereira 2018 sentences, stripping your ridge regression of statistical power.
* *The Fix:* Instead of just matching subsets, prioritize **Variance Partitioning** (also known as commonality analysis). Run your regressions with surprisal alone, structure alone (perhaps using a symbolic parser's depth metrics), and both combined. This allows you to quantify the *unique* variance explained by model representations over and above the variance explained by simple token surprisal.

**2. Tokenization Mismatches**
Pereira (2018) presents stimuli as whole words/sentences. GPT-2, Pythia, and Llama all use different subword tokenizers (BPE vs. SentencePiece).

* *The Risk:* When calculating surprisal or extracting activations for a specific "word" in the fMRI data, an LLM might split that word into three tokens.
* *The Fix:* You will need a strict pooling strategy (e.g., taking the mean or the last token's activation for a given word) to align the sequence lengths before mapping to the BOLD signal. Document this explicitly.

**3. Representational Similarity Analysis (RSA) vs. Predictivity**
You listed `rsatoolbox` as optional, but I highly recommend making RSA a core part of Step 5. Linear predictivity (ridge regression) shows that the *information* is linearly accessible, but RSA (comparing the Representational Dissimilarity Matrices of the LLM layers directly to the fMRI RDMs) is often a cleaner way to show if the *geometry* of the representations matches, which is crucial for claims about shared "linguistic structure."

---

### **Publication Strategy**

Your target venues are perfectly chosen.

* **CCN 2026 (Aug):** This is the ideal home for this. A 2-page poster abstract would be a near-guaranteed acceptance given the clarity of the problem, but the 8-page proceedings track is well within reach if the variance partitioning yields clear results.
* **CMCL Workshop:** Excellent backup, as they love precisely this kind of psycholinguistic probing.

Topic 1 — Does Linguistic Structure or Mere Surprisal Drive LLM–
Brain Alignment? A Probing Study
Title: Disentangling Statistical Surprisal from Hierarchical Structure in LLM–Brain
Alignment
Problem statement: LLM hidden states predict human language-network fMRI/EEG
responses — in Schrimpf et al. 2021 (PNAS 118(45):e2105646118), GPT2-xl "predicts
Pereira2018 and Fedorenko2016 at close to 100% of the noise ceiling," with "Intermediate
layer representations... most predictive." But it is contested whether this alignment reflects
1 / 8shared linguistic structure or just shared sensitivity to next-word predictability (surprisal).
The project quantifies how much alignment survives when surprisal is controlled.
Why it matters now (2026): Brain–LLM alignment is one of the hottest neuro-AI threads,
and the structure-vs-statistics question is a genuinely live debate: Kauf, Tuckute et al. 2023
(Neurobiology of Language 5(1):7–42) argue that "Lexical semantic content, not syntactic
structure, is the main contributor to ANN-brain similarity of fMRI responses in the language
network," while other 2025–2026 work shows larger models can become less human-like in
reading-time/surprisal fit (e.g., "To model human linguistic prediction, make LLMs less
superhuman," arXiv 2510.05141).
Novelty angle: Rather than reporting yet another correlation, systematically ablate the
surprisal confound: match stimuli on GPT-2 surprisal and test whether residual alignment
tracks syntactic/compositional manipulations. Compare base vs instruction-tuned models, and
small vs large models, to test the "intermediate-layer advantage" and "bigger ≠ more brain-
like" claims.
Datasets: - Pereira et al. 2018 (Nature Communications 9:963, DOI 10.1038/
s41467-018-03068-4) fMRI — language-network responses to sentences; experiments 2–3
used 384 and 243 sentences each shown for 4 s. Publicly available (OSF crwz7) and bundled
into the Brain-Score language harness. - Optional: Brain-Score language benchmarks
(Fedorenko2016 ECoG, Blank2014 stories) via the open brain-score/language GitHub repo.
Tech stack: Python, Hugging Face Transformers (GPT-2, Pythia-160M/410M, OPT-125M),
scikit-learn (ridge regression), brain-score/language , numpy/scipy, optional rsatoolbox for
RSA. PyTorch with MPS backend for activation extraction.
Implementation roadmap: 1. Reproduce the Brain-Score linear-predictivity pipeline on
Pereira (2018) with GPT-2 as a baseline. 2. Extract layer-wise activations for all sentences; fit
ridge regression to voxel responses; report normalized predictivity per layer. 3. Compute
surprisal for each token/sentence; build surprisal-matched stimulus subsets or include
surprisal as a regression covariate. 4. Run controlled comparisons: base vs instruction-tuned,
small vs large, and structure-manipulated stimuli. 5. Analyze layer-depth curves; test
intermediate-layer-advantage and platonic-convergence hypotheses.
Estimated compute (M2): Very light. Models ≤410M params run on CPU/MPS for inference
only; activation extraction over a few thousand sentences is minutes-to-hours. Ridge
regression is trivial on CPU. No training. Fits comfortably in 8–16GB. No Colab needed.
Difficulty: 4/10.
Timeline: 2–3 months part-time.
Publication feasibility: CCN 2026 (2-page poster or 8-page proceedings track, NYU, Aug
2026); NeurIPS UniReps workshop; ACL/EMNLP CMCL (Cognitive Modeling and
Computational Linguistics) workshop.
Extensions: Add MEG/EEG temporal data (THINGS-MEG) to test when structure emerges;
extend to multilingual or L2-reader alignment (cf. "Surprisal in reading: language models
predict the N400 for L2 readers"); test newer open models (Llama-3.2-1B).


