# Novelty Assessment: Disentangling Statistical Surprisal from Hierarchical Structure in LLM–Brain Alignment

## Overview

The proposed project is **partly novel but not fully unprecedented**. The broad question—whether LLM–brain alignment reflects linguistic structure, lexical semantics, or next-word prediction—has already become a central debate in neuro-AI, and multiple recent papers address important pieces of it.[cite:13][cite:18][cite:39]

What appears new is the **specific experimental package**: applying explicit surprisal control to hidden-state-to-brain alignment on Pereira2018, combining variance partitioning with surprisal-matched subsets, and comparing base versus instruction-tuned as well as smaller versus larger models under that control regime.[cite:24][cite:26][cite:39]

## What prior work already established

Schrimpf et al. and later follow-up work established that transformer language models, especially GPT-family models, predict neural responses in language-network datasets such as Pereira2018 very well, with intermediate layers often performing best.[cite:23][cite:25] Subsequent work also found that larger language models often improve neural predictivity, and that the peak-performing layer tends to move earlier as model size increases.[cite:24]

Kauf, Tuckute and colleagues directly tested what aspects of linguistic input matter for ANN–brain similarity by systematically perturbing word order, removing function versus content words, and replacing sentences with semantically related or unrelated alternatives on Pereira2018.[cite:13][cite:26] Their main result is that lexical-semantic content contributes more strongly than syntactic form, but their analysis does **not** statistically control for surprisal in the main hidden-state-to-brain mapping and does **not** construct surprisal-matched stimulus sets.[cite:13][cite:26]

Other studies have modeled neural data with explicitly statistical and structural predictors together. For example, recent MEG work jointly included GPT-derived surprisal and entropy with rule-based syntactic features such as depth and constituent closing operations, showing that both classes of predictors explain neural responses and have partly distinct temporal profiles.[cite:18] Likewise, work on Dutch story listening included surprisal as a control predictor while evaluating syntactic node-count measures, showing that structure-sensitive neural effects can survive information-theoretic controls.[cite:17]

There is also evidence that the relationship between prediction and brain alignment is not exhausted by plain next-word prediction. Merlin and Toneva explicitly compared conditions while controlling for changes in next-word prediction performance and argued that some alignment depends on information beyond next-word prediction and word-level content alone.[cite:39] That makes the project’s core motivation well grounded, but it also means the general “beyond surprisal” claim is no longer new by itself.[cite:39]

## Where the proposal looks novel

The most promising novelty lies in **formal surprisal ablation at the hidden-state level**. Existing papers either use surprisal directly as a neural predictor, or correlate condition-level drops in brain predictivity with increases in surprisal or perplexity, but they generally stop short of asking how much voxel-level hidden-state predictivity remains once surprisal is explicitly controlled in the same encoding framework.[cite:18][cite:26]

A second novel element is the idea of **surprisal-matched stimulus subsets** for Pereira2018 or related controlled sentence sets. The available literature discusses perturbations, semantic substitutions, and predictive metrics, but no clear prior example was found of matching stimuli on GPT-2 surprisal and then testing whether residual ANN–brain alignment still follows syntactic or compositional manipulations.[cite:13][cite:18][cite:26]

A third novel angle is the comparison of **base versus instruction-tuned models** in this neuro-alignment setting. The surveyed work mainly focuses on base autoregressive models such as GPT-2 variants and scaling families, and did not reveal an established line of work directly testing whether instruction tuning makes models more or less brain-like after controlling for surprisal.[cite:23][cite:24][cite:26]

A fourth plausible contribution is the explicit test of the claim that **bigger is not necessarily more brain-like once surprisal is controlled**. Some current work suggests larger models improve brain predictivity overall, while other behavioral work raises the possibility that highly capable models may become less human-like on prediction-based behavioral measures.[cite:24][cite:39][cite:40] Connecting that debate to residual brain alignment after surprisal control would be a useful and timely extension.[cite:24][cite:39][cite:40]

## Limits on novelty

The project should not be framed as the first study to question whether LLM–brain alignment is driven by structure versus prediction. That framing would be too strong, because current work already compares lexical-semantic, syntactic, and predictive explanations of neural fit, and some of it explicitly includes surprisal-like controls.[cite:13][cite:17][cite:18][cite:39]

It is also not the first study to use Pereira2018 for mechanistic probing of ANN–brain similarity. Kauf et al. already use Pereira2018 plus controlled sentence perturbations to ask what features drive alignment, so a new study on the same benchmark must clearly explain what additional causal leverage is gained by surprisal matching or joint variance partitioning.[cite:13][cite:26]

Finally, the proposal’s novelty depends heavily on **methodological rigor**. If surprisal is only added as one more post hoc correlation or summary statistic, the study risks looking incremental relative to work that already links perturbation-induced predictivity drops to reduced next-word prediction performance.[cite:26] The strongest version is one that quantifies unique versus shared variance from hidden states and surprisal, preferably region-wise and layer-wise, and then validates the conclusions with matched-stimulus analyses.[cite:18][cite:26]

## Assessment

The proposal is best described as a **targeted and publishable incremental contribution**, not a wholly new research question. Its contribution would come from combining existing debates into a sharper causal design: hidden-state encoding models, explicit surprisal control, surprisal-matched subsets, and model-family comparisons that the current literature does not appear to combine in one study.[cite:18][cite:24][cite:26][cite:39]

In practical terms, the novelty is strongest if the project is pitched as: **How much of LLM–brain alignment on Pereira2018 is uniquely explained by hidden representations beyond surprisal, and how does that residual depend on model size, layer, and instruction tuning?** That framing is narrower, more defensible, and better aligned with what appears missing from the existing literature.[cite:24][cite:26][cite:39]
