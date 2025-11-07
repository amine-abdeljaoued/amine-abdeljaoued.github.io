---
title: 'Why ML for Biology Demands Biological Understanding'
date: 2025-11-07
permalink: /posts/2025/ML-for-biology-requires-biological-knowledge/
tags:
  - Machine Learning for Biology
  - Biological Sequence Design
  - Understanding Biology
  - Wet-lab assays
  - Data processing
  - Communicating results
  - RNA Optimization
---

{% include toc %}
<br />


# Introduction

Real progress in ML for biology comes from models grounded in biological reality: how experiments work, how data are produced, and the biological mechanisms the readout reflects. That is the difference between a technically impressive model and one that is pertinent, robust, and can inform decisions and guide the next experiments.

In my day‑to‑day work on mRNA and protein design, the wins have come from pairing modeling skills with practical biology: understanding assays, making preprocessing choices that reflect the experiment, and picking model architectures that match the mechanisms I’m trying to capture. After that, communicating results with expectations set by the problem’s biological complexity and the variability of the measurements. 

This post highlights four places where biological insight is essential and makes modeling efforts actionable: assay understanding, data processing, modeling, and analysis/communication of results.

# Wet-lab assays

A biological assay is an experiment that turns a biological question into a measurable signal. Some examples include a fluorescence readout for gene expression or a binding assay signal reflecting the affinity between a protein and the molecule it recognizes. Understanding the biology behind that signal clarifies what information is present, what it reflects, and how to interpret the scale and units so modeling choices align with the phenomenon of interest.

A few examples show why biological understanding matters here. First, within the same setup, repeated measurements of the same sample don’t perfectly agree. Instrument limits and handling can add randomness that limits how consistently a model can match replicate measurements. A separate concern arises across runs: systematic context changes such as new reagent lots, altered plate layouts, or shifted incubation times, can move the entire distribution in ways that look like biology but are actually shifts caused by differences in how the experiment was run. Recognizing this distinction is key for preprocessing and interpretation.

Understanding biology also helps you design experiments so that modeling captures the right signal and can be acted on. Sometimes there isn’t an assay for the property you care about; understanding the mechanism lets you co‑develop a readout with wet‑lab scientists that reflects the phenomenon and yields data in a model‑friendly format. Collaborating early ensures the readout covers the range that matters, includes clear positives and negatives, and has enough replicates to estimate variation rather than guess it later. With the biology in view, assay design can align with the modeling goal, yielding models that are pertinent and impactful.

# Data processing

Data processing starts with what the assay measures and the biological level at which the signal lives, because that determines how to normalize, what to filter, and which comparisons make sense. If your dataset spans multiple transcripts, you might want to normalize within each transcript first so differences in RNAs are not mistaken for position‑level effects. You might then filter or weight using replicate variability so unstable measurements contribute less or are removed. Understand and acting upon the biology behind the assay ensures you move from raw numbers to data that reflects the biological problem you care about.

Some biological literacy about common tools helps you ingest raw outputs and interpret results without reinventing the wheel. Familiarity with biological data formats like FASTA and tools for sequence search and alignment such as BLAST and Bowtie helps you place assay signals in their proper biological context. These tools let you map raw data back to known sequences, genes, or motifs, making it easier to interpret what the assay is measuring. Together, these tools connect raw assay outputs to meaningful sequence information relevant to your task.

Biological understanding can also simplify the learning problem by matching the data representation to the mechanism. If the property varies at the level of local motifs, frame the task as window‑to‑value rather than full‑sequence‑to‑profile: fragment sequences into windows and predict a single value per window instead of forcing a long‑range sequence‑to‑sequence model. Conversely, if long‑range interactions are plausible, preserve the full context or add features that summarize global structure rather than fragmenting the input. 

Encoding choices should follow the biology of the task. For RNA, for instance, use nucleotides for motif‑level effects, codons when translation dynamics matter, and amino acids when modeling protein‑level properties. Train, validation and test splits should also reflect biological relatedness, not just random rows. In that same example, splitting by proteins can prevent leakage and produce robust evaluations. 

In practice, understanding the biology lets you align normalization, filtering, preprocessing, and encoding with what the assay can truly tell you, producing cleaner targets and representations that support pertinent models - and those are the models that can ultimately be acted upon.
# Model development and performance

It all starts with the biology. If there are concrete features known to drive the property (motifs, k‑mer counts, specific structural elements), featurize your inputs accordingly and use simple, auditable models such as GLMs or EBMs. If the mechanism is unclear or likely non‑linear, encode sequences (one‑hot, embeddings, etc.) and consider models that can discover patterns in these more complex representations.

As discussed earlier, understanding the biology of the assay clarifies its uncertainty. That uncertainty should be reflected rather than ignored. You can account for replicate variability in the loss or sample weights, and you have options to expose uncertainty when useful: use confidence intervals, ensemble models to estimate prediction uncertainty, or Bayesian models when you want uncertainty that can be propagated into downstream decisions.

Model architectures should also be based on the biological context. When effects are local, windowed setups or CNNs with kernel sizes aligned to expected motif lengths are efficient and interpretable; when interactions span longer ranges, use architectures that carry broader context, such as transformers or hybrids that mix local pattern detectors with global summaries, instead of forcing a purely local view. These are illustrative choices rather than hard rules: recurrent models, structure‑aware or graph models, and additive models all have places where they shine, with associated trade‑offs.

Bake in biological constraints up front so the model explores relevant regions. If certain motifs must be avoided, add masks or hard rules; if codon usage or structural constraints apply, encode them as constraints or penalties rather than filtering after the fact. This reduces wasteful exploration and keeps outputs usable without heavy post hoc cleanup.

Design evaluations in the frame of the next question you need to answer, not just what is easy to score. Use the biology‑aware split strategy discussed above to avoid leakage, and choose metrics that match the decision you will take, such as ranking for selection, probabilities you can threshold for go or no‑go decisions, or regression metrics when you will compare predictions to a numeric readout.

For a deeper dive on biological sequence modeling, see this [piece](https://amine-abdeljaoued.github.io/posts/2025/black-box-vs-explicit-biological-modelling/).

# Analyze and communicate results

Biological data are noisy and context‑dependent, so set expectations to match that reality and emphasize what is robust across repeats or conditions. Lower correlations can still be useful when the signal is subtle or when the practical goal is to rank the best candidates rather than predict exact values. Use visuals to make uncertainty and trade-offs visible at a glance, not to chase a perfect score; a good figure should clarify what is reliable and what is tentative. Focus on biologically meaningful insights, like motifs that consistently help or constraints that prevent failures, instead of chasing marginal metric gains that may not translate to biological language.

Formal training in biology helps, but it is not required. With curiosity and steady practice, anyone can learn and turn models into decisions that move the science forward.

