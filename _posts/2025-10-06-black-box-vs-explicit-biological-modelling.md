---
title: 'Choosing your Optimization Strategy for Biological Sequence Design—Black-Box Learning vs Explicit Rules'
date: 2025-10-06
permalink: /posts/2025/black-box-vs-explicit-biological-modelling/
tags:
  - Machine Learning for Biology
  - Biological Sequence Design
  - Black-box models
  - Mechanistic models
  - RNA Optimization
---

{% include toc %}
<br />
# I. Introduction: The Two-Layer Challenge

When establishing computational approaches to design biological sequences, one often faces two fundamental choices. How to predict the sequence properties you care about, and how to search the vast sequence space for candidates that are optimal with respect to these properties. 

**Layer 1: Property Prediction Models**\
At the core is the need to predict the properties or functionalities that the designed sequences should exhibit. These properties may include stability, binding affinity or translational efficiency. Property prediction models can be used to estimate how well a candidate sequence meets the design objectives. They take as input a candidate sequence and output a score or prediction reflecting the desired property. The optimization step can then aim to maximize or minimize that property, target specific values, a specific range etc. These models can be data-driven empirical models (often referred to as black-box models), or they can be derived from physical first principles (mechanistic models). The accuracy of this predictive layer significantly influences the downstream search for optimal sequences.

**Layer 2: Sequence Optimization Algorithms**\
After getting property prediction models, the second challenge is to navigate the astronomical sequence space to identify candidates that maximize the target properties. Consider the problem of designing mRNA sequences encoding the SARS-CoV-2 spike protein. This protein has 1,273 amino acids, and each amino acid can be encoded by different codons, called synonymous codons. For the SARS-CoV-2 spike protein, there are around $2.4 * 10^{632}$ candidate mRNA sequences [(Zhang et al., 2023)](https://www.nature.com/articles/s41586-023-06127-z), which is many orders of magnitude greater than the numbers of particles in the universe. Exhaustive enumeration is impossible, so we rely on sequence optimization algorithms to efficiently explore and exploit this space. Just like for the first layer, there are multiple ways to tackle these combinatorial optimization problems, but one important choice that has to be made and that I decided to explore in this article, is whether to choose an end-to-end differentiable optimization routine or a sampling-based one.

In the remainder of this article, I unpack these choices. Section II clarifies what “differentiable” really means in the context of biological sequence modelling, and why it matters. Section III dissects the choice of black-box models versus mechanistic models for Layer 1, property predictors. Section IV turns to Layer 2 and explores the tradeoffs between end-to-end differentiable optimization and sampling-based optimization. Finally, Section V summarizes the trade-offs into a decision guide.

# II.  Differentiability in Biological Context

At its core, differentiability refers to the existence of derivatives (gradients) of functions with respect to its inputs. Consider a function $f: \mathbb{R}^{N \times V} \to \mathbb{R}$ that maps an input matrix  (e.g., a one hot encoded biological sequence) to a scalar output  (e.g., predicted property score). The function is differentiable at $X \in \mathbb{R}^{N \times V}$  if $\nabla f(X)$, the gradient of $f$ at $X$, exists. This gradient tells us how infinitesimal changes in each component $X_{i,j}$ affect the output. For example, in neural networks, $f$ is a composition of differentiable layers, enabling back-propagation to efficiently compute $\nabla f(X)$. In our case, biological sequences are fundamentally discrete: nucleotides or amino acids are categorical variables, not continuous numbers. To apply differentiable optimization methods, sequences are often encoded. A simple yet very used example is to one-hot-encode sequences, using in the above definition N as the sequence length and V as the vocabulary size (4 for DNA or RNA sequences). 

Differentiability is orthogonal to whether a model is mechanistic (rule/physics‑based) or black-box (data-driven). Mechanistic models can be expressed with continuous states and differentiable update rules, so gradients propagate end to end and enable gradient‑based optimization. They can also contain non-differentiable operations, preventing back-propagation. Black-box data‑driven predictors such as neural networks are inherently differentiable, but others such as decision trees and random forests are not.

To avoid conflation, it is helpful to separate the pertinence of differentiability in our two axes:
- Property prediction: mechanistic models specify properties through explicit rules, black‑box models learn mappings from data. Either category may be differentiable or not.
- Optimization strategy: end-to-end differentiable optimization methods require differentiable components by definition, sampling‑based methods treat the scorer as a black box and work regardless of differentiability.

# III. Layer 1: Property Prediction Models

## **A. Black-box Property Predictors**

Black box property predictors take datasets of sequences and associated experimentally measured properties and learn a function $f(\mathbf{X})$ that maps sequences to the property of interest. The modeling toolbox is broad, ranging from linear regressions and shallow neural networks to Random Forests, CNNs and Transformers. Deep domain expertise is not strictly required to build these models, but biological understanding meaningfully improves outcomes. It informs how to process and normalize biological experimental assays to get ML ready data, and how to represent and encode sequences. When building deep neural network models, biological understanding guides architectural choices: if local motifs dominate, a CNN may suffice, for long-range dependencies, attention mechanisms or Transformers are more appropriate.

This setup is flexible enough to accommodate many tasks seen across the literature, such as predicting mRNA degradation with a hybrid CNN–RNN architecture as done with the [Saluki model](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) (Agarwal & Kelley, 2022), or modeling translation efficiency with the [RiboNN model](https://www.biorxiv.org/content/10.1101/2024.08.11.607362v1) (Zheng et al., 2024). The same pattern extends beyond these examples: whenever there is available data that links sequences to properties, one can use it to train a custom ML model. In practice, this often means starting from a straightforward baseline model such as a Random Forest, then iterating on data processing, sequence representation and model architecture until accuracy and calibration needs are met for downstream design.

Given adequate data, black box predictors can achieve strong held‑out accuracy, reflecting their ability to learn complex sequence-property relationships. These relationships might not be explicit, meaning that they could not be translated into heuristics or a mechanistic algorithm. When implemented in a differentiable way, they integrate seamlessly into gradient‑based pipelines, and when not, they still serve as fast scorers for sampling‑based search. Moreover, because these models can be trained quickly as new measurements arrive, they naturally support iterative improvements: train a model on current data, generate specific candidates with the model (targeting extreme property values for instance), screen them experimentally, and feed the new data back into the model. This active-learning cadence steadily improves the predictor in the regions that matter for the design objective. As candidates accumulate, their shared patterns can also reveal informative signal, enabling hypothesis generation and possible new biological insights.

There are, however, important trade-offs. While the models can be retrained quickly, incorporating new measurements still depends on experimental timescales, which can range from weeks to months. Another dominant constraint is coverage: biological sequence spaces are astronomically large, and models are almost always trained on a thin, biased subspace of that space. As a result, generalization beyond the measured subspace is difficult; uncertainty should be quantified, and used to govern exploration and risk. Interpretability is also limited compared to explicit, rule-based frameworks; even with attribution maps or attention analyses, justifying specific design decisions can be difficult. In a biotechnology/pharmaceutical industry setting, the black-box nature of these models can complicate their explanation and acceptance by stakeholders.

## **B.  Mechanistic Property Predictors**

Mechanistic property predictors tackle sequence–property tasks from a fundamentally different angle. Rather than learning end-to-end mappings from continuous sequence representations, these models explicitly characterize the mechanisms that drive property values. They encode hypothesized causal structure (e.g., thermodynamic energies, kinetic schemes, hard constraints, dynamic‑programming recurrences) to produce predictions whose components are transparent and auditable. Some implementations are fully differentiable and thus can participate in end‑to-end differentiable optimization pipelines; others rely on non-differentiable operations and are better paired with sampling‑based optimization.

Mechanistic approaches span diverse formalisms such as thermodynamic free-energy calculations, dynamic-programming algorithms that optimize over discrete structures or differential equation models solved with differentiable integrators such as [Diffrax](https://arxiv.org/abs/2202.02435) (Kidger, P. 2022). Some known dynamic programming examples include the [Zuker–Stiegler dynamic programming algorithm for RNA secondary structure prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC326673/) (Zuker & Stiegler, 1981), which computes minimum free energy through explicit thermodynamic parameters. [LinearDesign](https://www.nature.com/articles/s41586-023-06127-z) (Zhang et al., 2023) builds on this foundation with a different dynamic programming algorithm that takes as input a protein sequence and searches for optimal RNAs based on secondary structure and the codon-adaptation index (CAI) to reward preferred codons. 

A first benefit of mechanistic models is their interpretability. Models built on decades of biochemical research offer trustworthy predictions where every energy contribution or score component can be traced back to well-understood molecular interactions. For example, physics-based differential equation models offer high interpretability because of the possible mapping between parameters and physical concepts (such as decay rate constants that explain signal behavior over time). This interpretability advantage is particularly valuable in regulated industries where model decisions must be justified. Another very important benefit is the ability to use biological knowledge to create algorithms that can generalize well across the sequence space, whereas black-box data driven approaches are limited by the subspace represented by the available data. Additionally, these models do not need data to be trained on; the mechanism itself acts as prior knowledge that bounds sequence to properties even when examples are scarce.

However, these approaches face important constraints. Mechanistic models require deep domain expertise to construct and validate, though once built they can often be applied immediately without extensive training data. If parts of the underlying mechanisms behind the properties of a sequence are not well-understood, they can't be incorporated in the modeling.

Another drawback that can occur for both mechanistic and black-box models is that only differentiable implementations can be back‑propagated through. Non-differentiable models cannot be integrated into end-to-end differentiable optimization pipelines, limiting their compatibility with gradient-based design algorithms that require backpropagation through the property predictor.

# IV. Layer 2: Optimization Algorithms

## **A. End-to-end differentiable optimization routines**

End-to-end differentiable optimization routines leverage gradients to navigate sequence space systematically, steering updates toward candidates with improved predicted properties. They require property predictors to be differentiable. In the simplest setting, a sequence is encoded into a continuous embedding and standard gradient descent adjusts each position so that a differentiable property predictor’s score improves. The same principle scales to more expressive generators: diffusion models, Transformers, or VAEs learn to emit whole sequences and are trained end-to-end with the predictor’s loss as a guiding signal. In every case, back-propagation supplies an exact direction in which to nudge the sequence or the generator’s parameters, turning an otherwise intractable combinatorial problem into a tractable continuous optimization problem. One example is [RiboCode](https://www.biorxiv.org/content/10.1101/2024.09.06.611590v1) (Li, Y. et al. 2024), a model that optimizes mRNAs for both translational efficiency and RNA structure.

The chief advantage is purposeful navigation.  Because each update follows the objective’s gradient, the search can converge on high-scoring regions in far fewer evaluations than random or heuristic methods.  The machinery is also modular: any differentiable property model, whether black-box or mechanistic, alone or in a weighted combination can be “plugged in” without altering the optimization logic, letting practitioners tune multiple objectives simply by adding terms to the loss.

These gains come with familiar trade-offs. Like the predictors they rely on, gradient-driven methods are hard to interpret: the calculus explains why a step improves the numerical loss, but not the biology behind property improvement. As is typical in non‑convex optimization, gradient methods can [converge to local optima](https://apxml.com/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/gradient-descent-challenges), and after training the fixed weights tend to sample a narrow neighborhood of that solution. That means that sequences would have low diversity. There aren't straightforward fixes but one can experiment with injecting stochasticity through for instance noise or temperature scaling.

Despite these caveats, end-to-end differentiable optimization routines have become widely used in modern sequence-engineering pipelines, offering a scalable, mathematically principled route through the vast design space.

## **B. Sampling-based optimization routines**

Sampling-based optimization routines explore sequence space without gradients. The only feedback they need is a scalar score: be that a thermodynamic energy, a wet-lab measurement, the output of a differentiable model treated as a black box, or a combination of many such numbers. At the simplest end sit rule-based heuristics that pick codons or motifs position-by-position; at the other extreme are full population-based genetic algorithms such as [AdaLead](https://arxiv.org/abs/2010.02141) (Sinai, S. et al., 2020), which mutates and recombines sequences to climb a fitness landscape, or reinforcement-learning schemes like [DyNA-PPO](https://openreview.net/forum?id=HklxbgBKvr) (Angermueller, C. et al.) that refine a policy through repeated actions with well-defined states and rewards.

Because every move is hand-specified, these methods offer explicit control.  Biologists can embed hard constraints (such as specific sites to avoid) and design interpretable operators e.g., “swap synonymous codons”, that reflect known molecular rules. Their sampling nature naturally supports diverse exploration: populations, mutations and crossovers help escape the local minima that often trap gradient descent.  Finally, by depending only on a numerical reward, they can mix-and-match both differentiable and non-differentiable property predictors. 

The trade-offs are real.  Hand-crafted moves give no guarantee of optimality; the search may never stumble on remote high-fitness subspaces. Performance is hyper-parameter sensitive: mutation rate, population size and reward weights can each tilt the outcome and usually require empirical tuning.  And because there is no convergence guarantee, it is hard to diagnose stagnation: a flat fitness curve might mean a global optimum, or that exploration got stuck on a plateau.  For reliable deployment, one often needs to run many seeds and monitor diversity metrics.
# V. Putting it all together: a Practical Decision Guide

Use the checklist below to decide which flavor of method fits your data, constraints and goals, and keep in mind that the two philosophies can always be mixed.

**Layer 1: Predictor choice**

Black‑box predictors are suitable when:
- Data is available and/or there are possibilities to experimentally generate sequence‑property pairs.
- Target properties depend on subtle, high‑order interactions that resist rule writing.
- Generalizability is not primordial and a model tailored to a specific use case/subspace is sufficient.
- Interpretability is not essential.
- Rapid design–test–learn cycles will drive new insights and model retraining.

Mechanistic predictors are suitable when:
- Data is sparse or assays are costly.
- Mechanistic insight enables explicit rule sets/parameters.
- Generalizability across use cases/subspaces is important.
- Interpretability is required for regulatory or safety review.

**Layer 2: Optimization routine choice**

End-to-end differentiable optimization routines are suitable when:
- Differentiable property predictors are available.
- Objectives can be expressed (or relaxed) as differentiable losses for directed updates.
- Fast convergence via local gradient guidance is desired.
- Biological/task-specific optimization constraints can be encoded  in a differentiable way or handled with reliable differentiable estimators and relaxations.

Sampling‑based optimization routines are suitable when:
- Differentiable property predictors are unavailable.
- Diversity is a priority; stochastic or population‑based exploration keeps multiple paths open.
- Safety or regulatory constraints require explicit control over moves and traceable decision rules throughout the search.
- Biological/Task-specific optimization constraints can't be encoded in a differentiable way.

Choose carefully.  Resist the reflex to directly fit a neural network or code heuristic-based algorithms. Explore and understand the data you have, the constraints you face, and the confidence you need.  The “right” pipeline is the one that consistently yields sequences with the properties that matter to your project, and does so with a level of certainty that lets you act on the results.  Sometimes that means gradients all the way, sometimes pure rules, and often a blend that evolves as new data arrive.
# Acknowledgments
I’m grateful to Eric J. Ma for encouraging me to start this blog, and for his thoughtful review and feedback of this post which significantly improved the clarity and organization of the final piece.

# References

- Zhang, H., Zhang, L., Lin, A., Xu, C., Li, Z., Liu, K., Liu, B., Ma, X., Zhao, F., Jiang, H., Chen, C., Shen, H., Li, H., Mathews, D. H., Zhang, Y., & Huang, L. (2023). Algorithm for optimized mRNA design improves stability and immunogenicity. _Nature_, _621_(7978), 396–403. https://doi.org/10.1038/s41586-023-06127-z
- Agarwal, V., & Kelley, D. R. (2022). The genetic and biochemical determinants of mRNA degradation rates in mammals. _Genome Biology_, _23_(1). https://doi.org/10.1186/s13059-022-02811-x
- Zheng, D., Wang, J., Persyn, L., Liu, Y., Montoya, F. U., Cenik, C., & Agarwal, V. (2024). Predicting the translation efficiency of messenger RNA in mammalian cells. _bioRxiv (Cold Spring Harbor Laboratory)_. https://doi.org/10.1101/2024.08.11.607362
- Kidger, P. (2022, February 4). _On neural differential equations_. arXiv.org. https://arxiv.org/abs/2202.02435
- Zuker, M., & Stiegler, P. (1981). Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information. _Nucleic Acids Research_, _9_(1), 133–148. https://doi.org/10.1093/nar/9.1.133
- Li, Y., Wang, F., Yang, J., Han, Z., Chen, L., Jiang, W., Zhou, H., Li, T., Tang, Z., Deng, J., He, X., Zha, G., Hu, J., Hu, Y., Wu, L., Zhan, C., Sun, C., He, Y., & Xie, Z. (2024). Deep generative optimization of mRNA Codon sequences for enhanced protein production and therapeutic efficacy. _bioRxiv (Cold Spring Harbor Laboratory)_. https://doi.org/10.1101/2024.09.06.611590
- ApX Machine Learning. _Challenges: local minima and saddle points_. https://apxml.com/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/gradient-descent-challenges
- Sinai, S., Wang, R., Whatley, A., Slocum, S., Locane, E., & Kelsic, E. D. (2020, October 5). _AdaLead: A simple and robust adaptive greedy search algorithm for sequence design_. arXiv.org. https://arxiv.org/abs/2010.02141
- Angermueller, C., Dohan, D., Belanger, D., Deshpande, R., Murphy, K., & Colwell, L. (n.d.). _Model-based reinforcement learning for biological sequence design_. OpenReview. https://openreview.net/forum?id=HklxbgBKvr
