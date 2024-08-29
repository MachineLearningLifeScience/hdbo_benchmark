---
layout: docs
type: minimal
modules: ["katex"]
---
### Preliminaries

#### Bayesian Optimization and Gaussian processes  
Bayesian optimization requires a surrogate model and an acquisition function
[@Garnett:BOBook:2023]. Given both, the objective function is
sequentially optimized by fitting a model to the given observations and
numerically optimizing the acquisition function with respect to the
model to select the next configuration for evaluation. Frequently, the
model is a Gaussian process (GP, @RasmussenWilliams:GPs:2006), and
popular choices for the acquisition function are *Expected Improvement*
[@Jones:BO:1998; @Garnett2023BObook] and the *Upper Confidence Bound*
[@Srinivas:UCB:2012]. A GP allows to express a prior belief over
functions. 

{{< image src="img/figure_2.png" caption="Figure 2: Existing BO methods tackle problems with insufficiently low effective dimensions. This figure shows sequence length and nr. of categories of the highest search space in the original tests. For reference, the discrete optimization problems usually tackled by practitioners in chemistry and biology are of the order of 10<sup>2</sup> in sequence length, and > 10<sup>1</sup> in nr. of categories. Methods that optimize directly in discrete space (e.g. BODi, ProbRep, Bounce; Sec. 3.7) are tested in lower sequence lengths and dictionary sizes; methods that rely on unsupervised information (e.g. LaMBO, etc.; Sec. 3.5) are able to optimize more complex problems, like protein engineering or small molecule optimization.">}}

Formally, it is a collection of random variables, such that
every finite subset follows a multivariate normal distribution,
described by a mean function $\mu$, and a positive definite covariance
function (kernel) [@RasmussenWilliams:GPs:2006 p. 13]. Assuming that
observations of the function are distorted by Gaussian noise, the
posterior over the function conditioned on these observations is again
Gaussian. The prediction equations have a closed form and can be
evaluated in $\mathcal{O}(N^3)$ time where $N$ is the number of
observations.

**Is High-dimensional Bayesian Optimization difficult?**   There are
three reasons why BO is thought to scale poorly with dimension. The
first reason is that GPs fail to properly fit to the underlying
objective function in high dimensions. Secondly, even if the GPs were to
fit well there is still the problem of optimizing the high-dimensional
acquisition function. Finally, Gaussian Processes are believed to scale
poorly with the size of the dataset, limiting us to low-budget scenarios
[@BinoisWycoff:HDGPs:2022]. Folk knowledge suggests that GPs fail to fit
functions above the meager limit of $\sim 10^1$ dimensions
[@SantoniDoerr:HDBO:2023] and $\sim 10^4$ datapoints.

@Hvarfner:VanilaBO:2024 recently disputed these well-entrenched
narratives by showing that poor GP fitting could be caused by a poor
choice of regularizer; mitigating the curse of dimensionality could be
as easy as including a dimensionality-dependent prior over lengthscales.
Furthermore, @Xu:StdVanillaBO:2024 argues that even the simplest BO
outperforms highly elaborate methods.

![Existing BO methods tackle problems with insufficiently low effective
dimensions. This figure shows sequence length and nr.of categories of
the highest search space in the original tests. For reference, the
discrete optimization problems usually tackled by practitioners in
chemistry and biology are of the order of $10^2$ in sequence length, and
$>10^1$ in nr.of categories. Methods that optimize directly in discrete
space (e.g.`BODi`, `ProbRep`, `Bounce`;
Sec. [3.7](#sec:taxonomy:structured-spaces){reference-type="ref"
reference="sec:taxonomy:structured-spaces"}) are tested in lower
sequence lengths and dictionary sizes; methods that rely on unsupervised
information (e.g.`LaMBO`, etc.;
Sec. [3.5](#sec:taxonomy:non-linear-embeddings){reference-type="ref"
reference="sec:taxonomy:non-linear-embeddings"}) are able to optimize
more complex problems, like protein engineering or small molecule
optimization.](figures/comparison_between_sequence_lengths_and_dict_sizes.jpg){#fig:overview-of-effective-dim
width="95%"}

**Optimization of discrete sequences & applications.**   Most HDBO
methods are tested on toy examples, hyperparameter tuning, or
reinforcement learning tasks
[@BinoisWycoff:HDGPs:2022; @Penubothula:FirstOrder:2021]. We focus on
discrete sequence optimization (DSOpt), which has several applications
beyond the usual examples (e.g.MaxSAT, or Pest Control)
[@Papenmeier:BOUNCE:2024], and is key in applications to biology and
bioinformatics
[@Bombarelli:AutoChemDesign:2018; @Stanton:LAMBO:2022; @Gruver:LAMBO2:2023].
Drug design and protein engineering can be thought of as DSOpt problems,
if we consider the SMILES/SELFIES representation of small molecules
[@Weininger:SMILES:1988; @Krenn:SELFIES:2020], or the amino acid
sequence representation of proteins [@Needleman:AminoAcids:1970].

**Related work.**   @BinoisWycoff:HDGPs:2022 initially surveyed the
field of high-dimensional GPs, focusing on applications to BO, and
proposed a taxonomy of structural assumptions for GPs that includes
variable selection, additive models, linear, and non-linear embeddings.
This work has since been updated by @Wang:BOAdvances:2023 and
@SantoniDoerr:HDBO:2023. The latter presents an empirical study on the
continuous, toy-problem setting up to 60 dimensions and refines the
taxonomy [@BinoisWycoff:HDGPs:2022] into five categories, separating
trust regions from the rest. @Griffiths:GAUCHE:2023 compare different
kernel functions used with binary vector representations of molecules
and for the same application @Kristiadi:LLM:2024 study the use of
different large language models. Our work is most similar to
@dreczkowski:mcbo:2023's comprehensive overview of discrete BO (`MCBO`),
and [@Gao:PMOMolOpt:2022]'s benchmark of small molecule optimization. In
both, HDBO is not in focus.