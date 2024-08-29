---
layout: docs
type: minimal
modules: ["katex"]
---

# Benchmarking the performance of HDBO methods

![Initialization, evaluation budget, and nr.of replications using
different seeds reported in the experimental set-ups of several HDBO
methods. We see heterogeneity in the evaluation of
optimizers.](figures/comparison_between_experimental_setups.jpg){#fig:results:different-experimental-setups
width="1.0\\columnwidth"}

Practitioners that decide on what Bayesian optimization algorithm to use
for their application will face several challenges. While surveying the
field, we noticed two key discrepancies in the reported experimental
set-ups: (i) the initialization varies from as low as *none* to *five*
randomly/SOBOL sampled points to over $10^3$, (ii) evaluation budgets
also vary for the same types of tasks.
Fig. [3](#fig:results:different-experimental-setups){reference-type="ref"
reference="fig:results:different-experimental-setups"} visualizes these
different experimental set-ups as swarmplots. Moreover, our survey
covered code availability. The state-of-the-art is being pulled by
workhorses, which have democratized access to GP and BO implementations:
`GPyTorch` [@Gardner:Gpytorch:2018] and `BoTorch`
[@Balandat:BoTorch:2020], and `GPFlow`
[@Matthews2017gpflow; @Wilk2020gpflow] and `Trieste`
[@Picheny2023trieste]. These libraries are highly useful and impactful,
yet one can obtain cross-dependency conflicts between them especially if
third-party dependencies are introduced or if very specific versions are
required for solver setups. As a particular example, solvers like
`ProbRep` cannot co-exist with `Ax`-based solvers like `SAASBO` or
Hvarfner's `VanillaBO`. There is a need for *isolating* optimizers,
specifying up-to-date environments in which they can run. These issues
led to the development of `poli`.

## `poli` and `poli-baselines`: a framework for benchmarking discrete optimizers

We want to solve truly high dimensional problems that are relevant for
domains like biology and chemistry. To make the outcomes comparable, we
require a unified way of defining the problem which includes consistent
starting points, budgets, runtime environments, relevant assets
(i.e.used for the black-box), and a logging backend invoked for every
oracle observation. To that end, we provide the *Protein Objectives
Library* (`poli`) as a means to provide isolated black-box functions.
Building on open source tools, `poli` currently provides 35 black-box
tasks; besides the Practical Molecular Optimization (PMO) benchmark
[@Huang:TDC:2021; @Gao:PMOMolOpt:2022; @Brown:Guacamol:2019], it
includes `Dockstring` [@Garcia:DOCKSTRING:2022] as well as other
protein-related black boxes like stability and solvent accessibility
[@Delgado:FOLDX5:2019; @Blaabjerg:RASP:2023; @Chapman:Biopython:2000; @Stanton:LAMBO:2022].
The majority of black-box functions can be queried with any string that
complies with the corresponding alphabet making the oracles available
for free-form optimization. This is an important distinction compared to
pre-existing benchmarks that rely on pools of precompiled observations
[@Notin:ProtGym:2023]. We further provide an interface for the solvers
used for the individual optimization tasks: `poli-baselines`.
Consistent, stable (and up to date) environments of individual
optimizers can be found therein, as well as a standardized way to query
them and solve the problems raised in the previous section. These
environments and optimizers are tested weekly through GitHub actions,
guaranteeing their usability.
Sec. [6.4](#sec:appendix:technical-details-on-poli-and-poli-baselines){reference-type="ref"
reference="sec:appendix:technical-details-on-poli-and-poli-baselines"}
provides a broader introduction to this software's technical
details.[^4]

## Benchmarking HDBO on PMO

To benchmark the performance of HDBO on discrete sequences, we focus on
the PMO benchmark
[@Gao:PMOMolOpt:2022; @Huang:TDC:2021; @Brown:Guacamol:2019].[^5] From
the taxonomy (Section [3](#sec:taxonomy){reference-type="ref"
reference="sec:taxonomy"}) we select frequently-tested methods from
several families: Hvarfner's `VanillaBO`, `RandomLineBO`, `Turbo`,
`BAxUS`, `SAASBO`, `Bounce`, and `ProbRep`. We also include a
`HillClimbing` baseline, which explores the input space by taking random
Gaussian steps. All continuous solvers we start from the same initial
data to ensure a fair comparison, and the discrete solvers (`Bounce` and
`ProbRep`) initialize according to their implementations. We test the
aforementioned methods on PMO [@Gao:PMOMolOpt:2022; @Huang:TDC:2021],
which requires a discrete representation of small molecules. Thus, we
train two MLP VAEs on SELFIES representations of small molecules using
Zinc250k [@Irwin:ZINC20:2020; @Zhu:TorchDrug:2022]. These generative
models had 2 and 128 latent dimensions, allowing us to get an impression
of how these models scale with dimensionality. We restrict sequences to
be of length 70 (adding `[nop]` tokens for padding); the post-processing
renders an alphabet of 64 SELFIES tokens. Details can be found in
Sec. [6.3](#sec:appendix:training-vaes-on-selfies){reference-type="ref"
reference="sec:appendix:training-vaes-on-selfies"}.

The average best result over 3 runs (of maximum 300 iterations each) is
presented in
Tables [\[tab:results:absolute_values_for_128_latent_dim\]](#tab:results:absolute_values_for_128_latent_dim){reference-type="ref"
reference="tab:results:absolute_values_for_128_latent_dim"} and
 [\[tab:results:absolute_values_for_2_dim_latent_space\]](#tab:results:absolute_values_for_2_dim_latent_space){reference-type="ref"
reference="tab:results:absolute_values_for_2_dim_latent_space"} for 128D
and 2D latent spaces respectively. We see a clear advantage in the
optimizers that work on learned representations, instead of in discrete
space. Such a discrepancy is to be expected: methods that optimize in
latent space have been presented with information prior to their
optimization campaigns, while methods like `Bounce` and `ProbRep`
explore the whole discrete space. Further, the simple baseline is
reliably beaten by the continuous alternatives in lower dimensions
except `Turbo`, but this advantage is not as clear in the 128D case,
signaling a more complex problem. Some of these tasks, however, are
equally challenging for all solvers. `deco_hop` remains close to the
original default value of 0.5, and there is no improvement over
`valsartan_smarts` (which only REINVENT improves on in the original PMO
results [@Gao:PMOMolOpt:2022; @Loeffler:REINVENT:2024]). `SAASBO` did
not scale gracefully with dimension in terms of training time; results
are presented for the 2D case, and they are pending for 128D. This
phenomenon has also been reported in comparisons made by
@Papenmeier:BAxUS:2022 and @Hvarfner:VanilaBO:2024. At the time of
writing, these results are not comparable with PMO due to changes in
`scikit-learn`'s loading of oracles. This issue has been raised in the
TDC repository.[^6]