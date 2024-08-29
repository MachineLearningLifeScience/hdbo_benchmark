---
title: Benchmarking HDBO
description: A survey and benchmark of high-dimensional Bayesian optimization of discrete sequences
date: 2024-01-01
layout: docs
aliases:
  - "/docs/about"
type: minimal
---

{{< card header="none" footer="none" padding="4" class="my-4" title="Abstract" >}}
Optimizing discrete black-box functions is key in several domains,
e.g. protein engineering and drug design. Due to the lack of gradient
information and the need for sample efficiency, Bayesian optimization
is an ideal candidate for these tasks. Several methods for
high-dimensional continuous and categorical Bayesian optimization have
been proposed recently. However, our survey of the field reveals
highly heterogeneous experimental set-ups across methods and technical
barriers for the replicability and application of published algorithms
to real-world tasks. To address these issues, we develop a unified
framework to test a vast array of high-dimensional Bayesian
optimization methods and a collection of standardized black-box
functions representing real-world application domains in chemistry and
biology. These two components of the benchmark are each supported by
flexible, scalable, and easily extendable software libraries (`poli`
and `poli-baselines`), allowing practitioners to readily incorporate
new optimization objectives or discrete optimizers.

Read the full [paper on arXiv](https://arxiv.org/abs/2406.04739v1) or continue reading below.
{{< /card >}}

<!-- {{< button icon="fas file" color="secondary" tooltip="" href="" badge="arXiv" >}}
    Paper
{{< /button >}} -->

### Introduction

Optimizing an unknown and expensive-to-evaluate function is a frequent
problem across disciplines ([Shahriari et al., 2016]({{< relref "#Shahriari2016BOreview" >}})): examples are
finding the right parameters for machine learning models or simulators,
drug discovery ([Gómez-Bombarelli et al., 2018]({{< relref "#Bombarelli:AutoChemDesign:2018" >}}) [Griffiths and Hernández-Lobato, 2020]({{< relref "#Griffiths:ConstrainedBOVAEs:2020" >}}) [Pyzer-Knapp, 2018]({{< relref "#Pyzer-Knapp:BODrugDiscovery:2018" >}})), protein design 
([Stanton et al., 2022]({{< relref "#Stanton:LAMBO:2022" >}}), 
[Gruver et al., 2023]({{< relref "#Gruver:LAMBO2:2023" >}})),
hyperparameter tuning in Machine Learning
([Snoek et al., 2012]({{< relref "#Snoek:PracticalBO:2012" >}});
[Turner et al., 2021]({{< relref "#Turner:BOHyperTuning:202" >}}))
and train
scheduling. In some scenarios, evaluating the black-box involves an
expensive process (e.g. training a large model, or running a physical
simulation); Bayesian Optimization (BO, [ Močkus (1975)]({{< relref "#Turner:Mockus:OriginalBO:1975" >}})) is a
powerful method for sample efficient black-box optimization. High
dimensional (discrete) problems have long been identified as a key
challenge for Bayesian optimization algorithms
([Wang et al., 2013]({{< relref "#Wang2013rembo" >}});
[Snoek et al., 2012]({{< relref "#Snoek:PracticalBO:2012" >}}))
given that they tend to scale
poorly with both dataset size and dimensionality of the input.

{{< svg "/static/img/HDBO_timeline.svg" "<strong>Figure 1:</strong> A timeline of high-dimensional Bayesian optimization methods, with arrows drawn between methods that explicitly augment or use each other. References are clickable." >}}

High-dimensional BO has been the focus of an entire research field (see [Figure 1]({{< relref "#svg" >}})), in which methods are extended to address the curse of dimensionality and its consequences 
([Binois and Wycoff, 2022]({{< relref "#BinoisWycoff:HDGPs:2022" >}});
[Santoni et al., 2023]({{< relref "#SantoniDoerr:HDBO:2023" >}})). Within this setting, discrete sequence optimization has received particular focus, due to its applicability in the optimization of molecules and proteins. However, prior work often focuses on sequence lengths and number of categories below the hundreds (see Fig. 2), making it difficult for practitioners to judge expected performance on real-world problems in these domains. We contribute (i) a survey of the field while focusing on the real-world applications of high-dimensional discrete sequences, (ii) a benchmark several optimizers in established black boxes, and (iii) an open source, unified interface: `poli` and `poli-baselines`.
{.pb-4}

##### References

###### Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. (2016). {#Shahriari2016BOreview} 
Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148–175. 
{.pb-3}

###### Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., and Aspuru-Guzik, A. (2018). {#Bombarelli:AutoChemDesign:2018}
Automatic chemical design using a data-driven continuous representation of molecules. ACS Central Science, 4(2):268–276. PMID: 29532027. 
{.pb-3}

###### Griffiths, R.-R. and Hernández-Lobato, J. M. (2020). {#Griffiths:ConstrainedBOVAEs:2020}
Constrained bayesian optimization for automatic chemical design using variational autoencoders. Chemical Science, 11(2):577–586.
{.pb-3}

###### Pyzer-Knapp, E. O. (2018). {#Pyzer-Knapp:BODrugDiscovery:2018}
Bayesian optimization for accelerated drug discovery. IBM Journal of Research and Development, 62(6):2:1–2:7
{.pb-3}

###### Stanton, S., Maddox, W., Gruver, N., Maffettone, P., Delaney, E., Greenside, P., and Wilson, A. G. (2022). {#Stanton:LAMBO:2022}
Accelerating Bayesian optimization for biological sequence design with denoising
autoencoders. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S.,
editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of
Proceedings of Machine Learning Research, pages 20459–20478. PMLR.
{.pb-3}

###### Gruver, N., Stanton, S., Frey, N., Rudner, T. G. J., Hotzel, I., Lafrance-Vanasse, J., Rajpal, A., Cho, K., and Wilson, A. G. (2023). {#Gruver:LAMBO2:2023}
Protein design with guided discrete diffusion. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S., editors, Advances in Neural Information
Processing Systems, volume 36, pages 12489–12517. Curran Associates, Inc
{.pb-3}
