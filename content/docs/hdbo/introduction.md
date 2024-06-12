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
Optimizing discrete black-box functions is key in several domains, ranging from protein engineering to drug design. Due to the lack of gradient information and the need for sample efficiency, Bayesian optimization is an ideal candidate for these tasks. Several methods for categorical Bayesian optimization have been proposed recently, as well as a large collection of high-dimensional continuous alternatives. 

However, our survey of the field reveals highly heterogeneous experimental setups across methods and technical barriers for the replicability and application of published algorithms to real-world tasks. To address these issues, we implement a unified framework to test a vast array of high-dimensional Bayesian optimization methods, as well as a collection of standardized black-box functions representing real-world application domains in chemistry and biology. 

These two components of the benchmark are each supported by flexible, scalable, and easily extendable software libraries (poli and poli-baselines), allowing practitioners to readily incorporate new optimization objectives or discrete optimizers.

Read the full [paper on arXiv](https://arxiv.org/abs/2406.04739v1) or continue reading below.
{{< /card >}}

<!-- {{< button icon="fas file" color="secondary" tooltip="" href="" badge="arXiv" >}}
    Paper
{{< /button >}} -->

### Introduction

Optimizing an unknown and expensive-to-evaluate function is a frequent problem across disciplines [(Shahriari et al., 2016)]({{< relref "#fn:1" >}})&nbsp;[^1] examples are finding the right parameters for machine learning models or simulators, drug discovery or train scheduling, to name a few. In some scenarios, evaluating the black-box involves an expensive process (e.g. training a large model, or running a physical simulation); Bayesian Optimization 
[(BO, Močkus 1975)]({{< relref "#fn:2" >}})&nbsp;[^2] is a powerful method for sample efficient black-box optimization. High dimensional (discrete) problems have long been identified as a key challenge for Bayesian optimization algorithms 
([Wang et al., 2013]({{< relref "#fn:3" >}})&nbsp;[^3], [Snoek et al., 2012]({{< relref "#fn:4" >}})&nbsp;[^4]) given that they tend to scale poorly with both dataset size and dimensionality of the input.
{.px-4}

{{< svg "/static/img/HDBO_timeline.svg" "<strong>Figure 1:</strong> A timeline of high-dimensional Bayesian optimization methods, with arrows drawn between methods that explicitly augment or use each other. References are clickable." >}}

High-dimensional BO has been the focus of an entire research field (see [Figure 1]({{< relref "#svg" >}})), in which methods are extended to address the curse of dimensionality and its consequences (Binois and Wycoff, 2022; Santoni et al., 2023). Within this setting, discrete sequence optimization has received particular focus, due to its applicability in the optimization of molecules and proteins. However, prior work often focuses on sequence lengths and number of categories below the hundreds (see Fig. 2), making it difficult for practitioners to judge expected performance on real-world problems in these domains. We contribute (i) a survey of the field while focusing on the real-world applications of high-dimensional discrete sequences, (ii) a benchmark several optimizers in established black boxes, and (iii) an open source, unified interface: poli and poli-baselines.
{.px-4}

[^1]: Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. (2016). Taking the human out
of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148–175.

[^2]: Močkus, J. (1975). On bayesian methods for seeking the extremum. In Marchuk, G. I., editor,
Optimization Techniques IFIP Technical Conference Novosibirsk, July 1–7, 1974, page 400–404,
Berlin, Heidelberg. Springer.

[^3]: Wang

[^4]: Snoek
