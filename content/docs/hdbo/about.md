---
title: BenchmarkingHDBO
description: A survey and benchmark of high-dimensional Bayesian optimization of discrete sequences
date: 2024-01-01
layout: docs
aliases:
  - "/docs/about"
  - "/license/"
type: minimal
---

Optimizing discrete black-box functions is key in several domains, ranging from
protein engineering to drug design. Due to the lack of gradient information and the
need for sample efficiency, Bayesian optimization is an ideal candidate for these
tasks. Several methods for categorical Bayesian optimization have been proposed
recently, as well as a large collection of high-dimensional continuous alternatives.
However, our survey of the field reveals highly heterogeneous experimental set-
ups across methods and technical barriers for the replicability and application of
published algorithms to real-world tasks. To address these issues, we implement a
unified framework to test a vast array of high-dimensional Bayesian optimization
methods, as well as a collection of standardized black-box functions representing
real-world application domains in chemistry and biology. These two components
of the benchmark are each supported by flexible, scalable, and easily extendable
software libraries (poli and poli-baselines), allowing practitioners to readily
incorporate new optimization objectives or discrete optimizers.

{{< svg "/static/img/HDBO_timeline.svg" "aaa" >}}

<!-- > Don't communicate by sharing memory, share memory by communicating.
>
> — _Rob Pike[^1]_
{.blockquote}

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015. -->
