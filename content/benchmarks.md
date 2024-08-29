---
title: Benchmarks
description: 
type: minimal
layout: single_wide
---

We test several HDBO methods on discrete sequence optimization tasks. So far, we have optimized the [PMO benchmark](https://openreview.net/forum?id=yCZRdI0Y7G) on **SELFIES** representations, and also thermal stability of red fluorescent proteins and their mutations using [RaSP](https://elifesciences.org/articles/82593). Discrete solvers like `GeneticAlgorithm`, `Bounce` and `ProbRep` work directly on sequence space, while others rely on latent representations learned using autoencoders.

For details, check [our repository on GitHub](https://github.com/MachineLearningLifeScience/hdbo_benchmark).

{{< csv >}}