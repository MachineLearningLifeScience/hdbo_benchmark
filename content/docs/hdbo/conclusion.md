---
layout: docs
type: minimal
modules: ["katex"]
---

# Conclusion

In this paper, we surveyed the field of high-dimensional Bayesian
optimization (HDBO) focusing on discrete problems. This highlighted the
need for (i) a novel taxonomy of the field that emphasizes the
differences between methods that rely on unsupervised discrete
information, and methods that optimize sequences directly, and (ii) a
standardized framework for benchmarking these methods. We approach these
in the form of two software tools:
[`poli`](https://github.com/MachineLearningLifeScience/poli) and
[`poli-baselines`](https://github.com/MachineLearningLifeScience/poli-baselines).
Using these tools, we implemented several HDBO methods and tested them
in a standard benchmark for small molecule optimization. We find that
optimizers using pre-trained latent-variable models have an edge over
the other tested methods that work directly on sequence space. Our
framework opens the door to fair and easily-replicable comparisons. We
expect `poli-baselines` to be used by practitioners for running HDBO
solvers in up-to-date environments compared across several tasks in our
ongoing benchmark, which we plan to expand to other discrete objectives
in `poli`.

**Limitations and societal impact.**   Although we taxonomize different
families of HDBO methods, we have only benchmarked a subset; moreover,
we only test a single setting: max. 300 iterations in PMO. This limits
the generalizability of our conclusions. That being said, our benchmark
is ongoing and we plan to include further experiments in the project's
website with, hopefully, participation from the community. Note that
optimizing small molecules can open the door to both the potential of
drug discovery, but also dual use [@Urbina:DualUse:2022].

::: ack
The work was partly funded by the Novo Nordisk Foundation through the
Center for Basic Machine Learning Research in Life Science
(NNF20OC0062606). RM is funded by the Danish Data Science Academy, which
is funded by the Novo Nordisk Foundation (NNF21SA0069429) and VILLUM
FONDEN (40516). SH was further supported by a research grant (42062)
from VILLUM FONDEN as well as funding from the European Research Council
(ERC) under the European Union's Horizon 2020 research and innovation
programme (grant agreement 757360). WB was supported by VILLUM FONDEN
(40578). This work was in part supported by the Pioneer Centre for AI
(DRNF grant number P1). MGD thanks Sergio Garrido and Anshuk Uppal for
feedback on early versions of this document, and Peter Mørch Groth for
useful discussions.
:::