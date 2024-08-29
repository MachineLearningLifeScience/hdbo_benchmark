## Rasp (32-dim latent space)

In this experiment, we optimize a of pool Red Fluorescent Proteins (RFP) and their mutations to maximize their thermal stability according to `RaSP`, a neural-network proxy for Rosetta scores in single mutations.

To do so, we start by computing the [ESM-2](https://github.com/facebookresearch/esm) embeddings of [the initial pool of high-performing RFPs](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/fpbase/proxy_rfp_seed_data.csv) and their mutations maintained by the authors of [LaMBO](https://github.com/samuelstanton/lambo) and training a simple autoencoder from the ESM-2 back to sequence space. This neural network now provides a decoder from a 32-dimensional space back to protein sequences.

Therein we test several high-dimensional Bayesian Optimization methods, optimizing [our additive version of `RaSP`](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/RaSP.html). A table of ongoing results is presented here:


