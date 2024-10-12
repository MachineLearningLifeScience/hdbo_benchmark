from typing import Callable

import numpy as np

from poli.core.problem import Problem
from hdbo_benchmark.generative_models.vae import VAE, OptimizedVAE
from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.experiments.normalization import from_unit_cube_to_range


def _in_latent_space_of_proteins(
    problem: Problem,
    ae: LitAutoEncoder,
    latent_space_bounds: tuple[float, float],
) -> Callable[[np.ndarray], np.ndarray]:
    f = problem.black_box
    x0 = problem.x0

    def _latent_f(z: np.ndarray) -> np.ndarray:
        # We assume that z is in [0, 1]
        z = from_unit_cube_to_range(z, latent_space_bounds)
        protein_strings_w_special_tokens = ae.decode_to_string_array(z)
        protein_strings = [
            ["".join([c for c in p if c not in ["<pad>", "<cls>", "<eos>"]])]
            for p in protein_strings_w_special_tokens
        ]
        results = []
        valid_lengths = set([len("".join(x_i)) for x_i in x0])
        for p in protein_strings:
            if len(p[0]) not in valid_lengths:
                results.append(np.array([[-100.0]]))
            else:
                results.append(f(np.array([p])))
        val = np.array(results).reshape(z.shape[0], 1)
        return val

    _latent_f.info = f.info  # type: ignore[attr-defined]
    _latent_f.num_workers = f.num_workers  # type: ignore[attr-defined]
    _latent_f.observer = f.observer  # type: ignore[attr-defined]

    return _latent_f


def _in_the_latent_space_of_molecules(
    problem: Problem, vae: VAE, latent_space_bounds: tuple[float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    f = problem.black_box

    def _latent_f(z: np.ndarray) -> np.ndarray:
        z = from_unit_cube_to_range(z, latent_space_bounds)
        selfies_strings = vae.decode_to_string_array(z)
        val: np.ndarray = f(np.array(selfies_strings))
        return val

    _latent_f.info = f.info  # type: ignore[attr-defined]
    _latent_f.num_workers = f.num_workers  # type: ignore[attr-defined]
    _latent_f.observer = f.observer  # type: ignore[attr-defined]

    return _latent_f


def transform_problem_from_discrete_to_continuous(
    problem: Problem,
    generative_model: VAE | LitAutoEncoder,
    bounds: tuple[float, float],
) -> Problem:
    if isinstance(generative_model, LitAutoEncoder):
        # TODO: add z0
        continuous_f = _in_latent_space_of_proteins(
            problem=problem,
            ae=generative_model,
            latent_space_bounds=bounds,
        )
    elif isinstance(generative_model, (VAE, OptimizedVAE)):
        # TODO: add z0
        continuous_f = _in_the_latent_space_of_molecules(
            problem=problem,
            vae=generative_model,
            latent_space_bounds=bounds,
        )
    else:
        raise ValueError(
            f"The generative model must be either a LitAutoEncoder or a VAE. (Received {type(generative_model)})"
        )

    z0 = generative_model.encode_from_string_array(problem.x0)

    # TODO: continuous_f should be an AbstractBlackBox.
    return Problem(
        black_box=continuous_f,
        x0=z0,
        strict_validation=False,
    )
