from typing import Callable

import numpy as np

from poli.core.problem import Problem
from poli.core.black_box_information import BlackBoxInformation
from poli.core.lambda_black_box import LambdaBlackBox
from poli.core.data_package import DataPackage
from hdbo_benchmark.generative_models.vae import VAE, OptimizedVAE
from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.generative_models.onehot import OneHot
from hdbo_benchmark.utils.experiments.normalization import (
    from_unit_cube_to_range,
    from_range_to_unit_cube,
)


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

    new_black_box = LambdaBlackBox(
        function=_latent_f,
        info=BlackBoxInformation(
            name=f.info.name,
            max_sequence_length=ae.latent_dim,
            aligned=True,
            fixed_length=True,
            deterministic=f.info.deterministic,
            alphabet=None,
            discrete=False,
        ),
    )

    return new_black_box


def _in_the_latent_space_of_molecules(
    problem: Problem, vae: VAE, latent_space_bounds: tuple[float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    f = problem.black_box

    def _latent_f(z: np.ndarray) -> np.ndarray:
        z = from_unit_cube_to_range(z, latent_space_bounds)
        selfies_strings = vae.decode_to_string_array(z)
        val: np.ndarray = f(np.array(selfies_strings))
        return val

    new_black_box = LambdaBlackBox(
        function=_latent_f,
        info=BlackBoxInformation(
            name=f.info.name,
            max_sequence_length=vae.latent_dim,
            aligned=True,
            fixed_length=True,
            deterministic=f.info.deterministic,
            alphabet=None,
            discrete=False,
        ),
    )

    return new_black_box


def _in_onehot_space(
    problem: Problem, onehot: OneHot, latent_space_bounds: tuple[float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    f = problem.black_box

    def _latent_f(z: np.ndarray) -> np.ndarray:
        z = from_unit_cube_to_range(z, latent_space_bounds)
        x = onehot.decode_to_string_array(z)
        val = f(x)
        return val

    new_black_box = LambdaBlackBox(
        function=_latent_f,
        info=BlackBoxInformation(
            name=f.info.name,
            max_sequence_length=onehot.max_sequence_length * onehot.n_classes,
            aligned=True,
            fixed_length=True,
            deterministic=f.info.deterministic,
            alphabet=None,
            discrete=False,
        ),
    )

    return new_black_box


def transform_problem_from_discrete_to_continuous(
    problem: Problem,
    generative_model: VAE | LitAutoEncoder | OneHot,
    bounds: tuple[float, float],
) -> Problem:
    if isinstance(generative_model, LitAutoEncoder):
        continuous_f = _in_latent_space_of_proteins(
            problem=problem,
            ae=generative_model,
            latent_space_bounds=bounds,
        )
    elif isinstance(generative_model, (VAE, OptimizedVAE)):
        continuous_f = _in_the_latent_space_of_molecules(
            problem=problem,
            vae=generative_model,
            latent_space_bounds=bounds,
        )
    elif isinstance(generative_model, OneHot):
        continuous_f = _in_onehot_space(
            problem=problem,
            onehot=generative_model,
            latent_space_bounds=bounds,
        )
    else:
        raise ValueError(
            f"The generative model must be either a LitAutoEncoder or a VAE. (Received {type(generative_model)})"
        )

    z0_ = generative_model.encode_from_string_array(problem.x0)
    z0 = from_range_to_unit_cube(z0_, bounds)

    if problem.data_package is not None:
        data_package = problem.data_package
        if data_package.unsupervised_data is not None:
            new_unsupervised_data_ = generative_model.encode_from_string_array(
                data_package.unsupervised_data
            )
            new_unsupervised_data = from_range_to_unit_cube(
                new_unsupervised_data_, bounds
            )

        if data_package.supervised_data is not None:
            x_, y_ = data_package.supervised_data

            if x_ is not None:
                z_ = generative_model.encode_from_string_array(x_)
                z = from_range_to_unit_cube(z_, bounds)

        new_data_package = DataPackage(
            unsupervised_data=new_unsupervised_data, supervised_data=(z, y_)
        )
    else:
        new_data_package = problem.data_package

    return Problem(
        black_box=continuous_f,
        x0=z0,
        data_package=new_data_package,
    )
