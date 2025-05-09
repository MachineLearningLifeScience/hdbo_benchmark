"""Solvers for the benchmark on toy objective functions."""

# mypy: disable-error-code="import-untyped"
from typing import Any, Dict, Tuple

import numpy as np
import torch
from poli.core.problem import Problem
from poli_baselines.core.abstract_solver import AbstractSolver

from hdbo_benchmark.utils.constants import DEVICE

SOLVER_NAMES = [
    "directed_evolution",
    "hill_climbing",
    "genetic_algorithm",
    "cma_es",
    "random_line_bo",
    "coordinate_line_bo",
    "baxus",
    "turbo",
    "vanilla_bo_hvarfner",
    "alebo",
    "bounce",
    "pr",
    "saas_bo",
    "lambo2",
]

SOLVERS_THAT_DONT_ALLOW_CUSTOM_INPUTS = [
    "bounce",
    "baxus",
]

DISCRETE_SPACE_SOLVERS = [
    "directed_evolution",
    "pr",
    "bounce",
    "lambo2",
    "genetic_algorithm",
]

CONTINUOUS_SPACE_SOLVERS = [
    solver for solver in SOLVER_NAMES if solver not in DISCRETE_SPACE_SOLVERS
]

SOLVER_NAME_TO_ENV = {
    "directed_evolution": "hdbo_benchmark",
    "hill_climbing": "hdbo_benchmark",
    "genetic_algorithm": "hdbo_benchmark",
    "cma_es": "hdbo_benchmark",
    "random_line_bo": "hdbo_benchmark",
    "coordinate_line_bo": "hdbo_benchmark",
    "baxus": "hdbo_baxus",
    "turbo": "hdbo_benchmark",
    "vanilla_bo_hvarfner": "hdbo_ax",
    "alebo": "hdbo_alebo",
    "bounce": "hdbo_bounce",
    "pr": "hdbo_pr",
    "saas_bo": "hdbo_ax",
    "lambo2": "hdbo_lambo2",
}


def load_solver_class(
    solver_name: str,
    seed: int | None = None,
    n_dimensions: int | None = None,
    n_intrinsic_dimensions: int | None = None,
    max_iter: int | None = None,
    noise_std: float = 0.0,
    std: float = 0.25,
    n_initial_points: int = 10,
    **solver_kwargs,
) -> Tuple[AbstractSolver, Dict[str, Any]]:
    if solver_name in DISCRETE_SPACE_SOLVERS:
        solver_kwargs.pop("bounds", None)
        solver_kwargs.pop("std", None)

    if solver_name in CONTINUOUS_SPACE_SOLVERS:
        solver_kwargs.pop("sequence_length", None)
        solver_kwargs.pop("alphabet", None)

    match solver_name:
        case "directed_evolution":
            from poli_baselines.solvers.simple.random_mutation import RandomMutation

            solver_kwargs.pop("bounds", None)
            solver_kwargs.pop("sequence_length", None)

            return RandomMutation, solver_kwargs
        case "hill_climbing":
            from poli_baselines.solvers.simple.continuous_random_mutation import (
                ContinuousRandomMutation,
            )

            solver_kwargs.update(
                {
                    "std": std,
                }
            )

            return ContinuousRandomMutation, solver_kwargs
        case "genetic_algorithm":
            from poli_baselines.solvers.simple.genetic_algorithm import (
                FixedLengthGeneticAlgorithm,
            )

            solver_kwargs.update(
                {
                    "population_size": n_initial_points,
                    "prob_of_mutation": 0.25,
                }
            )
            solver_kwargs.pop("bounds", None)

            return FixedLengthGeneticAlgorithm, solver_kwargs
        case "vanilla_bo":
            from poli_baselines.solvers.bayesian_optimization.vanilla_bayesian_optimization import (
                VanillaBayesianOptimization,
            )

            return VanillaBayesianOptimization, solver_kwargs
        case "vanilla_bo_with_lognormal_prior":
            import gpytorch
            from poli_baselines.solvers.bayesian_optimization.vanilla_bayesian_optimization import (
                VanillaBayesianOptimization,
            )

            assert n_dimensions is not None
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=n_dimensions,
                    lengthscale_prior=gpytorch.priors.LogNormalPrior(
                        np.log(n_dimensions) / 2, 1.0
                    ),
                )
            )
            solver_kwargs.update(
                {
                    "kernel": kernel,
                }
            )

            return VanillaBayesianOptimization, solver_kwargs
        case "vanilla_bo_hvarfner":
            from poli_baselines.solvers.bayesian_optimization.vanilla_bo_hvarfner import (
                VanillaBOHvarfner,
            )

            solver_kwargs.update(
                {
                    "device": DEVICE,
                    "bounds": solver_kwargs.get("bounds", [0.0, 1.0]),
                }
            )

            return VanillaBOHvarfner, solver_kwargs
        case "random_line_bo":
            from poli_baselines.solvers.bayesian_optimization.line_bayesian_optimization import (
                LineBO,
            )

            solver_kwargs.update(
                {
                    "type_of_line": "random",
                }
            )

            return LineBO, solver_kwargs
        case "coordinate_line_bo":
            from poli_baselines.solvers.bayesian_optimization.line_bayesian_optimization import (
                LineBO,
            )

            solver_kwargs.update(
                {
                    "type_of_line": "coordinate",
                }
            )

            return LineBO, solver_kwargs
        case "saas_bo":
            from poli_baselines.solvers.bayesian_optimization.saasbo import SAASBO

            solver_kwargs.update(
                {
                    "noise_std": noise_std,
                    "device": DEVICE,
                    "bounds": solver_kwargs.get("bounds", [0.0, 1.0]),
                }
            )

            return SAASBO, solver_kwargs
        case "alebo":
            from poli_baselines.solvers.bayesian_optimization.alebo import ALEBO

            if n_intrinsic_dimensions is None:
                n_intrinsic_dimensions = n_dimensions // 2
            solver_kwargs.update(
                {
                    "lower_dim": n_intrinsic_dimensions,
                    "noise_std": noise_std,
                    "device": DEVICE,
                }
            )

            return ALEBO, solver_kwargs
        case "cma_es":
            from poli_baselines.solvers.evolutionary_strategies.cma_es import CMA_ES

            assert n_dimensions is not None
            solver_kwargs.update(
                {
                    "initial_mean": np.random.randn(n_dimensions)
                    .reshape(1, -1)
                    .clip(*solver_kwargs.get("bounds", [0.0, 1.0])),
                    "population_size": 10,
                    "initial_sigma": 1.0,
                }
            )

            return CMA_ES, solver_kwargs
        case "baxus":
            from poli_baselines.solvers.bayesian_optimization.baxus import BAxUS

            solver_kwargs.update(
                {
                    # "initial_trust_region_length": 0.8 * 4,
                    "noise_std": noise_std,
                    "n_dimensions": n_dimensions,
                    "n_init": n_initial_points,
                    "max_iter": max_iter,
                    "bounds": solver_kwargs.get("bounds", [0.0, 1.0]),
                }
            )

            return BAxUS, solver_kwargs
        case "turbo":
            from poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper import (
                Turbo,
            )

            torch.set_default_dtype(torch.float64)

            return Turbo, solver_kwargs
        case "bounce":
            from poli_baselines.solvers.bayesian_optimization.bounce import BounceSolver

            torch.set_default_dtype(torch.float64)

            solver_kwargs.update(
                {
                    "noise_std": noise_std,
                    "n_initial_points": n_initial_points,
                    "device": DEVICE,
                    "dtype": "float64",
                }
            )
            solver_kwargs.pop("bounds", None)
            return BounceSolver, solver_kwargs
        case "pr":
            from poli_baselines.solvers.bayesian_optimization.pr import (
                ProbabilisticReparametrizationSolver,
            )

            solver_kwargs.update(
                {
                    "seed": seed,
                    "n_initial_points": n_initial_points,
                    "device": DEVICE,
                }
            )
            solver_kwargs.pop("bounds", None)

            return ProbabilisticReparametrizationSolver, solver_kwargs
        case "lambo2":
            from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

            # TODO: write these out.
            solver_kwargs.update(
                {
                    "device": DEVICE,
                }
            )
            solver_kwargs.pop("bounds", None)

            return LaMBO2, solver_kwargs
        case _:
            raise ValueError(f"Unknown solver {solver_name}")


def load_solver_from_problem(
    solver_name: str,
    problem: Problem,
    seed: int | None = None,
):
    f, data_package = problem.black_box, problem.data_package
    x0, y0 = data_package.supervised_data

    if len(x0.shape) == 1:
        n_dimensions = None
    else:
        n_dimensions = x0.shape[1]

    solver_, kwargs = load_solver_class(
        solver_name=solver_name,
        seed=seed,
        n_dimensions=n_dimensions,
        n_initial_points=x0.shape[0],
    )

    return solver_(
        black_box=f,
        x0=x0,
        y0=y0,
        **kwargs,
    )
