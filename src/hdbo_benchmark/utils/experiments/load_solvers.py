"""Solvers for the benchmark on toy objective functions."""

# mypy: disable-error-code="import-untyped"
from typing import Tuple, Dict, Any

import numpy as np

from hdbo_benchmark.utils.constants import DEVICE

SOLVER_NAMES = [
    "random_mutation",
    "genetic_algorithm",
    "vanilla_bo",
    "vanilla_bo_with_lognormal_prior",
    "vanilla_bo_hvarfner",
    "line_bo",
    "saas_bo",
    "alebo",
    "cma_es",
    "baxus",
    "turbo",
    "bounce",
    "pr",
]


def load_solver(
    solver_name: str,
    seed: int | None = None,
    n_dimensions: int | None = None,
    n_intrinsic_dimensions: int | None = None,
    upper_bound: float | None = None,
    lower_bound: float | None = None,
    max_iter: int | None = None,
    noise_std: float = 0.0,
    n_initial_points: int = 10,
    **solver_kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    match solver_name:
        case "random_mutation":
            from poli_baselines.solvers.simple.continuous_random_mutation import (
                ContinuousRandomMutation,
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
            from poli_baselines.solvers.bayesian_optimization.vanilla_bayesian_optimization import (
                VanillaBayesianOptimization,
            )
            import gpytorch

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
                }
            )

            return VanillaBOHvarfner, solver_kwargs
        case "line_bo":
            from poli_baselines.solvers.bayesian_optimization.line_bayesian_optimization import (
                LineBO,
            )

            solver_kwargs.update(
                {
                    "type_of_line": "random",
                }
            )

            return LineBO, solver_kwargs
        case "saas_bo":
            from poli_baselines.solvers.bayesian_optimization.saasbo import SAASBO

            solver_kwargs.update(
                {
                    "noise_std": noise_std,
                    "device": DEVICE,
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
                    .clip(*solver_kwargs.get("bounds", [None, None])),
                    "population_size": n_initial_points,
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
                    "max_iter": max_iter,
                }
            )

            return BAxUS, solver_kwargs
        case "turbo":
            from poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper import (
                Turbo,
            )

            return Turbo, solver_kwargs
        case "bounce":
            from poli_baselines.solvers.bayesian_optimization.bounce import BounceSolver

            solver_kwargs.update(
                {
                    "noise_std": noise_std,
                    "n_initial_points": n_initial_points,
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
        case _:
            raise ValueError(f"Unknown solver {solver_name}")
