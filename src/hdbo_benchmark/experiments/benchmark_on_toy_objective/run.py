"""
Runs the benchmark experiment for the toy objective function.
"""

from uuid import uuid4

import click

import numpy as np
import torch

import poli  # type: ignore
import poli_baselines  # type: ignore
from poli.objective_repository import ToyContinuousBlackBox  # type: ignore
from poli.core.util.seeding import seed_numpy, seed_python  # type: ignore

import hdbo_benchmark
from hdbo_benchmark.utils.experiments.load_solvers import (
    load_solver,
    SOLVER_NAMES,
)
from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.utils.logging import (
    get_git_hash_of_library,
    WandbObserver,
    has_uncommitted_changes,
)


@click.command()
@click.option(
    "--solver-name",
    type=str,
    default="random_search",
    help=f"The name of the solver to run. One of {SOLVER_NAMES}.",
)
@click.option("--function-name", type=str, default="ackley_function_01")
@click.option("--n-dimensions", type=int, default=10)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=200)
@click.option("--initial-sample-size", type=int, default=1)
@click.option("--n-intrinsic-dimensions", type=int, default=None)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
def run(
    solver_name: str,
    function_name: str,
    n_dimensions: int,
    seed: int,
    max_iter: int,
    initial_sample_size: int,
    n_intrinsic_dimensions: int,
    strict_on_hash: bool,
):
    for module in [hdbo_benchmark, poli, poli_baselines]:
        if has_uncommitted_changes(module) and strict_on_hash:
            raise Exception(
                f"There are uncommitted changes in the repositories in {module.__name__}"
            )

    DATA_DIR = ROOT_DIR / "data" / "benchmark_on_toy_objective" / f"{solver_name}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{uuid4()}"[:8]

    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_python(seed)
    seed_numpy(seed)
    torch.manual_seed(seed)

    f = ToyContinuousBlackBox(
        function_name=function_name,
        n_dimensions=n_dimensions,
    )

    wandb_observer = WandbObserver(
        project_name="benchmark_on_toy_objective", allow_reinit=True
    )
    wandb_observer.initialize_observer(
        black_box_info=f.info,
        observer_init_info={
            "run_name": f"{f.function_name}-{solver_name}-n_dimensions-{n_dimensions}-seed-{seed}-exp_id-{experiment_id}",
            "solver_name": solver_name,
        },
        x0=np.array([[]]),
        y0=np.array([[]]),
        seed=seed,
        n_dimensions=n_dimensions,
        function_name=f.function_name,
        hdbo_benchmark_hash=get_git_hash_of_library(hdbo_benchmark),
        poli_hash=get_git_hash_of_library(poli),
        poli_baselines_hash=get_git_hash_of_library(poli_baselines),
    )

    f.set_observer(wandb_observer)

    # Initializing the solver
    lower_bound, upper_bound = [f_ for f_ in f.bounds]
    x0 = np.random.uniform(
        lower_bound, upper_bound, n_dimensions * initial_sample_size
    ).reshape(initial_sample_size, n_dimensions)
    y0 = f(x0)

    if n_intrinsic_dimensions is None:
        n_intrinsic_dimensions = n_dimensions - 1

    solver_, kwargs = load_solver(
        solver_name,
        n_dimensions=n_dimensions,
        n_intrinsic_dimensions=n_intrinsic_dimensions,
        bounds=f.bounds,
    )
    solver = solver_(
        black_box=f,
        x0=x0,
        y0=y0,
        **kwargs,
    )

    # Running the solver
    if "population_size" in kwargs:
        max_iter = max_iter // kwargs["population_size"]

    solver.solve(max_iter=max_iter, verbose=True)

    # Saving results
    DATA_DIR_FOR_N_DIM = DATA_DIR / f"{f.function_name}_n_dimensions_{n_dimensions}"
    DATA_DIR_FOR_N_DIM.mkdir(parents=True, exist_ok=True)
    solver.save_history(
        DATA_DIR_FOR_N_DIM / f"{solver_name}_{experiment_id}.json",
        metadata={
            "seed": seed,
            "solver": solver_name,
            "n_dimensions": n_dimensions,
            "experiment_id": str(experiment_id),
            "function_name": f.function_name,
            "x0": x0.tolist(),
            "y0": y0.tolist(),
            "kwargs": kwargs,
        },
    )


if __name__ == "__main__":
    run()
