"""
Running the benchmark on PMO.
"""

# mypy: disable-error-code="import-untyped"
from typing import Callable
from uuid import uuid4

import click

import torch
import numpy as np
from selfies import split_selfies

import poli
from poli.core.util.seeding import seed_numpy, seed_python
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.exceptions import BudgetExhaustedException

import poli_baselines

import hdbo_benchmark
from hdbo_benchmark.generative_models.vae_factory import VAEFactory, VAESelfies, VAE
from hdbo_benchmark.utils.experiments.load_solvers import (
    load_solver_class,
    SOLVER_NAMES,
)
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import (
    load_alphabet_for_pmo,
    load_sequence_length_for_pmo,
)
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE
from hdbo_benchmark.utils.logging.uncommited_changes import has_uncommitted_changes

from hdbo_benchmark.utils.logging.wandb_observer import initialize_observer
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)

torch.set_default_dtype(torch.float32)


def in_latent_space(
    f: AbstractBlackBox, vae: VAE
) -> Callable[[np.ndarray], np.ndarray]:
    def _latent_f(z: np.ndarray) -> np.ndarray:
        selfies_strings = vae.decode_to_string_array(z)
        val: np.ndarray = f(np.array(selfies_strings))
        return val

    _latent_f.info = f.info  # type: ignore[attr-defined]
    _latent_f.num_workers = f.num_workers  # type: ignore[attr-defined]

    return _latent_f


@click.command()
@click.option(
    "--solver-name",
    type=str,
    default="random_mutation",
    help=f"The name of the solver to run. All solvers available are: {SOLVER_NAMES}",
)
# TODO: add options for function name here using
# the PMO benchmark utilities from poli.
@click.option(
    "--function-name",
    type=str,
    default="albuterol_similarity",
    help="The name of the function to optimize.",
)
@click.option("--latent-dim", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--n-initial-points", type=int, default=10)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--solve-in-discrete-space/--no-solve-in-discrete-space", default=False)
@click.option("--tag", type=str, default="default")
def main(
    solver_name: str,
    function_name: str,
    latent_dim: int,
    seed: int,
    max_iter: int,
    n_initial_points: int,
    strict_on_hash: bool,
    force_run: bool,
    solve_in_discrete_space: bool,
    tag: str,
):
    print(f"Device: {DEVICE}")
    for module in [hdbo_benchmark, poli, poli_baselines]:
        if has_uncommitted_changes(module) and strict_on_hash:
            raise Exception(
                f"There are uncommitted changes in the repositories in {module.__name__}"
            )

    # Checking if this experimenr has already been run
    if (
        experiment_has_already_run(
            experiment_name="benchmark_on_pmo",
            solver_name=solver_name,
            function_name=function_name,
            n_dimensions=latent_dim,
            seed=seed,
        )
        and not force_run
    ):
        print(
            f"The experiment for solver {solver_name} with function "
            f"{function_name} and n_dimensions {latent_dim} "
            f" and seed {seed} has already been run."
        )
        return

    DATA_DIR = ROOT_DIR / "data" / "benchmark_on_pmo" / f"{solver_name}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = f"{uuid4()}"[:8]

    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_numpy(seed)
    seed_python(seed)
    torch.manual_seed(seed)

    # 2. Load a black box and solver
    problem = poli.create(
        name=function_name,
        string_representation="SELFIES",
        evaluation_budget=max_iter + n_initial_points,
    )
    f = problem.black_box

    obs = initialize_observer(
        experiment_name="benchmark_on_pmo",
        f=f,
        function_name=function_name,
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        experiment_id=experiment_id,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        tag=tag,
    )
    f.set_observer(obs)

    bounds = (-5.0, 5.0)
    vae: VAESelfies = VAEFactory().create(  # type: ignore[assignment]
        experiment_name="benchmark_on_pmo",
        latent_dim=latent_dim,
    )
    sobol_engine = torch.quasirandom.SobolEngine(
        dimension=latent_dim, scramble=True, seed=seed
    )
    solvers_that_dont_allow_eager_evaluation = ["bounce"]
    if solver_name in solvers_that_dont_allow_eager_evaluation:
        z0 = None
        x0 = None
        y0 = None
    else:
        z0 = sobol_engine.draw(n_initial_points).to(DEVICE).numpy(force=True)
        x0 = vae.decode_to_string_array(z0)
        y0 = f(x0)

    if solve_in_discrete_space:
        f_ = f
        alphabet = load_alphabet_for_pmo(n_dimensions=latent_dim)
        sequence_length = load_sequence_length_for_pmo(n_dimensions=latent_dim)
        # We need to split x0 into [b, L] tokens.
        if x0 is not None:
            split_x0 = [list(split_selfies(x_i)) for x_i in x0]
            split_x0 = [
                x_i + ["[nop]"] * (sequence_length - len(x_i)) for x_i in split_x0
            ]
            x0_for_solver = np.array(split_x0)
            # x0_for_solver = x0
        if x0 is None:
            x0_for_solver = None
        kwargs_ = {
            "alphabet": alphabet,
            "sequence_length": sequence_length,
        }
    else:
        f_ = in_latent_space(f, vae)
        x0_for_solver = z0
        kwargs_ = {
            "bounds": bounds,
        }

    solver_, kwargs = load_solver_class(
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        n_initial_points=n_initial_points,
        **kwargs_,
    )
    kwargs.update(kwargs_)
    solver = solver_(
        black_box=f_,
        x0=x0_for_solver,
        y0=y0,
        **kwargs,
    )

    # 3. Optimize
    try:
        solver.solve(max_iter=max_iter)
    except (KeyboardInterrupt, BudgetExhaustedException):
        pass


if __name__ == "__main__":
    main()
