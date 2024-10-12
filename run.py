"""
This script is an entry-point for all experiments.

For documentation on how to run each individual experiment,
please refer to the README.md.
"""

# mypy: disable-error-code="import-untyped"
from uuid import uuid4

import click
import numpy as np

from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.core.exceptions import BudgetExhaustedException


from hdbo_benchmark.utils.experiments.load_solvers import (
    load_solver_from_problem,
    SOLVER_NAMES,
    CONTINUOUS_SPACE_SOLVERS,
)
from hdbo_benchmark.utils.experiments.load_problems import load_problem
from hdbo_benchmark.utils.experiments.load_generative_models import (
    load_generative_model_and_bounds,
)
from hdbo_benchmark.utils.experiments.verify_status_pre_experiment import (
    verify_repos_are_clean,
)
from hdbo_benchmark.utils.experiments.problem_transformations import (
    transform_problem_from_discrete_to_continuous,
)

from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)
from hdbo_benchmark.utils.logging.wandb_observer import ObserverConfig


@click.command()
@click.option(
    "--function-name",
    type=str,
    default="foldx_stability",
    help="The name of the objective function to optimize.",
)
@click.option(
    "--solver-name",
    type=str,
    default="directed_evolution",
    help=f"The name of the solver to run. All solvers available are: {SOLVER_NAMES}",
)
@click.option("--n-dimensions", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--wandb-mode", type=str, default="online")
@click.option("--tag", type=str, default="default")
def main(
    function_name: str,
    solver_name: str,
    n_dimensions: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    wandb_mode: str,
    tag: str,
):
    # Defining a unique experiment id
    experiment_id = f"{uuid4()}"

    # Checking if there are uncommitted changes in the repositories
    verify_repos_are_clean(strict_on_hash)

    # Checking if this experimenr has already been run
    if (
        experiment_has_already_run(
            experiment_name="hdbo_benchmark_results",
            solver_name=solver_name,
            function_name=function_name,
            n_dimensions=n_dimensions,
            seed=seed,
        )
        and not force_run
    ):
        print(
            f"The experiment for solver {solver_name} with function "
            f"rasp and n_dimensions {n_dimensions} "
            f" and seed {seed} has already been run."
        )
        return

    # Seeding
    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_python_numpy_and_torch(seed)

    # Setting the observer configuration
    observer_config = ObserverConfig(
        experiment_name="hdbo_benchmark_results",
        function_name=function_name,
        solver_name=solver_name,
        n_dimensions=n_dimensions,
        seed=seed,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        force_run=force_run,
        experiment_id=experiment_id,
        wandb_mode=wandb_mode,
        tags=[tag],
    )

    # Load the problem
    problem = load_problem(
        function_name=function_name,
        max_iter=max_iter,
        set_observer=True,
        observer_config=observer_config,
    )
    print(problem)

    if solver_name in CONTINUOUS_SPACE_SOLVERS:
        # Load the generative model
        generative_model, bounds = load_generative_model_and_bounds(
            function_name=function_name,
            latent_dim=n_dimensions,
        )

        # Make the problem continuous
        problem = transform_problem_from_discrete_to_continuous(
            problem, generative_model, bounds
        )

    # load the solver
    solver = load_solver_from_problem(
        solver_name=solver_name,
        problem=problem,
        seed=seed,
    )
    print(solver)

    # 3. Optimize
    try:
        solver.solve(max_iter=max_iter)
    except KeyboardInterrupt:
        print("Interrupted optimization.")
    except BudgetExhaustedException:
        print("Budget exhausted.")

    problem.black_box.observer.finish()


if __name__ == "__main__":
    main()
