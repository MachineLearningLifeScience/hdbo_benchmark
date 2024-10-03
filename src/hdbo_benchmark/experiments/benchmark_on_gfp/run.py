from typing import Callable
from uuid import uuid4

import click
import numpy as np
import poli  # type: ignore[import]
import poli_baselines  # type: ignore[import]
import torch
from poli import objective_factory  # type: ignore[import]
from poli.core.abstract_black_box import \
    AbstractBlackBox  # type: ignore[import]
from poli.core.exceptions import \
    BudgetExhaustedException  # type: ignore[import]
from poli.core.util.seeding import seed_numpy  # type: ignore[import]
from poli.core.util.seeding import seed_python

import hdbo_benchmark
from hdbo_benchmark.utils.constants import DEVICE, ROOT_DIR
from hdbo_benchmark.utils.experiments.load_solvers import (SOLVER_NAMES,
                                                           load_solver)
from hdbo_benchmark.utils.logging.idempotence_of_experiments import \
    experiment_has_already_run
from hdbo_benchmark.utils.logging.uncommited_changes import \
    has_uncommitted_changes
from hdbo_benchmark.utils.logging.wandb_observer import initialize_observer

torch.set_default_dtype(torch.float32)


def from_unit_cube_to_range(z: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return z * (bounds[1] - bounds[0]) + bounds[0]


def from_range_to_unit_cube(z: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return (z - bounds[0]) / (bounds[1] - bounds[0])


def in_latent_space(
    f: AbstractBlackBox, ae, latent_space_bounds: tuple[float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    def _latent_f(z: np.ndarray) -> np.ndarray:
        # We assume that z is in [0, 1]
        z = from_unit_cube_to_range(z, latent_space_bounds)
        protein_strings_w_special_tokens = ae.decode_to_string_array(z)
        protein_strings = [
            ["".join([c for c in p if c not in ["<pad>", "<cls>", "<eos>"]])]
            for p in protein_strings_w_special_tokens
        ]
        results = []
        valid_lengths = set([len("".join(x_i)) for x_i in f.x0])
        for p in protein_strings:
            if len(p[0]) not in valid_lengths:
                results.append(np.array([[-100.0]]))
            else:
                results.append(f(np.array([p])))
        val = np.array(results).reshape(z.shape[0], 1)
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
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--solve-in-discrete-space/--no-solve-in-discrete-space", default=False)
@click.option("--tag", type=str, default="default")
def main(
    solver_name: str,
    latent_dim: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    solve_in_discrete_space: bool,
    tag: str,
):
    print(f"Device: {DEVICE}")
    experiment_name = "benchmark_on_gfp"
    n_initial_points = 3

    for module in [hdbo_benchmark, poli, poli_baselines]:
        if has_uncommitted_changes(module) and strict_on_hash:
            raise Exception(
                f"There are uncommitted changes in the repositories in {module.__name__}"
            )

    # Checking if this experimenr has already been run
    if (
        experiment_has_already_run(
            experiment_name=experiment_name,
            solver_name=solver_name,
            function_name="gfp",
            n_dimensions=latent_dim,
            seed=seed,
        )
        and not force_run
    ):
        print(
            f"The experiment for solver {solver_name} with function "
            f"rasp and n_dimensions {latent_dim} "
            f" and seed {seed} has already been run."
        )
        return

    DATA_DIR = ROOT_DIR / "data" / experiment_name / f"{solver_name}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = f"{uuid4()}"[:8]

    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_numpy(seed)
    seed_python(seed)
    torch.manual_seed(seed)

    # 2. Load a black box and solver
    problem = objective_factory.create(
        wildtype_pdb_path="gfp_cbas",
        seed=seed,
        problem_type="gp",
        n_starting_points=n_initial_points,
        evaluation_budget=max_iter + n_initial_points,
    )
    problem_vae = objective_factory.create(
        wildtype_pdb_path="gfp_cbas",
        seed=seed,
        problem_type="vae",
        n_starting_points=n_initial_points,
        evaluation_budget=max_iter + n_initial_points,
    )
    f = problem.black_box
    vae = problem_vae.black_box

    obs = initialize_observer(
        experiment_name=experiment_name,
        f=f,
        function_name="gfp",
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        experiment_id=experiment_id,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        tag=tag,
    )
    f.set_observer(obs)

    latent_space_bounds = (-3, 3)  # By inspecting z0.
    solvers_that_dont_allow_eager_evaluation = ["bounce", "baxus"]
    if solver_name in solvers_that_dont_allow_eager_evaluation:
        z0 = None
        x0 = None
        y0 = None
    else:
        z0 = vae.encode(x0.to(DEVICE)).numpy(
            force=True
        )
        z0 = from_range_to_unit_cube(z0, latent_space_bounds)
        x0 = problem.x0
        y0 = f(x0)

    if solve_in_discrete_space:
        f_ = f
        alphabet = f.info.alphabet + [""]
        sequence_length = f.info.max_sequence_length
        x0_for_solver = x0
        kwargs_ = {
            "alphabet": alphabet,
            "sequence_length": sequence_length,
        }
    else:
        f_ = in_latent_space(f, vae, latent_space_bounds)
        x0_for_solver = z0
        kwargs_ = {
            "bounds": f_input_bounds,
        }

    solver_, kwargs = load_solver(
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        n_initial_points=6,
        bounds=f_input_bounds,
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

    obs.finish()


if __name__ == "__main__":
    main()
