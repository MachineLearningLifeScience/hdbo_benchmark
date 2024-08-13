from typing import Callable
from uuid import uuid4

import click

import torch
import numpy as np
from selfies import split_selfies  # type: ignore[import]

import poli  # type: ignore[import]
from poli.core.util.seeding import seed_numpy, seed_python  # type: ignore[import]
from poli.objective_repository import EhrlichBlackBox  # type: ignore[import]
from poli.core.exceptions import BudgetExhaustedException  # type: ignore[import]

import poli_baselines  # type: ignore[import]

import hdbo_benchmark
from hdbo_benchmark.utils.experiments.load_solvers import load_solver, SOLVER_NAMES
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE
from hdbo_benchmark.utils.logging.uncommited_changes import has_uncommitted_changes

from hdbo_benchmark.utils.logging.wandb_observer import initialize_observer
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)

torch.set_default_dtype(torch.float32)


def _from_sequence_to_one_hot(
    x: np.ndarray, alphabet: list[str], sequence_length: int
) -> np.ndarray:
    one_hot_x = np.zeros((len(x), len(alphabet), sequence_length))
    for i, x_i in enumerate(x):
        for j, x_ij in enumerate(x_i):
            one_hot_x[i, alphabet.index(x_ij), j] = 1
    return one_hot_x


def _from_one_hot_to_sequence(
    x: np.ndarray, alphabet: list[str], sequence_length: int
) -> np.ndarray:
    x = x.reshape(x.shape[0], len(alphabet), sequence_length)
    x = np.argmax(x, axis=1)
    x = np.array([[alphabet[int_] for int_ in x_i] for x_i in x])
    return x


def _in_onehot_space(
    f: Callable[[np.ndarray], np.ndarray], sequence_length: int, alphabet: list[str]
) -> Callable[[np.ndarray], np.ndarray]:
    def _f(onehot_x: np.ndarray) -> np.ndarray:
        assert onehot_x.ndim > 1

        sequence_x = _from_one_hot_to_sequence(
            onehot_x, alphabet=alphabet, sequence_length=sequence_length
        )
        assert sequence_x.shape == (onehot_x.shape[0], sequence_length)

        return f(sequence_x)

    _f.info = f.info  # type: ignore[attr-defined]
    _f.num_workers = f.num_workers  # type: ignore[attr-defined]

    return _f


@click.command()
@click.option(
    "--solver-name",
    type=str,
    default="random_mutation",
    help=f"The name of the solver to run. All solvers available are: {SOLVER_NAMES}",
)
@click.option("--sequence-length", type=int, default=128)
@click.option("--n-motifs", type=int, default=128)
@click.option("--motif-length", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--n-initial-points", type=int, default=10)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--solve-in-discrete-space/--no-solve-in-discrete-space", default=False)
@click.option("--tag", type=str, default="default")
def main(
    solver_name: str,
    sequence_length: int,
    n_motifs: int,
    motif_length: int,
    seed: int,
    max_iter: int,
    n_initial_points: int,
    strict_on_hash: bool,
    force_run: bool,
    solve_in_discrete_space: bool,
    tag: str,
):
    print(f"Device: {DEVICE}")
    experiment_name = "benchmark_on_ehrlich"
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
            function_name="ehrlich",
            n_dimensions=sequence_length,
            seed=seed,
        )
        and not force_run
    ):
        print(
            f"The experiment for solver {solver_name} with function "
            f"ehrlich and sequence length {sequence_length} "
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

    # 2. Load the black box and solver
    f = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=seed,
        evaluation_budget=max_iter + n_initial_points,
    )

    obs = initialize_observer(
        experiment_name=experiment_name,
        f=f,
        function_name="ehrlich",
        solver_name=solver_name,
        n_dimensions=sequence_length,
        seed=seed,
        experiment_id=experiment_id,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        tag=tag,
    )
    f.set_observer(obs)

    bounds = (0.0, 1.0)
    solvers_that_dont_allow_eager_evaluation = ["bounce"]
    if solver_name in solvers_that_dont_allow_eager_evaluation:
        x0 = None
        onehot_x0 = None
        y0 = None
    else:
        x0 = np.array(
            [list(f._sample_random_sequence()) for _ in range(n_initial_points)]
        )
        onehot_x0 = _from_sequence_to_one_hot(
            x0, alphabet=f.info.alphabet, sequence_length=f.sequence_length
        )
        y0 = f(x0)

    if solve_in_discrete_space:
        f_ = f
        alphabet = f.info.alphabet
        # We need to split x0 into [b, L] tokens.
        if x0 is not None:
            # split_x0 = [list(split_selfies(x_i)) for x_i in x0]
            # split_x0 = [
            #     x_i + ["[nop]"] * (sequence_length - len(x_i)) for x_i in split_x0
            # ]
            # x0_for_solver = np.array(split_x0)
            x0_for_solver = x0
        if x0 is None:
            x0_for_solver = None
        kwargs_ = {
            "alphabet": alphabet,
            "sequence_length": sequence_length,
        }
    else:
        f_ = _in_onehot_space(
            f, sequence_length=sequence_length, alphabet=f.info.alphabet
        )
        x0_for_solver = onehot_x0
        kwargs_ = {
            "bounds": bounds,
        }

    solver_, kwargs = load_solver(
        solver_name=solver_name,
        n_dimensions=sequence_length * len(f.info.alphabet),
        seed=seed,
        n_initial_points=n_initial_points,
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
