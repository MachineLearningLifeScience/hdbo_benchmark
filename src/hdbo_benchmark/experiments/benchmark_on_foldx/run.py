"""
Running the benchmark on FoldX stability.
"""

# mypy: disable-error-code="import-untyped"
from typing import Callable
from uuid import uuid4
import json

import click

import pandas as pd
import torch
import numpy as np

import poli
from poli.repository import FoldXStabilityProblemFactory
from poli.core.util.seeding import seed_numpy, seed_python
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.exceptions import BudgetExhaustedException
from poli.core.problem import Problem

import poli_baselines

import hdbo_benchmark
from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.experiments.load_solvers import (
    load_solver,
    SOLVER_NAMES,
    DISCRETE_SPACE_SOLVERS,
    SOLVERS_THAT_DONT_ALLOW_CUSTOM_INPUTS,
)
from hdbo_benchmark.utils.experiments.load_generative_models import (
    load_generative_model_and_bounds,
)
from hdbo_benchmark.utils.experiments.normalization import (
    from_unit_cube_to_range,
    from_range_to_unit_cube,
)
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE
from hdbo_benchmark.utils.logging.uncommited_changes import has_uncommitted_changes

from hdbo_benchmark.utils.logging.wandb_observer import initialize_observer
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)

torch.set_default_dtype(torch.float32)


def in_latent_space(
    f: AbstractBlackBox,
    ae: LitAutoEncoder,
    latent_space_bounds: tuple[float, float],
    x0: np.ndarray,
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

    return _latent_f


def define_initial_values(
    solver_name: str,
    problem: Problem,
    ae: LitAutoEncoder,
    latent_space_bounds: tuple[float, float] = (-15.0, 15.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*_Repair.pdb"))

    ESM_EMBEDDINGS_DIR = ROOT_DIR / "data" / "esm_embeddings"

    with open(ESM_EMBEDDINGS_DIR / "closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)

    wildtype_pdbs = [p["closest_pdb"] for p in closest_pdbs]
    ALL_PDBS = list(filter(lambda x: x.parent.name in wildtype_pdbs, ALL_PDBS))
    ALL_PDB_IDS = [p.parent.name for p in ALL_PDBS]

    with open(ESM_EMBEDDINGS_DIR / "esm_embeddings.json") as f:
        embeddings_and_sequences = json.load(f)
        df = pd.DataFrame(embeddings_and_sequences)
        df.set_index("label", inplace=True)

    x0_esm = np.vstack(df.loc[ALL_PDB_IDS, "embedding"].values)

    if solver_name in SOLVERS_THAT_DONT_ALLOW_CUSTOM_INPUTS:
        z0 = None
        x0 = None
        y0 = None
    else:
        z0 = ae.encode(torch.from_numpy(x0_esm).to(torch.float32).to(DEVICE)).numpy(
            force=True
        )
        z0 = from_range_to_unit_cube(z0, latent_space_bounds)
        x0 = problem.x0
        y0 = problem.black_box(x0)

    return z0, x0, y0


@click.command()
@click.option(
    "--solver-name",
    type=str,
    default="directed_evolution",
    help=f"The name of the solver to run. All solvers available are: {SOLVER_NAMES}",
)
@click.option("--latent-dim", type=int, default=128)
@click.option("--seed", type=int, default=None)
@click.option("--max-iter", type=int, default=100)
@click.option("--strict-on-hash/--no-strict-on-hash", type=bool, default=True)
@click.option("--force-run/--no-force-run", default=True)
@click.option("--use-starting-pool/--no-use-starting-pool", default=False)
@click.option("--wandb-mode", type=str, default="online")
@click.option("--tag", type=str, default="default")
def main(
    solver_name: str,
    latent_dim: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    use_starting_pool: bool,
    wandb_mode: str,
    tag: str,
):
    print(f"Device: {DEVICE}")
    experiment_name = "benchmark_on_foldx"

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
            function_name="rasp",
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
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*_Repair.pdb"))

    experiment_id = f"{uuid4()}"[:8]

    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_numpy(seed)
    seed_python(seed)
    torch.manual_seed(seed)

    # 2. Load a black box and solver
    n_initial_points = 6  # TODO: incorporate pool, perhaps.
    problem = FoldXStabilityProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        evaluation_budget=max_iter + n_initial_points,
        verbose=True,
    )
    f = problem.black_box

    obs = initialize_observer(
        experiment_name=experiment_name,
        f=f,
        function_name="foldx_stability",
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        experiment_id=experiment_id,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        mode=wandb_mode,
        tag=tag,
    )
    f.set_observer(obs)

    ae, latent_space_bounds = load_generative_model_and_bounds(
        experiment_name=experiment_name,
        latent_dim=latent_dim,
    )

    z0, x0, y0 = define_initial_values(
        solver_name=solver_name,
        problem=problem,
        ae=ae,
        latent_space_bounds=latent_space_bounds,
    )

    f_input_bounds = (0.0, 1.0)
    solve_in_discrete_space = solver_name in DISCRETE_SPACE_SOLVERS
    if solve_in_discrete_space:
        f_ = f
        alphabet = f.info.alphabet + [""]
        sequence_length = f.info.max_sequence_length
        x0_for_solver = x0
    else:
        f_ = in_latent_space(
            f,
            ae,
            latent_space_bounds,
            problem.x0,
        )
        x0_for_solver = z0

    solver_, kwargs = load_solver(
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        n_initial_points=n_initial_points,
        bounds=f_input_bounds,
        alphabet=alphabet,
        sequence_length=sequence_length,
    )
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
