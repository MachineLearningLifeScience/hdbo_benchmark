from typing import Callable
from uuid import uuid4
import json

import click

import pandas as pd
import torch
import numpy as np

import esm

import poli  # type: ignore[import]
from poli.repository import RaspProblemFactory  # type: ignore[import]
from poli.core.util.seeding import seed_numpy, seed_python  # type: ignore[import]
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]
from poli.core.exceptions import BudgetExhaustedException  # type: ignore[import]

import poli_baselines  # type: ignore[import]

import hdbo_benchmark
from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.experiments.load_solvers import load_solver, SOLVER_NAMES
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE
from hdbo_benchmark.utils.logging.uncommited_changes import has_uncommitted_changes

from hdbo_benchmark.utils.logging.wandb_observer import initialize_observer
from hdbo_benchmark.utils.logging.idempotence_of_experiments import (
    experiment_has_already_run,
)

torch.set_default_dtype(torch.float32)


def in_latent_space(
    f: AbstractBlackBox, ae: LitAutoEncoder
) -> Callable[[np.ndarray], np.ndarray]:
    def _latent_f(z: np.ndarray) -> np.ndarray:
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
    latent_dim: int,
    seed: int,
    max_iter: int,
    strict_on_hash: bool,
    force_run: bool,
    solve_in_discrete_space: bool,
    tag: str,
):
    print(f"Device: {DEVICE}")
    experiment_name = "benchmark_on_rasp"
    n_initial_points = 6

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
    MODELS_DIR = (
        ROOT_DIR / "data" / "trained_models" / "ae_for_esm" / f"latent_dim_{latent_dim}"
    )

    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"

    ESM_EMBEDDINGS_DIR = ROOT_DIR / "data" / "esm_embeddings"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*.pdb"))

    with open(ESM_EMBEDDINGS_DIR / "closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)

    wildtype_pdbs = [p["closest_pdb"] for p in closest_pdbs]
    ALL_PDBS = list(filter(lambda x: x.parent.name in wildtype_pdbs, ALL_PDBS))
    ALL_PDB_IDS = [p.parent.name for p in ALL_PDBS]
    chains_to_keep = [p.stem.split("_")[1] for p in ALL_PDBS]

    with open(ESM_EMBEDDINGS_DIR / "esm_embeddings.json") as f:
        embeddings_and_sequences = json.load(f)
        df = pd.DataFrame(embeddings_and_sequences)
        df.set_index("label", inplace=True)

    x0_esm = np.vstack(df.loc[ALL_PDB_IDS, "embedding"].values)

    experiment_id = f"{uuid4()}"[:8]

    if seed is None:
        seed = np.random.randint(0, 10_000)

    seed_numpy(seed)
    seed_python(seed)
    torch.manual_seed(seed)

    # 2. Load a black box and solver
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=chains_to_keep,
        evaluation_budget=max_iter + n_initial_points,
    )
    f = problem.black_box

    obs = initialize_observer(
        experiment_name=experiment_name,
        f=f,
        function_name="rasp",
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        experiment_id=experiment_id,
        max_iter=max_iter,
        strict_on_hash=strict_on_hash,
        tag=tag,
    )
    f.set_observer(obs)

    bounds = (-5.0, 5.0)  # TODO: ?
    _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    ae = LitAutoEncoder.load_from_checkpoint(
        MODELS_DIR / "version_23" / "checkpoints" / "epoch=99-step=700.ckpt",
        alphabet=alphabet,
        latent_dim=latent_dim,
    )
    solvers_that_dont_allow_eager_evaluation = ["bounce"]
    if solver_name in solvers_that_dont_allow_eager_evaluation:
        z0 = None
        x0 = None
        y0 = None
    else:
        z0 = ae.encode(torch.from_numpy(x0_esm).to(torch.float32).to(DEVICE)).numpy(
            force=True
        )
        x0 = ae.decode_to_string_array(z0)
        y0 = f(x0)

    if solve_in_discrete_space:
        f_ = f
        alphabet = f.info.alphabet
        sequence_length = ae.sequence_length
        if x0 is None:
            x0_for_solver = None
        kwargs_ = {
            "alphabet": alphabet,
            "sequence_length": sequence_length,
        }
    else:
        f_ = in_latent_space(f, ae)
        x0_for_solver = z0
        kwargs_ = {
            "bounds": bounds,
        }

    solver_, kwargs = load_solver(
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=seed,
        n_initial_points=6,
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