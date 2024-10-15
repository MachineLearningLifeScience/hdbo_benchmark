import json

import poli
from poli.benchmarks import PMOBenchmark
from poli.core.problem import Problem
from poli.repository import RaspProblemFactory, FoldXStabilityProblemFactory
from poli.core.data_package import DataPackage
from poli.core.chemistry.data_packages import RandomMoleculesDataPackage
from poli.core.proteins.data_packages import RFPRaspSupervisedDataPackage

from hdbo_benchmark.utils.constants import ROOT_DIR, PENALIZE_UNFEASIBLE_WITH
from hdbo_benchmark.utils.logging.wandb_observer import (
    initialize_observer,
    ObserverConfig,
)
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import (
    load_alphabet_for_pmo,
    load_sequence_length_for_pmo,
)

def turn_into_supervised_problem(problem: Problem) -> Problem:
    data_package = problem.data_package
    x0 = data_package.unsupervised_data
    y0 = problem.black_box(problem.data_package.unsupervised_data)

    return Problem(
        black_box=problem.black_box,
        x0=problem.x0,
        data_package=DataPackage(
            unsupervised_data=x0,
            supervised_data=(x0, y0)
        )
    )


def _load_pmo_problem(function_name: str) -> Problem:
    problem = poli.create(
        name=function_name,
        string_representation="SELFIES",
        alphabet=load_alphabet_for_pmo(),
        max_sequence_length=load_sequence_length_for_pmo(),
    )
    problem.data_package = RandomMoleculesDataPackage(string_representation="SELFIES", n_molecules=10)
    return turn_into_supervised_problem(problem)



def _load_rasp():
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"

    ESM_EMBEDDINGS_DIR = ROOT_DIR / "data" / "esm_embeddings"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*.pdb"))

    with open(ESM_EMBEDDINGS_DIR / "closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)

    wildtype_pdbs = [p["closest_pdb"] for p in closest_pdbs]
    ALL_PDBS = list(filter(lambda x: x.parent.name in wildtype_pdbs, ALL_PDBS))
    chains_to_keep = [p.stem.split("_")[1] for p in ALL_PDBS]

    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=chains_to_keep,
        penalize_unfeasible_with=PENALIZE_UNFEASIBLE_WITH,
    )

    problem.data_package = RFPRaspSupervisedDataPackage()

    return problem


def _load_foldx_stability():
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*_Repair.pdb"))

    problem = FoldXStabilityProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        verbose=True,
    )

    return problem


def _load_problem(function_name: str) -> Problem:
    match function_name:
        case function_name if function_name in PMOBenchmark(
            string_representation="SELFIES"
        ).problem_names:
            return _load_pmo_problem(function_name)
        case "rasp":
            return _load_rasp()
        case "foldx_stability":
            return _load_foldx_stability()
        case _:
            raise ValueError(f"Unknown function name: {function_name}")


def load_problem(
    function_name: str,
    max_iter: int,
    set_observer: bool = True,
    observer_config: ObserverConfig = None,
):
    problem = _load_problem(function_name)
    
    if problem.data_package.supervised_data is None:
        raise ValueError("Expected the problem to have a supervised dataset attached.")

    initial_x, _ = problem.data_package.supervised_data
    n_initial_points = initial_x.shape[0]
    problem.black_box.set_evaluation_budget(max_iter + n_initial_points)

    if set_observer:
        assert (
            observer_config is not None
        ), "Observer config must be provided if set_observer is True"

        obs = initialize_observer(
            problem=problem,
            observer_config=observer_config,
        )
        problem.black_box.set_observer(obs)

    return problem
