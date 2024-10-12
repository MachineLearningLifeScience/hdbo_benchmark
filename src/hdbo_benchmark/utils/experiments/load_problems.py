import json

import poli
from poli.benchmarks import PMOBenchmark
from poli.core.problem import Problem
from poli.repository import RaspProblemFactory, FoldXStabilityProblemFactory

from hdbo_benchmark.utils.constants import ROOT_DIR, PENALIZE_UNFEASIBLE_WITH
from hdbo_benchmark.utils.logging.wandb_observer import (
    initialize_observer,
    ObserverConfig,
)
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import (
    load_alphabet_for_pmo,
    load_sequence_length_for_pmo,
)


def _load_pmo_problem(function_name: str) -> Problem:
    problem = poli.create(
        name=function_name,
        string_representation="SELFIES",
        alphabet=load_alphabet_for_pmo(),
        max_sequence_length=load_sequence_length_for_pmo(),
    )

    return problem


def _load_rasp():
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"

    ESM_EMBEDDINGS_DIR = ROOT_DIR / "data" / "esm_embeddings"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*.pdb"))

    with open(ESM_EMBEDDINGS_DIR / "closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)

    wildtype_pdbs = [p["closest_pdb"] for p in closest_pdbs]
    ALL_PDBS = list(filter(lambda x: x.parent.name in wildtype_pdbs, ALL_PDBS))
    chains_to_keep = [p.stem.split("_")[1] for p in ALL_PDBS]

    return RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=chains_to_keep,
        penalize_unfeasible_with=PENALIZE_UNFEASIBLE_WITH,
    )


def _load_foldx_stability():
    PDBS_DIR = ROOT_DIR / "data" / "rfp_pdbs"
    ALL_PDBS = list(PDBS_DIR.rglob("**/*_Repair.pdb"))

    return FoldXStabilityProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        verbose=True,
    )


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

    n_initial_points = problem.x0.shape[0]
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
