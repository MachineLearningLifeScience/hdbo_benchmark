from typing import Literal

import selfies as sf
import numpy as np

import poli
from poli.benchmarks import PMOBenchmark
from poli.core.problem import Problem
from poli.repository import RFPRaspProblemFactory, RFPFoldXStabilityProblemFactory
from poli.repository import EhrlichHoloProblemFactory, EhrlichHoloBlackBox
from poli.core.data_package import DataPackage
from poli.core.chemistry.data_packages import RandomMoleculesDataPackage

from hdbo_benchmark.utils.constants import PENALIZE_UNFEASIBLE_WITH
from hdbo_benchmark.utils.logging.wandb_observer import (
    initialize_observer,
    ObserverConfig,
)
from hdbo_benchmark.utils.experiments.load_metadata_for_vaes import (
    load_alphabet_for_pmo,
    load_sequence_length_for_pmo,
)


def turn_into_supervised_problem(problem: Problem, strict: bool = True) -> Problem:
    data_package = problem.data_package
    if strict:
        assert (
            problem.data_package.supervised_data is None
        ), f"The problem {problem} is already supervised."
    x0 = data_package.unsupervised_data
    y0 = problem.black_box(problem.data_package.unsupervised_data)

    return Problem(
        black_box=problem.black_box,
        x0=problem.x0,
        data_package=DataPackage(unsupervised_data=x0, supervised_data=(x0, y0)),
    )


def tokenize_selfies(x: str, max_sequence_length: int) -> list[str]:
    selfies_tokens = list(sf.split_selfies(x))
    return selfies_tokens + ["[nop]"] * (max_sequence_length - len(selfies_tokens))


def _load_pmo_problem(function_name: str) -> Problem:
    problem = poli.create(
        name=function_name,
        string_representation="SELFIES",
        alphabet=load_alphabet_for_pmo(),
        max_sequence_length=load_sequence_length_for_pmo(),
    )
    problem.data_package = RandomMoleculesDataPackage(
        string_representation="SELFIES",
        n_molecules=10,
        tokenize_with=lambda x: tokenize_selfies(x, problem.info.max_sequence_length),
    )
    return turn_into_supervised_problem(problem)


def _load_rasp() -> Problem:
    problem = RFPRaspProblemFactory().create(
        additive=True,
        penalize_unfeasible_with=PENALIZE_UNFEASIBLE_WITH,
    )

    return problem


def _load_foldx_stability() -> Problem:
    problem = RFPFoldXStabilityProblemFactory().create(
        verbose=True,
    )

    return problem

def _load_ehrlich_holo(size: Literal["small", "large"]) -> Problem:

    # TODO: decide on these according to the original paper.
    if size == "small":
        sequence_length = 5
        n_motifs = 1
        motif_length = 4
        n_supervised_points = 10
    elif size == "large":
        sequence_length = 64
        n_motifs = 4
        motif_length = 10
        n_supervised_points = 1000
    else:
        raise ValueError()
    
    problem = EhrlichHoloProblemFactory().create(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        return_value_on_unfeasible=-1.0
    )
    f: EhrlichHoloBlackBox = problem.black_box
    unsupervised_data = np.array([list(x_i) for x_i in f.initial_solution(n_samples=n_supervised_points)])

    problem.data_package = DataPackage(
        unsupervised_data=unsupervised_data,
        supervised_data=(
            unsupervised_data,
            f(unsupervised_data),
        )
    )

    return problem



def _load_problem(function_name: str) -> Problem:
    match function_name:
        case function_name if function_name in PMOBenchmark(
            string_representation="SELFIES"
        ).problem_names:
            return _load_pmo_problem(function_name)
        case "rfp_rasp":
            return _load_rasp()
        case "rfp_foldx_stability":
            return _load_foldx_stability()
        case "ehrlich_holo_small":
            return _load_ehrlich_holo(size="small")
        case "ehrlich_holo_large":
            return _load_ehrlich_holo(size="large")
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
