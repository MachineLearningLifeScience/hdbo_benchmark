import pytest

from run import _main

from poli.benchmarks import PMOBenchmark

from hdbo_benchmark.utils.experiments.load_solvers import (
    SOLVER_NAMES,
    SOLVER_NAME_TO_ENV,
)

SOLVERS_THAT_RUN_IN_BASE_ENV = [
    solver_name
    for solver_name in SOLVER_NAMES
    if SOLVER_NAME_TO_ENV[solver_name] == "hdbo_benchmark"
]

MAX_ITER = 1
SEED = 1


def construct_setups(solvers: list[str]):
    pmo_setups = [
        (function_name, solver_name, 128)
        for solver_name in solvers
        for function_name in PMOBenchmark(
            string_representation="SELFIES"
        ).problem_names[:1]
    ]

    ehrlich_like_setups = [
        (function_name, solver_name, 128)
        for function_name in ["ehrlich_holo_tiny", "pest_control_equivalent"]
        for solver_name in solvers
    ]

    return pmo_setups + ehrlich_like_setups


@pytest.mark.hdbo_base
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(SOLVERS_THAT_RUN_IN_BASE_ENV),
)
def test_main_run(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=SEED,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_ax
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(["vanilla_bo_hvarfner", "saas_bo"]),
)
def test_main_run_ax(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_baxus
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(["baxus"]),
)
def test_main_run_baxus(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_alebo
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(["alebo"]),
)
def test_main_run_alebo(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_bounce
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(["bounce"]),
)
def test_main_run_bounce(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_pr
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    construct_setups(["pr"]),
)
def test_main_run_pr(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


@pytest.mark.hdbo_lambo2
@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    filter(
        lambda tuple_: tuple_[0] == "foldx_stability",
        construct_setups(["lambo2"]),
    ),
)
def test_main_run_lambo2(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=MAX_ITER,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


if __name__ == "__main__":
    test_setups_in_main = construct_setups(SOLVERS_THAT_RUN_IN_BASE_ENV)
    print(len(test_setups_in_main))
    for test_setup in test_setups_in_main:
        print(test_setup)
        try:
            test_main_run(*test_setup)
        except Exception as e:
            print(f"could not run for {test_setup}")
            print(e)
        print("-" * 80)
