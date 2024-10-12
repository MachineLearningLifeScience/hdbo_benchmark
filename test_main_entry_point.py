import torch

import pytest

from run import _main

from poli.benchmarks import PMOBenchmark

from hdbo_benchmark.utils.experiments.load_solvers import SOLVER_NAMES

SOLVERS_THAT_RUN_IN_BASE_ENV = SOLVER_NAMES

DEVICE = torch.device("cpu")

PMO_BENCHMARK_SETUPS = [
    (function_name, solver_name, 128)
    for solver_name in SOLVER_NAMES
    for function_name in PMOBenchmark(string_representation="SELFIES").problem_names[:1]
]

RASP_BENCHMARK_SETUPS = []
for latent_dim in [128]:
    for solver_name in SOLVER_NAMES:
        RASP_BENCHMARK_SETUPS.append(("rasp", solver_name, latent_dim))

TEST_SETUPS = PMO_BENCHMARK_SETUPS + RASP_BENCHMARK_SETUPS


@pytest.mark.parametrize(
    "function_name, solver_name, latent_dim",
    TEST_SETUPS,
)
def test_main_run(function_name, solver_name, latent_dim):
    _main(
        solver_name=solver_name,
        function_name=function_name,
        n_dimensions=latent_dim,
        seed=None,
        max_iter=1,
        strict_on_hash=False,
        force_run=True,
        tag="test",
        wandb_mode="disabled",
    )


if __name__ == "__main__":
    print(len(TEST_SETUPS))
    for test_setup in TEST_SETUPS:
        print(test_setup)
        test_main_run(*test_setup)
