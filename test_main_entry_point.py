import os

import pytest

from click.testing import CliRunner
from run import main

from poli.benchmarks import PMOBenchmark

from hdbo_benchmark.utils.experiments.load_solvers import SOLVER_NAMES

SOLVERS_THAT_RUN_IN_BASE_ENV = SOLVER_NAMES

PMO_BENCHMARK_SETUPS = [
    (function_name, solver_name, 128)
    for solver_name in SOLVER_NAMES
    for function_name in PMOBenchmark(string_representation="SELFIES").problem_names
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
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--solver-name",
            solver_name,
            "--function-name",
            function_name,
            "--n-dimensions",
            str(latent_dim),
            "--max-iter",
            "1",
            "--no-strict-on-hash",
            "--force-run",
            "--tag",
            "test",
            "--wandb-mode",
            "disabled",
        ],
    )

    print(result.output)
    assert result.exit_code == 0


if __name__ == "__main__":
    for test_setup in TEST_SETUPS[:5]:
        print(test_setup)
        test_main_run(*test_setup)
