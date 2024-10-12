import os

import pytest

from click.testing import CliRunner
from hdbo_benchmark.experiments.benchmark_on_pmo.run import main


@pytest.mark.parametrize("function_name", ["albuterol_similarity", "valsartan_smarts"])
@pytest.mark.parametrize("solver_name", ["random_mutation", "line_bo", "turbo"])
@pytest.mark.parametrize("latent_dim", [2, 128])
def test_main_run(function_name, solver_name, latent_dim):
    os.environ["WANDB_MODE"] = "disabled"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--solver-name",
            solver_name,
            "--function-name",
            function_name,
            "--latent-dim",
            str(latent_dim),
            "--max-iter",
            "3",
            "--n-initial-points",
            "2",
            "--no-strict-on-hash",
            "--force-run",
            "--solve-in-discrete-space",
            "--tag",
            "test",
        ],
    )
