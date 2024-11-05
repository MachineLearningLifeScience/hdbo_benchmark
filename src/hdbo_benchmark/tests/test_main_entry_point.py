import os
import sys

import pytest

from hdbo_benchmark.utils.constants import ROOT_DIR

sys.path.append(str(ROOT_DIR))

from run import _main


@pytest.mark.hdbo_base
@pytest.mark.parametrize("function_name", ["albuterol_similarity", "valsartan_smarts"])
@pytest.mark.parametrize("solver_name", ["random_mutation", "line_bo", "turbo"])
@pytest.mark.parametrize("latent_dim", [2, 128])
def test_main_run(function_name, solver_name, latent_dim):
    os.environ["WANDB_MODE"] = "disabled"
    _main(
        function_name=function_name,
        solver_name=solver_name,
        n_dimensions=latent_dim,
        seed=42,
        max_iter=1,
        strict_on_hash=False,
        force_run=True,
        wandb_mode="disabled",
        tag=None,
    )
