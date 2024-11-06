import wandb
from hdbo_benchmark.utils.constants import WANDB_ENTITY


def experiment_has_already_run(
    experiment_name: str,
    solver_name: str,
    function_name: str,
    n_dimensions: int,
    seed: int,
) -> bool:
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{experiment_name}",
        {
            "config.solver_name": solver_name,
            "config.function_name": function_name,
            "config.seed": seed,
            "config.n_dimensions": n_dimensions,
            "display_name": {"$regex": f".*n_dimensions-{n_dimensions}-.*"},
            "state": "finished",
        },
    )
    try:
        return len(list(runs)) > 0
    except ValueError:
        # wandb throws an error if the project does not exist.
        return False
