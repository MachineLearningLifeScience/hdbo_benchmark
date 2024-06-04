import wandb

from hdbo_benchmark.utils.constants import WANDB_ENTITY


def get_all_runs_for_experiment(
    experiment_name: str,
    solver_name: str | None = None,
    function_name: str | None = None,
    n_dimensions: int | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
) -> list[wandb.apis.public.Run]:
    api = wandb.Api()

    filter_: dict[str, str | int | dict] = {
        "state": {"$in": ["finished", "running", "failed", "crashed"]},
    }
    if solver_name is not None:
        filter_["config.solver_name"] = solver_name

    if function_name is not None:
        filter_["config.function_name"] = function_name

    if n_dimensions is not None:
        # If it's a discrete solver, we don't need to filter by n_dimensions
        if solver_name in ["bounce", "pr"]:
            pass
        else:
            filter_["config.n_dimensions"] = n_dimensions

    if seed is not None:
        filter_["config.seed"] = seed

    if tags is not None:
        filter_["tags"] = {"$in": tags}

    runs = api.runs(
        f"{WANDB_ENTITY}/{experiment_name}",
        filter_ if filter_ else None,
    )
    return list(runs)
