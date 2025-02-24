import json
from datetime import datetime
import pandas as pd
from typing import Literal
from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.utils.slurm.create_batch_file import (
    write_batch_script_for_commands,
)

SOLVER_TO_CONDA_ENV_NAME = {
    "pr": "hdbo_pr",
    "vanilla_bo_hvarfner": "hdbo_ax",
    "saas_bo": "hdbo_ax",
    "turbo": "hdbo_benchmark",
}


def load_run_lengths(n_dimensions: Literal[128, 2]) -> pd.DataFrame:
    return pd.read_csv(
        ROOT_DIR / "data" / "run_lengths" / f"run_lengths_{n_dimensions}.csv"
    )


def compute_experiments_to_run(
    n_dimensions: Literal[128, 2],
) -> list[dict[str, list[int]]]:
    ITERATIONS_THRESHOLD = 20

    run_lengths = load_run_lengths(n_dimensions)

    experiments_to_run = []
    for experiment in run_lengths.itertuples():
        if experiment.max_steps <= ITERATIONS_THRESHOLD:
            experiments_to_run.append(
                {
                    "function_name": experiment.function_name,
                    "solver_name": experiment.solver_name,
                    "max_steps": experiment.max_steps,
                    "seed": experiment.seed,
                    "n_dimensions": n_dimensions,
                }
            )

    return experiments_to_run


def save_experiments_to_run(experiments_to_run: list[dict[str, list[int]]]):
    with open(
        ROOT_DIR / "data" / "results_cache" / "incomplete_experiments.json", "w"
    ) as f:
        json.dump(experiments_to_run, f)


def from_incomplete_experiments_to_slurm_script(
    incomplete_experiments: list[dict[str, list[int]]],
    n_dimensions: Literal[128, 2],
    *,
    wandb_mode: str = "online",
    tag: str = f"incomplete-experiments-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
):
    commands = []
    for experiment in incomplete_experiments:
        max_iter = 300
        commands.append(
            f"conda run -n {SOLVER_TO_CONDA_ENV_NAME[experiment['solver_name']]} python run.py --function-name {experiment['function_name']} --solver-name {experiment['solver_name']} --n-dimensions {experiment['n_dimensions']} --seed {experiment['seed']} --max-iter {max_iter} --strict-on-hash --force-run --wandb-mode {wandb_mode} --tag {tag}"
        )

    write_batch_script_for_commands(
        commands,
        job_name=f"incomplete-experiments-{n_dimensions}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        gpu_resources="",
        slurm_script_output_path=ROOT_DIR
        / f"incomplete-experiments-{n_dimensions}.local.sh",
        instruction_file_output_path=ROOT_DIR
        / f"incomplete-experiments-instructions-{n_dimensions}.local.sh",
    )


if __name__ == "__main__":
    experiments_to_run = compute_experiments_to_run(n_dimensions=128)
    save_experiments_to_run(experiments_to_run)
    print(len(experiments_to_run))
    from_incomplete_experiments_to_slurm_script(experiments_to_run, n_dimensions=128)

    experiments_to_run = compute_experiments_to_run(n_dimensions=2)
    save_experiments_to_run(experiments_to_run)
    print(len(experiments_to_run))
    from_incomplete_experiments_to_slurm_script(experiments_to_run, n_dimensions=2)
