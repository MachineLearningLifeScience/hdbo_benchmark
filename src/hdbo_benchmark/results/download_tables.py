"""
This script creates a table in pandas, downloading the data
from wandb.

The model for each row of the base table goes like this:
{
    "model": "model_name",
    "function": "function_name",
    "step": "step_value",
    "x": "x_value",
    "y": "y_value",
    "best_y": "best_y_value_so_far",
}

After that, we can create a summary table with the best_y
of each model for each function. The structure of this table
goes like this:

{
    "function": "function_name",
    "model_1": "best_y_value",
    "model_2": "best_y_value",
    ...
    "model_n": "best_y_value",
}
"""

import pandas as pd
import wandb

from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.utils.results.download_from_wandb import (
    get_all_runs_for_function_names,
)


def convert_data_to_dataframes(
    all_data: list[wandb.apis.public.Run],
) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for i, run in enumerate(all_data):
        print(f"Processing run {i + 1}/{len(all_data)}")
        df = run.history()
        df["solver_name"] = run.config["solver_name"]
        df["function_name"] = run.config["function_name"]
        df["seed"] = run.config["seed"]
        df["poli_hash"] = run.config["poli_hash"]
        df["hdbo_benchmark_hash"] = run.config["hdbo_benchmark_hash"]
        df["poli_baselines_hash"] = run.config["poli_baselines_hash"]
        df["experiment_id"] = run.config["experiment_id"]
        df["state"] = run.state
        dfs.append(df)

    return dfs


def create_base_table_for_ehrlich(
    save_cache: bool = True,
    use_cache: bool = False,
    tags: list[str] | None = None,
) -> pd.DataFrame:
    function_names = [
        "ehrlich_holo_tiny",
        "ehrlich_holo_small",
        "ehrlich_holo_large",
        "pest_control_equivalent",
    ]

    CACHE_PATH = ROOT_DIR / "data" / "results_cache"
    CACHE_PATH.mkdir(exist_ok=True, parents=True)
    tags_str = "-".join(tags) if tags is not None else "all"
    CACHE_FILE = CACHE_PATH / f"base_table_ehrlich-tags-{tags_str}.csv"

    if use_cache and CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE)
        return df

    all_runs = get_all_runs_for_function_names(function_names=function_names, tags=tags)
    all_dfs = convert_data_to_dataframes(all_runs)

    df = pd.concat(all_dfs)
    if save_cache:
        df.to_csv(CACHE_FILE, index=False)

    return df


if __name__ == "__main__":
    df = create_base_table_for_ehrlich(
        save_cache=True,
        use_cache=True,
        tags=None,
    )
    print(df)
