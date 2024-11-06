"""Utilities for loading the results of the experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import wandb
from hdbo_benchmark.utils.constants import WANDB_ENTITY

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"


def load_all_results(
    experiment_name: str, solver_name: str, function_name: str, n_dimensions: int
) -> List[Dict[str, Any]]:
    """Load all the results of a solver."""
    results_paths = []
    if (DATA_DIR / experiment_name / solver_name).exists():
        for path in (
            DATA_DIR
            / experiment_name
            / solver_name
            / f"{function_name}_n_dimensions_{n_dimensions}"
        ).glob("*.json"):
            results_paths.append(path)
    else:
        raise ValueError(f"Couldn't find {solver_name} in {DATA_DIR / experiment_name}")

    results = []
    for path in results_paths:
        with open(path, "r") as f:
            results.append(json.load(f))

    return results


def load_results_as_dataframe(
    experiment_name: str, solver_name: str, function_name: str, n_dimensions: int
) -> pd.DataFrame:
    """Parse the results of a solver as a dataframe."""
    results = load_all_results(
        experiment_name, solver_name, function_name, n_dimensions
    )

    # building the rows of the dataframe
    rows = []
    for result in results:
        # Appending x0 and y0
        x0 = result["metadata"]["x0"][0]
        y0 = result["metadata"]["y0"][0][0]
        best_y_so_far = y0
        rows.append(
            {
                "solver": result["metadata"]["solver"],
                "seed": result["metadata"]["seed"],
                "experiment_id": result["metadata"]["experiment_id"],
                "iteration": 0,
                "x": x0,
                "y": y0,
                "best_y_so_far": best_y_so_far,
            }
        )

        # Appending the rest
        for i, (x, y) in enumerate(zip(result["x"], result["y"])):
            if y > best_y_so_far:
                best_y_so_far = y

            rows.append(
                {
                    "solver": result["metadata"]["solver"],
                    "seed": result["metadata"]["seed"],
                    "experiment_id": result["metadata"]["experiment_id"],
                    "iteration": i + 1,
                    "x": x,
                    "y": y,
                    "best_y_so_far": best_y_so_far,
                }
            )

    # building the dataframe
    df = pd.DataFrame(rows)

    return df


def load_results_as_dataframe_from_wandb(
    experiment_name: str, solver_name: str, function_name: str, n_dimensions: int
) -> pd.DataFrame:
    df = load_all_results_as_dataframe_from_wandb(experiment_name)
    df = df[df["solver"] == solver_name]
    df = df[df["function_name"] == function_name]
    df = df[df["n_dimensions"] == n_dimensions]

    return df


def _load_run_as_row(run: Any) -> list[dict[str, str | int | float]]:
    run_history = run.history()
    if run_history.empty:
        return []

    return [
        {
            "solver": run.config["solver_name"],
            "seed": run.config["seed"],
            "function_name": run.config["function_name"],
            "run_name": run.name,
            "n_dimensions": int(run.name.split("-")[3]),
            "iteration": step,
            "x": x,
            "y": y,
            "best_y_so_far": best_y,
        }
        for step, x, y, best_y in zip(
            run_history["_step"],
            run_history["x"],
            run_history["y"],
            run_history["best_y"],
        )
    ]


def load_all_results_as_dataframe_from_wandb(experiment_name) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{experiment_name}")

    rows: list[dict[str, str | int | float]] = []
    for i, run in enumerate(runs):
        print(i)
        rows.extend(_load_run_as_row(run))

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    df = load_all_results_as_dataframe_from_wandb(
        "benchmark_on_low_intrinsic_dimensionality"
    )

    print(df)
