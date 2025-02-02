"""
Scripts for backing up entire projects in WANDB.

Taken from wandb docs:
https://docs.wandb.ai/guides/track/public-api-guide#export-data
"""

from time import time

import pandas as pd
import wandb

from hdbo_benchmark.utils.constants import ROOT_DIR, WANDB_ENTITY, WANDB_PROJECT
from hdbo_benchmark.utils.data.experiments.loading_results import _load_run_as_row


def back_up(project: str = WANDB_PROJECT):
    api = wandb.Api()
    entity = WANDB_ENTITY
    runs = api.runs(entity + "/" + project)

    BACKUP_DIR = ROOT_DIR / "data" / "backups" / project
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, run in enumerate(runs):
        print(f"Backing up run {i + 1}/{len(runs)}")
        row = _load_run_as_row(run)
        rows.extend(row)

    runs_df = pd.DataFrame(rows)

    runs_df.to_csv(BACKUP_DIR / f"{project}_{int(time())}.csv")


if __name__ == "__main__":
    back_up("benchmark_on_pmo")
