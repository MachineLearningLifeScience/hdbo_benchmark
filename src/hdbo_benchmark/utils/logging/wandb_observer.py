"""A simple example of how to log objective function calls using wandb.

To run this example, you will need to install wandb:

    pip install wandb
"""

# mypy: disable-error-code="import-untyped"
from typing import Literal
from dataclasses import dataclass

import numpy as np
import wandb

import poli
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.abstract_observer import AbstractObserver

import poli_baselines

import hdbo_benchmark

from hdbo_benchmark.utils.constants import WANDB_PROJECT, WANDB_ENTITY
from hdbo_benchmark.utils.logging.library_hashes import get_git_hash_of_library
from hdbo_benchmark.utils.logging.uncommited_changes import has_uncommitted_changes


@dataclass
class ObserverConfig:
    experiment_name: str
    function_name: str
    solver_name: str
    n_dimensions: int
    seed: int
    max_iter: int
    strict_on_hash: bool
    force_run: bool
    experiment_id: str
    wandb_mode: str
    tags: list[str]


class WandbObserver(AbstractObserver):
    def __init__(
        self, project_name: str | None = None, allow_reinit: bool = False
    ) -> None:
        # Log into wandb
        wandb.login()

        # Some variables to keep track of the run
        self.best_y = -float("inf")
        self.run = None

        self.project_name = project_name
        self.allow_reinit = allow_reinit
        super().__init__()

    def initialize_observer(
        self,
        black_box_info: BlackBoxInformation,
        caller_info: ObserverConfig,
        seed: int,
        mode: Literal["online", "offline", "disabled"] = "online",
        **kwargs,
    ) -> object:
        run_name = f"{caller_info.experiment_name}-{caller_info.solver_name}-{caller_info.function_name}-n_dimensions-{caller_info.n_dimensions}-seed-{seed}"
        run = wandb.init(
            project=self.project_name if self.project_name else WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                **caller_info.__dict__,
                **kwargs,
            },
            name=run_name,
            tags=caller_info.tags,
            reinit=self.allow_reinit,
            mode=mode,
        )
        self.run = run  # type: ignore[assignment]

        return run

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # This assumes that y is [b, 1]
        for x_i, y_i in zip(x.tolist(), y.flatten().tolist()):
            if y_i > self.best_y:
                self.best_y = y_i
            self.run.log(  # type: ignore[attr-defined]
                {
                    "x": x_i,
                    "y": y_i,
                    "best_y": self.best_y,
                }
            )

    def log_loss(self, training_loss: float, testing_loss: float) -> None:
        self.run.log(  # type: ignore[attr-defined]
            {
                "training_loss": training_loss,
                "testing_loss": testing_loss,
            }
        )

    def finish(self) -> None:
        wandb.finish()


def initialize_observer(
    problem: Problem,
    observer_config: ObserverConfig,
) -> WandbObserver:
    experiment_name = observer_config.experiment_name
    n_dimensions = observer_config.n_dimensions
    seed = observer_config.seed
    max_iter = observer_config.max_iter
    mode = observer_config.wandb_mode
    tag = observer_config.tags

    wandb_observer = WandbObserver(project_name=experiment_name)
    wandb_observer.initialize_observer(
        black_box_info=problem.info,
        caller_info=observer_config,
        seed=seed,
        hdbo_benchmark_hash=(
            get_git_hash_of_library(hdbo_benchmark)
            if not has_uncommitted_changes(hdbo_benchmark)
            else None
        ),
        poli_hash=(
            get_git_hash_of_library(poli) if not has_uncommitted_changes(poli) else None
        ),
        poli_baselines_hash=(
            get_git_hash_of_library(poli_baselines)
            if not has_uncommitted_changes(poli_baselines)
            else None
        ),
        n_dimensions=n_dimensions,
        max_iter=max_iter,
        mode=mode,
        tag=tag,
    )

    return wandb_observer
