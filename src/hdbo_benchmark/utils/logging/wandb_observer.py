"""A simple example of how to log objective function calls using wandb.

To run this example, you will need to install wandb:

    pip install wandb
"""

# mypy: disable-error-code="import-untyped"
from typing import Literal

import numpy as np
import wandb

import poli
from poli.core.black_box_information import BlackBoxInformation
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.abstract_observer import AbstractObserver

import poli_baselines

import hdbo_benchmark

from hdbo_benchmark.utils.constants import WANDB_PROJECT, WANDB_ENTITY
from hdbo_benchmark.utils.logging.library_hashes import get_git_hash_of_library


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
        observer_init_info: dict[str, str],
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
        mode: Literal["online", "offline", "disabled"] = "online",
        **kwargs,
    ) -> object:
        run = wandb.init(
            project=self.project_name if self.project_name else WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config={
                "name": black_box_info.name,
                "x0": x0,
                "y0": y0,
                "seed": seed,
                "solver_name": observer_init_info["solver_name"],
                **kwargs,
            },
            name=observer_init_info["run_name"],
            tags=[kwargs.get("tag", "default")],
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
    experiment_name: str,
    f: AbstractBlackBox,
    function_name: str,
    solver_name: str,
    n_dimensions: int,
    seed: int,
    experiment_id: str,
    max_iter: int,
    strict_on_hash: bool,
    mode: str = "online",
    tag: str = "default",
) -> WandbObserver:
    wandb_observer = WandbObserver(project_name=experiment_name)
    wandb_observer.initialize_observer(
        black_box_info=f.info,
        observer_init_info={
            "run_name": f"{function_name}-{solver_name}-n_dimensions-{n_dimensions}-seed-{seed}-exp_id-{experiment_id}",
            "solver_name": solver_name,
        },
        x0=np.array([[]]),
        y0=np.array([[]]),
        seed=seed,
        function_name=function_name,
        hdbo_benchmark_hash=(
            get_git_hash_of_library(hdbo_benchmark) if strict_on_hash else None
        ),
        poli_hash=get_git_hash_of_library(poli) if strict_on_hash else None,
        poli_baselines_hash=(
            get_git_hash_of_library(poli_baselines) if strict_on_hash else None
        ),
        n_dimensions=n_dimensions,
        max_iter=max_iter,
        mode=mode,
        tag=tag,
    )

    return wandb_observer
