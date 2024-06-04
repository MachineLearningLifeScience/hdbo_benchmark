from uuid import uuid4
import json
import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver

from hdbo_benchmark.utils.constants import ROOT_DIR


class BenchmarkObserver(AbstractObserver):
    def __init__(self, experiment_name: str, experiment_id: str | None = None):
        if experiment_id is None:
            experiment_id = str(uuid4())[:16]

        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

        # Creating a local directory for the results
        experiment_path = ROOT_DIR / "results" / experiment_name / experiment_id
        experiment_path.mkdir(exist_ok=True, parents=True)

        self.experiment_path = experiment_path
        self.results: list[dict[str, str | float | list[str] | list[float]]] = []

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
        **kwargs,
    ) -> object:
        # Saving the metadata for this experiment
        metadata = problem_setup_info.as_dict()

        # Adding the information the user wanted to provide
        # (Recall that this caller info gets forwarded
        # from the objective_factory.create function)
        metadata["caller_info"] = caller_info

        # Saving the initial evaluations and seed
        metadata["x0"] = x0.tolist()
        metadata["y0"] = y0.tolist()
        metadata["seed"] = seed

        metadata.update(kwargs)

        # Saving the metadata
        with open(self.experiment_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return self

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # Appending these results to the results file.
        self.results.append({"x": x.tolist(), "y": y.tolist()})
        with open(self.experiment_path / "results.json", "w") as f:
            json.dump(self.results, f)
