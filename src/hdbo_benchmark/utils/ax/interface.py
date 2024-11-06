import numpy as np
from ax import (  # type: ignore[import]
    Data,
    Experiment,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core.generator_run import GeneratorRun  # type: ignore[import]
from ax.core.metric import Metric  # type: ignore[import]
from ax.metrics.noisy_function import NoisyFunctionMetric  # type: ignore[import]
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]


def define_search_space(
    x0: np.ndarray, bounds: list[tuple[float, float]]
) -> SearchSpace:
    """
    Defines the search space for the optimization problem.

    Parameters
    ----------
    x0 : np.ndarray
        The initial point in the original space.
    bounds : list[tuple[float]]
        The lower and upper bounds for the optimization problem.

    Returns
    -------
    SearchSpace
        The search space for the optimization problem.
    """
    n_dimensions = x0.shape[1]
    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=bounds[i][0],
                upper=bounds[i][1],
            )
            for i in range(n_dimensions)
        ]
    )

    return search_space


def from_black_box_to_ax_metric(
    black_box: AbstractBlackBox, search_space: SearchSpace, noise_sd: float = 0.0
) -> NoisyFunctionMetric:
    class BlackBoxMetric(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> np.ndarray:
            val: np.ndarray = black_box(x.reshape(-1, len(self.param_names)))
            return val.flatten()

    return BlackBoxMetric(
        name=black_box.info.name,
        param_names=list(search_space.parameters.keys()),
        noise_sd=noise_sd,
        lower_is_better=False,
    )


def generator_run_from_initial_data(
    x0: np.ndarray, y0: np.ndarray, black_box_metric: Metric
):
    """
    Generates a run from the initial data.

    Parameters
    ----------
    x0 : np.ndarray
        The initial point in the original space.
    y0 : np.ndarray
        The initial observation.
    black_box_metric : Metric
        The metric for the optimization problem.

    Returns
    -------
    GeneratorRun
        The generator run from the initial data.
    """
    generator_run = GeneratorRun(
        arms=[
            black_box_metric.arm_from_dict(
                {f"x{i}": x0[j, i] for i in range(x0.shape[1])}
            )
            for j in range(x0.shape[0])
        ],
        weights=y0.flatten(),
    )

    return generator_run
