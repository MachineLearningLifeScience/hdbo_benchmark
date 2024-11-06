"""
This script verifies that Vanilla BO works in 2D. In the process,
we visualize the acquisition function and the posterior mean and variance.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from poli import objective_factory  # type: ignore
from poli_baselines.core.utils.visualization.objective_functions import (  # type: ignore
    plot_objective_function,
)
from poli_baselines.solvers import VanillaBayesianOptimization  # type: ignore

warnings.filterwarnings("ignore", module="botorch")

if __name__ == "__main__":
    problem_info, f_ackley, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=2,
    )

    x0 = np.random.randn(2).reshape(1, -1).clip(-2.0, 2.0)
    y0 = f_ackley(x0)

    vanilla_bo = VanillaBayesianOptimization(
        black_box=f_ackley,
        x0=x0,
        y0=y0,
    )

    # Usually, one could just run:
    # line_bo.solve(max_iter=10)

    # But since we want to visualize what's happening, we'll run it
    # step by step:
    _, (
        ax_objective_function,
        ax_model_prediction,
        ax_acquisition,
    ) = plt.subplots(1, 3, figsize=(3 * 5, 5))
    for _ in range(20):
        # At each step, "x = solver.next_candidate()" is called. In
        # the case of BO-related implementations, this step updates
        # the underlying GP model, and maximizes the acquisition
        # function to find the next candidate solution.

        # The GP model can be found under solver.model
        vanilla_bo.step()

        # Plotting the objective
        plot_objective_function(
            f_ackley,
            ax=ax_objective_function,
            limits=vanilla_bo.bounds,
            cmap="jet",
        )

        # Plotting the GP model's predictions
        vanilla_bo.plot_model_predictions(ax=ax_model_prediction)

        # Plotting the acquisition function in the current random line
        vanilla_bo.plot_acquisition_function(ax=ax_acquisition)

        # Animating the plot
        ax_objective_function.set_title("Objective function")
        ax_model_prediction.set_title("GP model predictions")
        ax_acquisition.set_title("Acquisition function")
        plt.tight_layout()
        plt.pause(1)

        # Clearing the axes
        ax_objective_function.clear()
        ax_model_prediction.clear()
        ax_acquisition.clear()
