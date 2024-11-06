from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
import torch

from .constants import FONT_SIZE, SQUARE_FIG_SIZE

sns.set_style("darkgrid")


def plot_comparison_between_actual_and_predicted_values(
    actual_values: Union[np.ndarray, torch.Tensor],
    predicted_values: Union[np.ndarray, torch.Tensor],
    error_bars: Union[np.ndarray, torch.Tensor],
    figure_path: Optional[Path],
    n_dimensions: int,
    n_points: int,
    ax: Optional[plt.Axes] = None,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
    padding: Optional[float] = None,
    linewidth: Optional[float] = None,
    markersize: int = 10,
) -> plt.Figure:
    if isinstance(actual_values, torch.Tensor):
        actual_values = actual_values.numpy(force=True)

    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.numpy(force=True)

    if isinstance(error_bars, torch.Tensor):
        error_bars = error_bars.numpy(force=True)

    if min_ is None:
        min_ = min(np.concatenate((predicted_values, actual_values)))
    if max_ is None:
        max_ = max(np.concatenate((predicted_values, actual_values)))
    if padding is None:
        padding = 0.05 * (max_ - min_)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIG_SIZE)
    else:
        fig = ax.get_figure()  # type: ignore

    ax.errorbar(
        x=actual_values,
        y=predicted_values,
        yerr=error_bars,
        fmt=".k",
        label="mean predictions vs. actual values",
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor="gray",
        alpha=0.85,
        # errorbar=test_predictions_95.numpy(force=True),
        linewidth=linewidth,
    )

    ax.plot(
        np.linspace(min_ - padding, max_ + padding, 100),
        np.linspace(min_ - padding, max_ + padding, 100),
        color="green",
        label="y=x",
        linestyle="--",
        alpha=0.5,
        linewidth=linewidth,
    )
    ax.set_xlabel("actual values")
    ax.set_ylabel("mean predictions")
    ax.set_title(f"nr. dimensions: {n_dimensions}, nr. points: {n_points}")
    ax.legend()
    ax.set_xlim(min_ - padding, max_ + padding)
    ax.set_ylim(min_ - padding, max_ + padding)

    if figure_path:
        plt.tight_layout()
        fig.savefig(figure_path, dpi=300)

    return fig
