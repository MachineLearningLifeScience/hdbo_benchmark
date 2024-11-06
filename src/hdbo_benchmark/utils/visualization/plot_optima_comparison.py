import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import]


from hdbo_benchmark.utils.data.experiments import load_results_as_dataframe

from .colors import COLORS

sns.set_theme(style="whitegrid")


def plot_optima_comparison(
    ax: plt.Axes,
    experiment_name: str,
    function_name: str,
    n_dimensions: int,
    methods: list[str],
    global_maximum: float = 0.0,
    max_iteration: int = 200,
    n_starting_points: int = 10,
):
    dfs = []
    for method in methods:
        dfs.append(
            load_results_as_dataframe(
                experiment_name, method, function_name, n_dimensions
            )
        )

    df = pd.concat(dfs)
    plot_optima_comparison_from_df(
        ax,
        df,
        global_maximum=global_maximum,
        max_iteration=max_iteration,
        n_starting_points=n_starting_points,
    )


def plot_optima_comparison_from_df(
    ax: plt.Axes,
    df: pd.DataFrame,
    global_maximum: float = 0.0,
    max_iteration: int = 200,
    n_starting_points: int = 10,
):
    methods = df["solver"].unique()

    # Let's get average and std of the best y so far
    df_for_results = df.groupby(["solver", "iteration"])[["best_y_so_far", "y"]]

    # Filter by max iteration
    # df_for_results = df_for_results.filter(lambda x: x.name[1] <= max_iteration)

    # Compute mean, min and max
    mean = df_for_results.mean()
    min_ = df_for_results.min()
    max_ = df_for_results.max()

    # Filter by max iteration, and by starting number of points
    mean = mean[
        mean.index.get_level_values("iteration") <= max_iteration + n_starting_points
    ]
    max_ = max_[
        max_.index.get_level_values("iteration") <= max_iteration + n_starting_points
    ]
    min_ = min_[
        min_.index.get_level_values("iteration") <= max_iteration + n_starting_points
    ]

    mean = mean[mean.index.get_level_values("iteration") > n_starting_points]
    max_ = max_[max_.index.get_level_values("iteration") > n_starting_points]
    min_ = min_[min_.index.get_level_values("iteration") > n_starting_points]

    for solver, color in zip(methods, COLORS):
        # Let's plot the mean
        sns.lineplot(
            data=mean.loc[solver],
            x="iteration",
            y="best_y_so_far",
            label=solver,
            ax=ax,
            color=color,
        )

        # Let's plot the std
        lower_bound = min_.loc[solver]["best_y_so_far"]
        upper_bound = max_.loc[solver]["best_y_so_far"]
        upper_bound = [min(x, global_maximum) for x in upper_bound]
        ax.fill_between(
            mean.loc[solver].index,
            lower_bound,
            upper_bound,
            alpha=0.1,
            color=color,
        )

    # Let's add a horizontal line at the global maximum
    ax.axhline(
        y=global_maximum,
        color="black",
        linestyle="--",
        label="Global maximum",
    )

    ax.set_xlabel("Nr. function evaluations")
    ax.set_ylabel("Obj. value")
