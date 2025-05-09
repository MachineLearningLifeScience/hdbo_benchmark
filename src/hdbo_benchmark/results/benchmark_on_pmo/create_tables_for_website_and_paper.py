"""
When run, this script downloads all the tables for the website.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hdbo_benchmark.results.benchmark_on_pmo.create_table import create_base_table
from hdbo_benchmark.utils.constants import ROOT_DIR

pd.set_option("display.max_colwidth", None)

# N_DIMENSIONS = 128
COLOR_IN_TABLE = "Green"

matplotlib.rcParams.update({"text.usetex": True})
sns.set_theme(style="whitegrid", font_scale=1.75)


def compute_pretty_names(n_dimensions: int, use_tex: bool = True):
    match n_dimensions:
        case 2:
            solver_name_but_pretty = {
                "random_mutation": (
                    r"\texttt{HillClimbing}" if use_tex else "HillClimbing"
                ),
                "genetic_algorithm": (
                    r"\texttt{GeneticAlgorithm}" if use_tex else "GeneticAlgorithm"
                ),
                "cma_es": r"\texttt{CMAES}",
                "vanilla_bo_hvarfner": (
                    r"Hvarfner's \texttt{VanillaBO}"
                    if use_tex
                    else "Hvarfner's VanillaBO"
                ),
                "line_bo": r"\texttt{RandomLineBO}" if use_tex else "RandomLineBO",
                "saas_bo": r"\texttt{SAASBO}" if use_tex else "SAASBO",
                # "alebo": r"\texttt{ALEBO}",
                "turbo": r"\texttt{Turbo}" if use_tex else "Turbo",
                # "baxus": r"\texttt{BAxUS}",
                "bounce": r"\texttt{Bounce}" if use_tex else "Bounce",
                "pr": r"\texttt{ProbRep}" if use_tex else "ProbRep",
            }
        case 128:
            solver_name_but_pretty = {
                "random_mutation": (
                    r"\texttt{HillClimbing}" if use_tex else "HillClimbing"
                ),
                "genetic_algorithm": (
                    r"\texttt{GeneticAlgorithm}" if use_tex else "GeneticAlgorithm"
                ),
                "cma_es": r"\texttt{CMAES}",
                "vanilla_bo_hvarfner": (
                    r"Hvarfner's \texttt{VanillaBO}"
                    if use_tex
                    else "Hvarfner's VanillaBO"
                ),
                "line_bo": r"\texttt{RandomLineBO}" if use_tex else "RandomLineBO",
                "saas_bo": r"\texttt{SAASBO}" if use_tex else "SAASBO",
                # "alebo": r"\texttt{ALEBO}",
                "baxus": r"\texttt{BAxUS}" if use_tex else "BAxUS",
                "turbo": r"\texttt{Turbo}" if use_tex else "Turbo",
                "bounce": r"\texttt{Bounce}" if use_tex else "Bounce",
                "pr": r"\texttt{ProbRep}" if use_tex else "ProbRep",
            }
        case _:
            raise NotImplementedError()

    return solver_name_but_pretty


def select_runs(df: pd.DataFrame, function_name: str, solver_name: str) -> pd.DataFrame:
    sliced_df = df[
        (df["function_name"] == function_name) & (df["solver_name"] == solver_name)
    ]
    sliced_df = sliced_df[sliced_df["seed"].isin([1, 2, 3])]

    if len(sliced_df["experiment_id"].unique()) > 3:
        slices_per_seed = []
        for seed_ in range(1, 4):
            assert seed_ in sliced_df["seed"].unique()
            sliced_df_of_seed = sliced_df[sliced_df["seed"] == seed_]
            experiment_id_of_longest_runs = (
                sliced_df_of_seed.groupby("experiment_id")["_step"].max().nlargest(1)
            )
            print(
                f"Largest experiment id for {solver_name} in {function_name} (seed {seed_}): {experiment_id_of_longest_runs.index[0]} ({experiment_id_of_longest_runs.values[0]})"
            )
            slices_per_seed.append(
                sliced_df_of_seed[
                    sliced_df_of_seed["experiment_id"]
                    == experiment_id_of_longest_runs.index[0]
                ]
            )

        sliced_df = pd.concat(slices_per_seed)

        # Let's make sure we're still selecting the 5 seeds
        assert len(sliced_df["seed"].unique()) == 3
        for i in range(1, 4):
            assert i in sliced_df["seed"].unique()

    return sliced_df

    ...


def summary_per_function(
    df: pd.DataFrame,
    normalized_per_row: bool = True,
    n_dimensions: int = 2,
    use_tex: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver_name_but_pretty = compute_pretty_names(n_dimensions, use_tex=use_tex)

    rows = []
    for function_name in df["function_name"].unique():
        for solver_name in df["solver_name"].unique():
            slice_df = select_runs(df, function_name, solver_name)
            if (
                len(slice_df["seed"].unique()) != 3
                and solver_name in solver_name_but_pretty.keys()
            ):
                # Only prints for one function (valsartan_smarts) in Bounce.
                # Should be fixed by Wednesday.
                print(
                    f"Something's fishy with {function_name} and {solver_name} ({slice_df['seed'].unique()})"
                )
            # assert len(slice_df["seed"].unique()) == 3
            best_y_per_seed = slice_df.groupby("seed")["y"].max()
            best_y = best_y_per_seed.mean()
            best_y_std = best_y_per_seed.std()
            rows.append(
                {
                    "function_name": function_name,
                    "solver_name": solver_name,
                    "average_best_y": best_y,
                    "std_best_y": best_y_std,
                }
            )

    summary = pd.DataFrame(rows)
    summary_avg = summary.pivot(
        index="function_name", columns="solver_name", values="average_best_y"
    )

    summary_std = summary.pivot(
        index="function_name", columns="solver_name", values="std_best_y"
    )

    # Normalize each row to be a percentage of the best value
    if normalized_per_row:
        for i, row in summary_avg.iterrows():
            best_value = row.max()
            lowest_value = row.min()
            if best_value == 0:
                continue
            summary_avg.loc[i] = (row - lowest_value) / (best_value - lowest_value)

    return summary_avg, summary_std


def plot_heatmap(df, normalized: bool = True, n_dimensions: int = 2):
    solver_name_but_pretty = compute_pretty_names(n_dimensions)
    summary_avg, _ = summary_per_function(df, normalized_per_row=normalized)

    # We keep the columns in solver_name_but_pretty order
    summary_avg = summary_avg[solver_name_but_pretty.keys()]

    # Rename columns to their pretty names
    summary_avg.columns = [solver_name_but_pretty[col] for col in summary_avg.columns]

    # Adjust the size of the figure to make squares smaller
    fig, ax = plt.subplots(1, 1, figsize=(17, 5))  # Adjust these numbers as needed

    # Capture the heatmap in a variable
    hmap = sns.heatmap(
        summary_avg.T,
        ax=ax,
        cmap="inferno",
        cbar_kws={"orientation": "vertical", "pad": 0.01},
    )

    # Modify the colorbar to only show min and max
    colorbar = hmap.collections[0].colorbar
    colorbar.set_ticks([colorbar.vmin, colorbar.vmax])
    colorbar.set_ticklabels(["min", "max"])

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(
        f"Avg. Best Value ({n_dimensions}D latent space, 3 seeds, max. 10+100 function calls)"
        + "\n",
        fontsize=25,
    )
    fig.tight_layout()
    fig.savefig(
        ROOT_DIR / "reports" / "figures" / f"table_as_heatmap_pmo_{n_dimensions}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


def compute_pmo_benchmark_table(
    df, n_dimensions: int, use_tex: bool = True
) -> pd.DataFrame:
    solver_name_but_pretty = compute_pretty_names(n_dimensions, use_tex=use_tex)
    summary_avg, summary_std = summary_per_function(
        df, normalized_per_row=normalized, n_dimensions=n_dimensions, use_tex=use_tex
    )
    summary_avg_normalized, _ = summary_per_function(
        df, normalized_per_row=True, n_dimensions=n_dimensions, use_tex=use_tex
    )

    final_table_rows: list[dict[str, str]] = []
    for function_name in summary_avg.index:
        row = {
            "Oracle": function_name.replace("_", r"\_") if use_tex else function_name,
        }
        for solver_name, pretty_solver_name in solver_name_but_pretty.items():
            if n_dimensions == 2 and solver_name == "baxus":
                continue
            if solver_name not in summary_avg.columns:
                row[pretty_solver_name] = r"[TBD]"
                continue

            average = summary_avg.loc[function_name, solver_name]
            std = summary_std.loc[function_name, solver_name]

            if np.isnan(average):
                row[pretty_solver_name] = r"[TBD]"
            else:
                avg = f"{average:.2f}"
                std = f"{std:.2f}" if not np.isnan(std) else r"\alert{?}"
                row[pretty_solver_name] = f"{avg} ± {std}"

        final_table_rows.append(row)

    final_table = pd.DataFrame(final_table_rows)
    final_table.set_index("Oracle", inplace=True)

    # Let's add a final row in whcih we sum the (normalized) scores of each solver
    row = {
        "Oracle": "Sum (normalized per row)",
    }
    ranks = {}
    for solver_name, pretty_solver_name in solver_name_but_pretty.items():
        if n_dimensions == 2 and solver_name == "baxus":
            continue
        if solver_name not in summary_avg.columns:
            row[pretty_solver_name] = r"[TBD]"
            continue

        solver_score = summary_avg_normalized.loc[:, solver_name].sum()
        sum_std_scores = summary_std.loc[:, solver_name].sum()
        ranks[solver_name] = solver_score
        row[pretty_solver_name] = f"{solver_score:.2f} ± {sum_std_scores:.2f}"

    return final_table


def print_table_as_tex(df, normalized: bool = False, n_dimensions: int = 2):
    solver_name_but_pretty = compute_pretty_names(n_dimensions)

    summary_avg, summary_std = summary_per_function(
        df, normalized_per_row=normalized, n_dimensions=n_dimensions
    )
    summary_avg_normalized, _ = summary_per_function(
        df, normalized_per_row=True, n_dimensions=n_dimensions
    )

    # solver_name_but_pretty = {
    #     "random_mutation": r"\texttt{HillClimbing}",
    #     # "cma_es": r"\texttt{CMAES}",
    #     "vanilla_bo_hvarfner": r"Hvarfner's \texttt{VanillaBO}",
    #     "line_bo": r"\texttt{RandomLineBO}",
    #     # "saas_bo": r"\texttt{SAASBO}",
    #     # "alebo": r"\texttt{ALEBO}",
    #     # "turbo": r"\texttt{Turbo}",
    #     "baxus": r"\texttt{BAxUS}",
    #     "bounce": r"\texttt{Bounce}",
    #     "pr": r"\texttt{ProbRep}",
    # }

    final_table_rows: list[dict[str, str]] = []
    for function_name in summary_avg.index:
        row = {
            "Oracle": function_name.replace("_", r"\_"),
        }
        for solver_name, pretty_solver_name in solver_name_but_pretty.items():
            if n_dimensions == 2 and solver_name == "baxus":
                continue
            if solver_name not in summary_avg.columns:
                row[pretty_solver_name] = r"\alert{[TBD]}"
                continue

            average = summary_avg.loc[function_name, solver_name]
            std = summary_std.loc[function_name, solver_name]

            if np.isnan(average):
                row[pretty_solver_name] = r"\alert{[TBD]}"
            else:
                avg = f"{average:.2f}"
                std = f"{std:.2f}" if not np.isnan(std) else r"\alert{?}"
                normalized_avg = summary_avg_normalized.loc[function_name, solver_name]
                cell_color_str = (
                    r"\cellcolor{"
                    + COLOR_IN_TABLE
                    + "!"
                    + f"{int(50 * normalized_avg)}"
                    + "}"
                )
                row[pretty_solver_name] = (
                    cell_color_str + f"${avg}" r"{\pm \scriptstyle " + f"{std}" + "}$"
                )

        final_table_rows.append(row)

    final_table = pd.DataFrame(final_table_rows)
    final_table.set_index("Oracle", inplace=True)

    # Let's add a final row in whcih we sum the (normalized) scores of each solver
    row = {
        "Oracle": "Sum (normalized per row)",
    }
    ranks = {}
    for solver_name, pretty_solver_name in solver_name_but_pretty.items():
        if n_dimensions == 2 and solver_name == "baxus":
            continue
        if solver_name not in summary_avg.columns:
            row[pretty_solver_name] = r"\alert{[TBD]}"
            continue

        solver_score = summary_avg_normalized.loc[:, solver_name].sum()
        sum_std_scores = summary_std.loc[:, solver_name].sum()
        ranks[solver_name] = solver_score
        row[pretty_solver_name] = (
            f"${solver_score:.2f}"
            + r"{\pm \scriptstyle "
            + f"{sum_std_scores:.2f}"
            + "}$"
        )

    # Let's compute the colors, normalizing the ranks
    ranks = pd.Series(ranks)
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    for solver_name, pretty_solver_name in solver_name_but_pretty.items():
        if n_dimensions == 2 and solver_name == "baxus":
            continue
        if solver_name not in summary_avg.columns:
            continue
        rank = ranks[solver_name]
        cell_color_str = (
            r"\cellcolor{" + COLOR_IN_TABLE + "!" + f"{int(50 * rank)}" + "}"
        )
        row[pretty_solver_name] = cell_color_str + row[pretty_solver_name]

    new_row = pd.DataFrame(row, index=[0])
    new_row.set_index("Oracle", inplace=True)

    final_table = pd.concat([final_table, new_row])
    # final_table = final_table.append(row, ignore_index=True)

    latex_table = final_table.to_latex(escape=False)
    latex_table = r"\resizebox{\textwidth}{!}{" + latex_table + "}"

    print(latex_table)

    with open(
        ROOT_DIR / "data" / "results_cache" / f"table_{n_dimensions}.tex", "w"
    ) as f:
        f.write(latex_table)

    # print(final_table)


if __name__ == "__main__":
    normalized = True
    # tags = ["2024-06-03", "2024-06-02", "2024-06-01", "2024-05-31", "Old-PR-Results"]
    tags: None = None
    for n_dimensions in [2, 128]:
        df = create_base_table(
            n_dimensions=n_dimensions,
            save_cache=True,
            use_cache=True,
            tags=tags,
            datetime_cutoff="2024-08-15T00:00:00",
        )
        max_iter = 310 if n_dimensions == 128 else 110
        df = df[df["_step"] <= max_iter]
        plot_heatmap(df, normalized=normalized)
        table_ = compute_pmo_benchmark_table(
            df, n_dimensions=n_dimensions, use_tex=False
        )
        table_.to_csv(f"table_{n_dimensions}.csv")
        print_table_as_tex(df, normalized=False, n_dimensions=n_dimensions)
