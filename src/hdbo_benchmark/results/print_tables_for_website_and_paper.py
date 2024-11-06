"""
When run, this script downloads all the tables for the website.
"""

import json

import numpy as np

import pandas as pd

from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.results.download_tables import create_base_table_for_ehrlich

pd.set_option("display.max_colwidth", None)

# N_DIMENSIONS = 128
COLOR_IN_TABLE = "Green"


def compute_pretty_names_for_solvers(use_tex: bool = True):
    solver_name_but_pretty = {
        "directed_evolution": (
            r"\texttt{DirectedEvolution}" if use_tex else "DirectedEvolution"
        ),
        "hill_climbing": (r"\texttt{HillClimbing}" if use_tex else "HillClimbing"),
        "cma_es": r"\texttt{CMAES}" if use_tex else "CMAES",
        "genetic_algorithm": (
            r"\texttt{GeneticAlgorithm}" if use_tex else "GeneticAlgorithm"
        ),
        "vanilla_bo_hvarfner": (
            r"Hvarfner's \texttt{VanillaBO}" if use_tex else "Hvarfner's VanillaBO"
        ),
        "random_line_bo": r"\texttt{RandomLineBO}" if use_tex else "RandomLineBO",
        # "coordinate_line_bo": (
        #     r"\texttt{CoordinateLineBO}" if use_tex else "CoordinateLineBO"
        # ),
        "saas_bo": r"\texttt{SAASBO}" if use_tex else "SAASBO",
        # "alebo": r"\texttt{ALEBO}",
        "turbo": r"\texttt{Turbo}" if use_tex else "Turbo",
        "baxus": r"\texttt{BAxUS}" if use_tex else "BAxUS",
        "bounce": r"\texttt{Bounce}" if use_tex else "Bounce",
        "pr": r"\texttt{ProbRep}" if use_tex else "ProbRep",
        # "lambo2": r"\texttt{Lambo2}" if use_tex else "Lambo2",
    }

    return solver_name_but_pretty


def compute_pretty_names_for_functions():
    return {
        "pest_control_equivalent": "PestControlEquiv",
        "ehrlich_holo_tiny": "Ehrlich(L=5)",
        "ehrlich_holo_small": "Ehrlich(L=15)",
        "ehrlich_holo_large": "Ehrlich(L=64)",
    }


def select_runs(df: pd.DataFrame, function_name: str, solver_name: str) -> pd.DataFrame:
    sliced_df = df[
        (df["function_name"] == function_name) & (df["solver_name"] == solver_name)
    ]
    sliced_df = sliced_df[sliced_df["seed"].isin([1, 2, 3, 4, 5])]

    max_iter = 1299 if function_name == "ehrlich_holo_large" else 309
    sliced_df = sliced_df[sliced_df["_step"] <= max_iter]

    # If the number of experiments is greater than 5, we need to select the "longest" 5,
    # i.e. the ones in which the number of iterations is the greatest (for each seed)
    if len(sliced_df["experiment_id"].unique()) > 5:
        slices_per_seed = []
        for seed_ in range(1, 6):
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
        assert len(sliced_df["seed"].unique()) == 5
        for i in range(1, 6):
            assert i in sliced_df["seed"].unique()

    return sliced_df


def summary_per_function(
    df: pd.DataFrame,
    normalized_per_row: bool = True,
    use_tex: bool = True,
    normalize_with_max_value: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver_name_but_pretty = compute_pretty_names_for_solvers(use_tex=use_tex)

    rows = []
    missing_experiments = []
    for function_name in df["function_name"].unique():
        for solver_name in df["solver_name"].unique():
            slice_df = select_runs(df, function_name, solver_name)
            if (
                len(slice_df["seed"].unique()) != 5
                and solver_name in solver_name_but_pretty.keys()
            ):
                print(
                    f"Something's fishy with {function_name} and {solver_name} (seeds: {slice_df['seed'].unique()})"
                )
                missing_seeds = set([1, 2, 3, 4, 5]) - set(slice_df["seed"].unique())
                missing_experiments.append(
                    {
                        "function_name": function_name,
                        "solver_name": solver_name,
                        "missing_seeds": list(missing_seeds),
                    }
                )

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
            if normalize_with_max_value is not None:
                best_value = normalize_with_max_value
            else:
                best_value = row.max()
            lowest_value = row.min()
            if best_value == 0:
                continue
            summary_avg.loc[i] = (row - lowest_value) / (best_value - lowest_value)

    return summary_avg, summary_std, missing_experiments


def print_table_as_tex(
    df,
    normalized: bool = False,
    transpose: bool = False,
    use_tex: bool = True,
    include_color: bool = True,
):
    solver_name_but_pretty = compute_pretty_names_for_solvers(use_tex=use_tex)
    problem_name_but_pretty = compute_pretty_names_for_functions()
    index_name = "Solver" + r"\textbackslash " + "Oracle" if transpose else "Oracle"
    summary_avg, summary_std, missing_experiments = summary_per_function(
        df, normalized_per_row=normalized
    )
    summary_avg_normalized, _, _ = summary_per_function(
        df, normalized_per_row=True, normalize_with_max_value=1.0
    )

    final_table_rows: list[dict[str, str]] = []
    for function_name, pretty_function_name in problem_name_but_pretty.items():
        row = {
            index_name: pretty_function_name,
        }
        for solver_name, pretty_solver_name in solver_name_but_pretty.items():
            if solver_name not in summary_avg.columns:
                row[pretty_solver_name] = r"\alert{[TBD]}"
                continue

            average = summary_avg.loc[function_name, solver_name]
            std = summary_std.loc[function_name, solver_name]

            if np.isnan(average):
                row[pretty_solver_name] = r"\alert{[TBD]}"
            else:
                avg = f"{average:.3f}"
                std = f"{std:.2f}" if not np.isnan(std) else r"\alert{?}"
                normalized_avg = summary_avg_normalized.loc[function_name, solver_name]
                cell_color_str = (
                    r"\cellcolor{"
                    + COLOR_IN_TABLE
                    + "!"
                    + f"{int(50 * normalized_avg)}"
                    + "}"
                )
                if use_tex:
                    if include_color:
                        cell_string = (
                            cell_color_str + f"${avg}"
                            r"{\pm \scriptstyle " + f"{std}" + "}$"
                        )
                    else:
                        cell_string = f"${avg}" r"{\pm \scriptstyle " + f"{std}" + "}$"
                else:
                    cell_string = f"{avg} +/- {std}"

                row[pretty_solver_name] = cell_string

        final_table_rows.append(row)

    final_table = pd.DataFrame(final_table_rows)
    final_table.set_index(index_name, inplace=True)

    # Let's add a final row in whcih we sum the (normalized) scores of each solver
    row = {
        index_name: "Sum (normalized per row)",
    }
    ranks = {}
    for solver_name, pretty_solver_name in solver_name_but_pretty.items():
        if solver_name not in summary_avg.columns:
            row[pretty_solver_name] = r"\alert{[TBD]}"
            continue

        solver_score = summary_avg_normalized.loc[:, solver_name].sum()
        sum_std_scores = summary_std.loc[:, solver_name].sum()
        ranks[solver_name] = solver_score
        if use_tex:
            sum_cell_string = (
                f"${solver_score:.2f}"
                + r"{\pm \scriptstyle "
                + f"{sum_std_scores:.2f}"
                + "}$"
            )
        else:
            sum_cell_string = f"{solver_score:.2f}" + " +/- " + f"{sum_std_scores:.2f}"

        row[pretty_solver_name] = sum_cell_string

    # Let's compute the colors, normalizing the ranks
    ranks = pd.Series(ranks)
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    for solver_name, pretty_solver_name in solver_name_but_pretty.items():
        if solver_name not in summary_avg.columns:
            continue
        rank = ranks[solver_name]
        if use_tex and include_color:
            cell_color_str = (
                r"\cellcolor{" + COLOR_IN_TABLE + "!" + f"{int(50 * rank)}" + "}"
            )
            row[pretty_solver_name] = cell_color_str + row[pretty_solver_name]

    new_row = pd.DataFrame(row, index=[0])
    new_row.set_index(index_name, inplace=True)

    final_table = pd.concat([final_table, new_row])
    # final_table = final_table.append(row, ignore_index=True)

    if transpose:
        final_table = final_table.T

    final_table.to_csv(ROOT_DIR / "data" / "results_cache" / "table_ehrlich.csv")

    latex_table = final_table.to_latex(escape=False)
    latex_table = r"\resizebox{\textwidth}{!}{" + latex_table + "}"

    print(latex_table)

    with open(ROOT_DIR / "data" / "results_cache" / "table_ehrlich.tex", "w") as f:
        f.write(latex_table)

    with open(
        ROOT_DIR / "data" / "results_cache" / "missing_experiments_ehrlich.json", "w"
    ) as f:
        json.dump(missing_experiments, f)


def print_table_for_ehrlich(use_tex: bool = True, include_color: bool = True):
    df = create_base_table_for_ehrlich(save_cache=True, use_cache=True)

    print_table_as_tex(
        df,
        transpose=True,
        use_tex=use_tex,
        include_color=include_color,
    )


if __name__ == "__main__":
    print_table_for_ehrlich(use_tex=True, include_color=True)
    print_table_for_ehrlich(use_tex=False, include_color=False)
