import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hdbo_benchmark.utils.constants import ROOT_DIR

sns.set_theme(style="whitegrid", font_scale=2.8)
matplotlib.rcParams.update({"text.usetex": True})

max_sequence_length_solvers = {
    r"\texttt{Combo}": 60,  # wMaxSAT60
    r"\texttt{BOSS}": 3672,  # Protein MFE minimization
    r"\texttt{CoCaBo}": 5,  # The categorical part of NAS-CIFAR10
    r"\texttt{BODi}": 60,  # wMaxSAT60
    r"\texttt{ProbRep}": 50,  # SVM feature selection
    r"\texttt{Bounce}": 125,  # 125D Cluster
    r"\texttt{Casmo}": 80,  # DifficultPest.
    # r"\texttt{LOL-BO}$^*$": 70,  # Not reported explicitly, assuming similar zinc250k postprocessings.
    r"\texttt{LaMBO}": 228,  # Longest RFP sequence.
    # r"\texttt{LADDER}$^*$": 70,  # Not reported explicitly, assuming similar zinc250k postprocessings.
    # r"\texttt{WR}$^*$": 70,
    # r"\texttt{DML}": [...],
}

max_dict_size_solvers = {
    r"\texttt{Combo}": 2,
    r"\texttt{BOSS}": 20,
    r"\texttt{CoCaBo}": 3,
    r"\texttt{BODi}": 2,
    r"\texttt{ProbRep}": 2,
    r"\texttt{Bounce}": 2,
    r"\texttt{Casmo}": 5,
    # r"\texttt{LOL-BO}$^*$": 64,
    r"\texttt{LaMBO}": 20,
    # r"\texttt{LADDER}$^*$": 64,
    # r"\texttt{WR}$^*$": 64,
    # r"\texttt{DML}": [...],
}

max_sequence_length_problems = {
    "Prot.": 228,  # Longest RFP sequence.
    "Chem.": 70,
}
max_dict_size_problems = {
    "Prot.": 20,
    "Chem.": 64,
}

rows_solvers = [
    (solver_name, sequence_length, dict_size)
    for solver_name, sequence_length, dict_size in zip(
        max_sequence_length_solvers.keys(),
        max_sequence_length_solvers.values(),
        max_dict_size_solvers.values(),
    )
]
rows_problems = [
    (solver_name, sequence_length, dict_size)
    for solver_name, sequence_length, dict_size in zip(
        max_sequence_length_problems.keys(),
        max_sequence_length_problems.values(),
        max_dict_size_problems.values(),
    )
]

df_solvers = pd.DataFrame(
    rows_solvers, columns=["Solver", "Max. Sequence Length", "Nr. of Categories"]
)
df_problems = pd.DataFrame(
    rows_problems, columns=["Problem", "Max. Sequence Length", "Nr. of Categories"]
)

fig, axes = plt.subplot_mosaic(
    [["A", "A", "A", "A", "B"]], figsize=(20, 6), sharey=True
)
ax_solvers = axes["A"]
ax_problems = axes["B"]

# Let's make a barplot with two bars per solver:

# First, we need to melt the dataframe to have the sequence length and dictionary size in the same column.
df_solvers_melted = df_solvers.melt(
    id_vars="Solver", var_name="Metric", value_name="Value"
)
df_problems_melted = df_problems.melt(
    id_vars="Problem", var_name="Metric", value_name="Value"
)

# Now we can plot it.
sns.barplot(
    x="Solver",
    y="Value",
    hue="Metric",
    data=df_solvers_melted,
    ax=ax_solvers,
)
ax_solvers.set_yscale("log")
sns.barplot(
    x="Problem",
    y="Value",
    hue="Metric",
    data=df_problems_melted,
    ax=ax_problems,
)
# ax.set_ylabel("Value")
# ax.set_xlabel("Solver")
# ax.axhline(np.log10(70), color="blue", linestyle="--", linewidth=2)
# ax.axhline(np.log10(64), color="orange", linestyle="dotted", linewidth=2)

ax_solvers.set_ylim([0, 10**4])
ax_solvers.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, 1.15))
ax_problems.get_legend().remove()

# Adding a vertical line to separate prot and chem from the rest
# ax.axvline(7.5, color="black", linestyle="--", linewidth=2)

# Annotating that one side is solvers, and the other is problems
# ax.text(6.9, 10**3 + 700, "Solvers", ha="center")
# ax.text(8.5, 10**3 + 700, "Problems", ha="center")

fig.tight_layout()
fig.savefig(
    ROOT_DIR
    / "reports"
    / "figures"
    / "comparison_between_sequence_lengths_and_dict_sizes.jpg",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
