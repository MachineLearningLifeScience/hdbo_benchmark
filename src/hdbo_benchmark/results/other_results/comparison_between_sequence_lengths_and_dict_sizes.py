import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

from hdbo_benchmark.utils.constants import ROOT_DIR

sns.set_theme(style="whitegrid", font_scale=2.8)
matplotlib.rcParams.update({"text.usetex": True})

max_sequence_length = {
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

max_dict_size = {
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

rows = [
    (solver_name, np.log10(sequence_length), np.log10(dict_size))
    for solver_name, sequence_length, dict_size in zip(
        max_sequence_length.keys(), max_sequence_length.values(), max_dict_size.values()
    )
]

df = pd.DataFrame(rows, columns=["Solver", "Max. Sequence Length", "Nr. of Categories"])

fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Let's make a barplot with two bars per solver:

# First, we need to melt the dataframe to have the sequence length and dictionary size in the same column.
df_melted = df.melt(id_vars="Solver", var_name="Metric", value_name="Value")

# Now we can plot it.
sns.barplot(
    x="Solver",
    y="Value",
    hue="Metric",
    data=df_melted,
    ax=ax,
)
ax.set_ylabel("Log-Value")
ax.set_xlabel("")
# ax.axhline(np.log10(70), color="blue", linestyle="--", linewidth=2)
# ax.axhline(np.log10(64), color="orange", linestyle="dotted", linewidth=2)

ax.set_ylim([0, 4])
ax.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, 1.15))

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
