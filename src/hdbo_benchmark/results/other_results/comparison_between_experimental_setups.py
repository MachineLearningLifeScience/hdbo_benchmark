import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hdbo_benchmark.utils.constants import ROOT_DIR

sns.set_theme(style="whitegrid", font_scale=2.5)
matplotlib.rcParams.update({"text.usetex": True})

initialization = {
    r"Hvarfner's \texttt{VanillaBO}": [33, 40, 59, 89, 240],
    r"\texttt{RandomLineBO}": [],
    r"\texttt{SAASBO}": [10, 20],
    r"\texttt{Turbo}": [50, 100, 1000],
    r"\texttt{BAxUS}": [10],
    r"\texttt{Bounce}": [5],
    r"\texttt{ProbRep}": [20],
    r"\texttt{LOL-BO}": [10_000],
    r"\texttt{DML}": [500, 8000],
    r"\texttt{WR}": [10_000],
    r"\texttt{CoBo}": [100, 10_000, 40_000],
}

evaluation_budget = {
    r"Hvarfner's \texttt{VanillaBO}": [100, 200, 500, 1000],
    r"\texttt{RandomLineBO}": [1000],
    r"\texttt{BAxUS}": [100, 500, 1000],
    r"\texttt{SAASBO}": [50, 100, 400],
    r"\texttt{Bounce}": [200],
    r"\texttt{Turbo}": [10_000],
    r"\texttt{ProbRep}": [100, 200, 400],
    r"\texttt{LOL-BO}": [100, 500, 1000, 5000],
    r"\texttt{DML}": [500, 1000],
    r"\texttt{WR}": [500],
    r"\texttt{CoBo}": [500, 3_000, 70_000],
}

independent_replications = {
    r"Hvarfner's \texttt{VanillaBO}": [10, 20],
    r"\texttt{RandomLineBO}": [100],
    r"\texttt{BAxUS}": [20],
    r"\texttt{SAASBO}": [30],
    r"\texttt{Bounce}": [20, 50],
    r"\texttt{Turbo}": [50],
    r"\texttt{ProbRep}": [20],
    r"\texttt{LOL-BO}": [],
    r"\texttt{DML}": [5],
    r"\texttt{WR}": [3],
    r"\texttt{CoBo}": [3],
}


fig, ax = plt.subplots(1, 1, figsize=(20, 6))

dfs = []
for i, (title, data) in enumerate(
    [
        ("Initialization", initialization),
        ("Eval. Budget", evaluation_budget),
        ("Replications", independent_replications),
    ]
):
    df_ = pd.DataFrame(
        [(title, k, v) for k, vs in data.items() for v in vs],
        columns=["Type", "Solver", "Value"],
    )
    # df = pd.DataFrame(data)
    # df = df.melt(var_name="Solver", value_name="Value")
    # df_["Log-Value"] = np.log10(df_["Value"])
    # axs[i].set_title(title)

    dfs.append(df_)

df = pd.concat(dfs)

sns.swarmplot(
    data=df,
    x="Type",
    y="Value",
    ax=ax,
    hue="Solver",
    size=15,
    palette="tab20",
    linewidth=1.0,
)

ax.set_yscale("log")
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=2)
ax.set_xlabel("")
ax.set_ylabel("Value")
ax.set_ylim([0, 10**5])

plt.tight_layout()
plt.savefig(
    ROOT_DIR / "reports" / "figures" / "comparison_between_experimental_setups.jpg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
