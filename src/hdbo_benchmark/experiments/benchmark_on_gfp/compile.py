from datetime import datetime

from hdbo_benchmark.utils.experiments.load_solvers import SOLVER_NAMES

current_date = datetime.now().strftime("%Y-%m-%d")

TEST_RUN = False

DISCETE_SOLVERS = ["pr", "bounce", "genetic_algorithm"]
max_iter = 5 if TEST_RUN else 500

tag = f"GFPExperiments-{current_date}"
if TEST_RUN:
    tag += "-debug"


def condition_to_write(solver_name, latent_dim) -> bool:
    return solver_name in [
        "random_mutation",
        "genetic_algorithm",
        "cma_es",
        # "line_bo",
        "vanilla_bo_hvarfner",
        "saas_bo",
        # "alebo",
        # "baxus",
        "bounce",
        "pr",
        "turbo",
    ]


for solver_name in SOLVER_NAMES[::-1]:
    if not condition_to_write(
        solver_name=solver_name,
    ):
        continue

    if solver_name == "baxus":
        conda_env_name = "poli__baxus"
    elif solver_name in ["bounce"]:
        conda_env_name = "poli__bounce"
    elif solver_name == "pr":
        conda_env_name = "poli__pr"
    elif solver_name in ["vanilla_bo_hvarfner", "saas_bo"]:
        conda_env_name = "poli__ax"
    else:
        conda_env_name = "hdbo_benchmark"

    command_ = f"sbatch src/hdbo_benchmark/experiments/benchmark_on_gfp/batch.sh {conda_env_name} {solver_name} {max_iter} {tag}"

    print(command_)