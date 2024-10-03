from datetime import datetime

from poli.benchmarks import PMOBenchmark  # type: ignore[import]

from hdbo_benchmark.utils.experiments.load_solvers import SOLVER_NAMES

current_date = datetime.now().strftime("%Y-%m-%d")
latent_dims = [128]
TEST_RUN = False

function_names = PMOBenchmark(string_representation="SELFIES").problem_names
if TEST_RUN:
    function_names = function_names[:1]


def condition_to_write(solver_name, function_name, n_dimensions) -> bool:
    return solver_name in [
        "random_mutation",
        "vanilla_bo_hvarfner",
        # "cma_es",
        "line_bo",
        "saas_bo",
        "alebo",
        # "cma_es",
        "baxus",
        "bounce",
        "pr",
        # "turbo",
    ]


for function_name in function_names:
    for latent_dim in latent_dims:
        if TEST_RUN:
            max_iter = 3
        else:
            max_iter = 100 if latent_dim <= 100 else 300

        for solver_name in SOLVER_NAMES:
            if not condition_to_write(
                solver_name=solver_name,
                function_name=function_name,
                n_dimensions=latent_dim,
            ):
                continue

            if solver_name == "baxus":
                conda_env_name = "baxus_"
            elif solver_name == "bounce":
                conda_env_name = "bounce"
            elif solver_name == "pr":
                conda_env_name = "hdbo__pr"
            else:
                conda_env_name = "hdbo_benchmark"

            print(
                f"sbatch src/hdbo_benchmark/experiments/benchmark_on_pmo/batch.sh {solver_name} {function_name} {latent_dim} {conda_env_name} {max_iter} {current_date}"
            )

# scp data/trained_models/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-714.pt hendrixgate03fl:Projects/high_dimensional_bo_benchmark/data/trained_models/training_vae_on_zinc_250k/latent_dim-128-batch_size-512-lr-0.0005-seed-714.pt
