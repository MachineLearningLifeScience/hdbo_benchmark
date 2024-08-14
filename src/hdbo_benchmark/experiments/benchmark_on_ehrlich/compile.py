from datetime import datetime

from hdbo_benchmark.utils.experiments.load_solvers import SOLVER_NAMES

current_date = datetime.now().strftime("%Y-%m-%d")

TEST_RUN = True

if TEST_RUN:
    sequence_lengths = [2]
    mapping_n_motifs = {
        2: [1],
    }
    mapping_motif_lengths = {
        2: [2],
    }
else:
    sequence_lengths = [8, 64, 256]
    mapping_n_motifs = {
        8: [2],
        64: [4],
        256: [16],
    }
    mapping_motif_lengths = {
        8: [4],
        64: [8],
        256: [10],
    }

DISCETE_SOLVERS = ["pr", "bounce", "genetic_algorithm"]
max_iter = 5 if TEST_RUN else 500

tag = f"EhrlichExperiments-{current_date}"
if TEST_RUN:
    tag += "-debug"


def condition_to_write(solver_name, sequence_length) -> bool:
    return solver_name in [
        "random_mutation",
        "genetic_algorithm",
        "cma_es",
        "line_bo",
        "vanilla_bo_hvarfner",
        # "saas_bo",
        # "alebo",
        # "baxus",
        # "bounce",
        # "pr",
        # "turbo",
    ]


for solver_name in SOLVER_NAMES:
    for sequence_length in sequence_lengths:
        n_motifs = mapping_n_motifs[sequence_length]
        motif_lengths = mapping_motif_lengths[sequence_length]
        for n_motif in n_motifs:
            for motif_length in motif_lengths:
                if not condition_to_write(
                    solver_name=solver_name,
                    sequence_length=sequence_length,
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

                command_ = f"sbatch src/hdbo_benchmark/experiments/benchmark_on_ehrlich/batch.sh {conda_env_name} {solver_name} {sequence_length} {n_motif} {motif_length} {max_iter} {tag}"

                print(command_)
