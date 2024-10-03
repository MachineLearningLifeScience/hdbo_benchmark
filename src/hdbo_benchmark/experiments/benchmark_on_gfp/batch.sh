#!/bin/bash
#SBATCH --job-name=gfp-benchmark
#SBATCH --output=output_gfp.log
#SBATCH --error=error_gfp.log
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --array=1-7%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

ENV_NAME=$1
SOLVER_NAME=$2
MAX_ITER=$3
TAG=$4

# Add your commands here
if [ "${SOLVER_NAME}" == "bounce" ] || [ "${SOLVER_NAME}" == "pr" ] || [ "${SOLVER_NAME}" == "genetic_algorithm" ]; then
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_gfp/run.py --solver-name=${SOLVER_NAME} --max-iter=${MAX_ITER} --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG} --solve-in-discrete-space"
else
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_gfp/run.py --solver-name=${SOLVER_NAME} --max-iter=${MAX_ITER} --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG}"
fi
echo $command
eval $command