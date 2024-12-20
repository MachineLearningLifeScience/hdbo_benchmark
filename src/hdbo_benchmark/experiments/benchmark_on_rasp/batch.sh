#!/bin/bash
#SBATCH --job-name=rasp-benchmark
#SBATCH --output=output_rasp.log
#SBATCH --error=error_rasp.log
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --array=1-5%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00

ENV_NAME=$1
SOLVER_NAME=$2
LATENT_DIM=$3
MAX_ITER=$4
TAG=$5

# Add your commands here
if [ "${SOLVER_NAME}" == "bounce" ] || [ "${SOLVER_NAME}" == "pr" ] || [ "${SOLVER_NAME}" == "genetic_algorithm" ]; then
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_rasp/run.py --solver-name=${SOLVER_NAME} --latent-dim=${LATENT_DIM} --max-iter=${MAX_ITER} --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG} --solve-in-discrete-space"
else
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_rasp/run.py --solver-name=${SOLVER_NAME} --latent-dim=${LATENT_DIM} --max-iter=${MAX_ITER} --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG}"
fi
echo $command
eval $command