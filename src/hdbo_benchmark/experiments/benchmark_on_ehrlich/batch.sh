#!/bin/bash
#SBATCH --job-name=ehrlich-benchmark
#SBATCH --output=output_ehrlich.log
#SBATCH --error=error_ehrlich.log
#SBATCH --partition=gpu
#SBATCH --array=1-3%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00

ENV_NAME=$1
SOLVER_NAME=$2
SEQUENCE_LENGTH=$3
N_MOTIFS=$4
MOTIF_LENGTH=$5
MAX_ITER=$6
TAG=$7

# Add your commands here
if [ "${SOLVER_NAME}" == "bounce" ] || [ "${SOLVER_NAME}" == "pr" ] || [ "${SOLVER_NAME}" == "genetic_algorithm" ]; then
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_ehrlich/run.py --solver-name=${SOLVER_NAME} --sequence-length=${SEQUENCE_LENGTH} --motif-length=${MOTIF_LENGTH} --max-iter=${MAX_ITER} --solve-in-discrete-space --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG}"
else
    command="conda run -n ${ENV_NAME} python src/hdbo_benchmark/experiments/benchmark_on_ehrlich/run.py --solver-name=${SOLVER_NAME} --sequence-length=${SEQUENCE_LENGTH} --motif-length=${MOTIF_LENGTH} --max-iter=${MAX_ITER} --seed=${SLURM_ARRAY_TASK_ID} --tag=${TAG}"
fi
echo $command
eval $command