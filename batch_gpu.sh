#!/bin/bash
#SBATCH --job-name=hdbo_benchmark_gpu
#SBATCH --output=slurm_logs/output.%j.log
#SBATCH --error=slurm_logs/output.%j.err
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --array=1-5%5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00

FUNCTION_NAME=$1
SOLVER_NAME=$2
N_DIMENSIONS=$3
MAX_ITER=$4
TAG=$5
ENV_NAME=$6

# Add your commands here
command="conda run -n ${ENV_NAME} python run.py \
        --function-name=${FUNCTION_NAME} \
        --solver-name=${SOLVER_NAME} \
        --n-dimensions=${N_DIMENSIONS} \
        --max-iter=${MAX_ITER} \
        --strict-on-hash \
        --force-run \
        --wandb-mode=online \
        --seed=${SLURM_ARRAY_TASK_ID} \
        --tag=${TAG}"

echo $command
eval $command

# python run.py --function-name=albuterol_similarity --solver-name=directed_evolution --n-dimensions=128 --max-iter=500 --strict-on-hash --wandb-mode=online --seed=9999 --tag=debug-on-cluster