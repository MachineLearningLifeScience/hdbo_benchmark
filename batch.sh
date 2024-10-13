#!/bin/bash
#SBATCH --job-name=hdbo_benchmark
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

# Assuming that conda is already activated...
# Creating the conda env if it doesn't exist
if [ ! -d "$CONDA_PREFIX/envs/${ENV_NAME}" ]; then
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Making sure the dependencies are there
if [ ${ENV_NAME} == "hdbo_ax" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[ax]"
fi
if [ ${ENV_NAME} == "hdbo_bounce" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[bounce]"
fi
if [ ${ENV_NAME} == "hdbo_baxus" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[baxus]"
fi
if [ ${ENV_NAME} == "hdbo_lambo2" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[lambo2]"
fi
if [ ${ENV_NAME} == "hdbo_pr" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[pr]"
fi
if [ ${ENV_NAME} == "hdbo_alebo" ]; then
    conda run -n ${ENV_NAME} pip install -e ".[alebo]"
fi
if [ ${ENV_NAME} == "hdbo_benchmark" ]; then
    conda run -n ${ENV_NAME} pip install -e "."
fi


# Add your commands here
command="conda run -n ${ENV_NAME} python run.py \
        --function-name=${FUNCTION_NAME} \
        --solver-name=${SOLVER_NAME} \
        --n-dimensions=${N_DIMENSIONS} \
        --max-iter=${MAX_ITER} \
        --strict-on-hash \
        --wandb-mode=online \
        --seed=${SLURM_ARRAY_TASK_ID} \
        --tag=${TAG}"

echo $command
eval $command