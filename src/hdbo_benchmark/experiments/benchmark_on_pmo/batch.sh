#!/bin/bash
#SBATCH --job-name=pmo-benchmark
#SBATCH --output=output_pmo.log
#SBATCH --error=error_pmo.log
#SBATCH --partition=boomsma
#SBATCH --array=1-3%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G
#SBATCH --time=2-23:00:00

# Telling all models to not use GPU
export CUDA_VISIBLE_DEVICES=""

discrete_solvers=("bounce" "pr")

# Add your commands here
if [ "$1" == "bounce" ] || [ "$1" == "pr" ]; then
    command="conda run -n $4 python src/hdbo_benchmark/experiments/benchmark_on_pmo/run.py --function-name=$2 --solver-name=$1 --latent-dim=$3 --max-iter=$5 --solve-in-discrete-space --seed=${SLURM_ARRAY_TASK_ID} --tag=$6"
else
    command="conda run -n $4 python src/hdbo_benchmark/experiments/benchmark_on_pmo/run.py --function-name=$2 --solver-name=$1 --latent-dim=$3 --max-iter=$5 --seed=${SLURM_ARRAY_TASK_ID} --tag=$6"
fi
echo $command
eval $command