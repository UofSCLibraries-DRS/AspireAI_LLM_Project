#!/bin/sh
#SBATCH --job-name=library_ft
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output logs/lora_ft_%j.out
#SBATCH --error logs/lora_ft_%j.err
#SBATCH -p gpu-A100

export CUDA_VISIBLE_DEVICES=0
cd /work/jaaydin/AspireAI_LLM_Project

source /work/jaaydin/miniconda3/etc/profile.d/conda.sh
conda activate lib_train

module load cuda/12.1

# Force conda's libstdc++ to be used
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -u -m fine_tuning.main \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/llama/M11.E20.json \
    --env .env.rci

python -u -m fine_tuning.main \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/llama/M11.E10.json \
    --env .env.rci