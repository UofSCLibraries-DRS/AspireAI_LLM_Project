#!/bin/sh
#SBATCH --job-name=library_ft
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --output logs/lora_ft_%j.out
#SBATCH --error logs/lora_ft_%j.err
#SBATCH -p AI_Center_L40S

export CUDA_VISIBLE_DEVICES=0
cd /work/jaaydin/AspireAI_LLM_Project

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lib_train

module load cuda/12.1

# Force conda's libstdc++ to be used
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -u -m fine_tuning.main \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/baseline_inference.json \
    --env .env.rci