#!/bin/sh
#SBATCH --job-name=library_ft
#SBATCH -N 1
#SBATCH -n 16    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output logs/lora_ft_%j.out
#SBATCH --error logs/lora_ft_%j.err
#SBATCH -p AI_Center_L40S,gpu-v100-32gb

export CUDA_VISIBLE_DEVICES=0

cd /work/jaaydin/AspireAI_LLM_Project

module load cuda/12.3

source ~/miniforge3/etc/profile.d/conda.sh\

conda activate lib_ft

conda python -u -m fine_tuning.main \
    --pipeline-path ./fine_tuning/config/gemma_pipeline.json \
    --env .env.rci