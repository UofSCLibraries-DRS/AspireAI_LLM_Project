#!/bin/sh
#SBATCH --job-name=library_ft
#SBATCH -N 1
#SBATCH -n 16    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output logs/lora_ft_%j.out
#SBATCH --error logs/lora_ft_%j.err
#SBATCH -p AI_Center_L40S

export CUDA_VISIBLE_DEVICES=0

cd /work/jaaydin/AspireAI_LLM_Project

module load cuda/12.1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source ~/miniforge3/etc/profile.d/conda.sh

conda activate lib_train


python -u -m fine_tuning.main \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/llama/llama_pipeline.json \
    --env .env.rci

# conda run -p /home/jaaydin/.conda/envs/test_env python -m fine_tuning.main --pipeline-path ./fine_tuning/config/llama_pipeline.json --env .env.rci
# python -m fine_tuning.main --pipeline-path ./fine_tuning/config/llama_pipeline.json --env .env.rci


