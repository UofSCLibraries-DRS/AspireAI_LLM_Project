#!/bin/sh
#SBATCH --job-name=library_ft
#SBATCH -N 1                   # 1 node
#SBATCH -n 16                  # 16 CPU tasks (threads)
#SBATCH --gres=gpu:4           # request 4 A100 GPUs
#SBATCH --account=rc_gener#SBATCH --output=logs/lora_ft_%j.out
#SBATCH --error=logs/lora_ft_%j.err
#SBATCH -p gpu-A100
#SBATCH --time=00:05:00

# Navigate to project directory
cd /work/jaaydin/AspireAI_LLM_Project

# Load conda
source /work/jaaydin/miniconda3/etc/profile.d/conda.sh
conda activate lib_train

# Load CUDA module
module load cuda/12.1

# Force conda's libstdc++ to be used (avoid conflicts)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Optional: set TMPDIR for large pip wheels or temp files
export TMPDIR=/work/jaaydin/tmp

export PYTHONPATH="$(pwd):$PYTHONPATH"

# Optional: use Accelerate for multi-GPU training
# Make sure your training script supports Accelerate
accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes 4 \
    fine_tuning/main.py \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/llama/llama_cpt_lora64.json \
    --env .env.rci