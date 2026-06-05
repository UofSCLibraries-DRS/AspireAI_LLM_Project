#!/bin/bash
#SBATCH --job-name=M12_full_greedy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output logs/M12_full_greedy_%j.out
#SBATCH --error logs/M12_full_greedy_%j.err
#SBATCH -p gpu-A100
#SBATCH --mail-user=jaaydin@email.sc.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -e

cd /work/jaaydin/AspireAI_LLM_Project

source /work/jaaydin/miniconda3/etc/profile.d/conda.sh
conda activate lib_train

module load cuda/12.1

# Force conda's libstdc++ to be used
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -u -m fine_tuning.main \
    --pipeline-path /work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/pipelines/llama/M12_full.json \
    --env .env.rci
