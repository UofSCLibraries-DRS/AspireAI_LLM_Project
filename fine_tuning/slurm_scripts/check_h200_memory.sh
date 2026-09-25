#!/usr/bin/env bash
#SBATCH --job-name=h200_memory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-H200
#SBATCH --time=00:05:00
#SBATCH --output=h200_memory_%j.out
#SBATCH --error=h200_memory_%j.err

# Submit from the directory where you want the h200_memory_<job-id>.out log:
#   sbatch fine_tuning/slurm_scripts/check_h200_memory.sh

set -euo pipefail

echo "SLURM job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-not set}"
echo

echo "GPU memory (MiB):"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version \
    --format=csv,noheader,nounits

echo
echo "Full GPU status:"
nvidia-smi

echo
echo "Allocated SLURM resources:"
scontrol show job "${SLURM_JOB_ID}" | tr ' ' '\n' | \
    grep -E '^(JobId|NodeList|Partition|ReqTRES|AllocTRES)='

echo
echo "Host RAM:"
free -h
