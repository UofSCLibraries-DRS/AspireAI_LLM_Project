#!/bin/sh
#SBATCH --job-name=library_inference
#SBATCH -N 1
#SBATCH -n 16    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output job_%j.out
#SBATCH --error job_%j.err
#SBATCH -p AI_Center_L40S,dgx_aic,gpu-v100-32gb,gpu-v100-16gb-hiprio

export CUDA_VISIBLE_DEVICES=0

cd /work/jaaydin/ft_test

module load python3/anaconda/2021.07 gcc/12.2.0 cuda/12.3
source activate /home/jaaydin/.conda/envs/ft_test

python inference.py \
    --model_dir /work/jaaydn/100_clean \
    --input_csv /work/jaaydin/qa_data.csv \
    --output_csv /work/jaaydin/qa_answered.csv \
    --max_new_tokens 150 \
    --temperature 0.8