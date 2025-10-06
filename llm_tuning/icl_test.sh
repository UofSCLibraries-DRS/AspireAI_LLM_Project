#!/bin/sh
#SBATCH --job-name=library_inference
#SBATCH -N 1
#SBATCH -n 16    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output logs/inference_base_%j.out
#SBATCH --error logs/inference_base_%j.err
#SBATCH -p gpu-v100-16gb,AI_Center_L40S,gpu-v100-32gb

export CUDA_VISIBLE_DEVICES=0

cd /work/jaaydin/AspireAI_LLM_Project/llm_tuning

module load python3/anaconda/2021.07 gcc/12.2.0 cuda/12.3
source activate /home/jaaydin/.conda/envs/ft_test

python inf_eos.py \
    --model_dir /work/jaaydin/outputs/100_mccray_clean_base_Llama_8b/model \
    --input_csv /work/jaaydin/data/factoid_qa.csv \
    --output_csv /work/jaaydin/outputs/100_mccray_clean_base_Llama_8b/results/icl_p4_eos.csv \
    --temperature 1.0 \
    --num_samples 5 \
    --system_prompt config/prompts/icl_test/icl_p4.yaml