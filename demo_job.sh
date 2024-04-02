#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=video_demo_llama2
#SBATCH --output=video_demo_llama2.out
#SBATCH --error=video_demo_llama2.err
#SBATCH --time=0-10:30:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1

#  Choose the model to test 
# Mistral 
# ckpt="checkpoints/video_mistral_checkpoint_last.pth"
# config="test_configs/mistral_test_config.yaml"

# Llama2
ckpt="checkpoints/video_llama_checkpoint_last.pth"
config="test_configs/llama2_test_config.yaml"


python minigpt4_video_demo.py --cfg-path $config --ckpt $ckpt
