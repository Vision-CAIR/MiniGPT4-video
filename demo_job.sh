#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=video_demo_general_llama2_1day
#SBATCH --output=video_demo_general_llama2_1day.out
#SBATCH --error=video_demo_general_llama2_1day.err
#SBATCH --time=0-23:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
/ibex/ai/home/ataallka/miniforge-pypy3/envs/gradio_test/bin/python3.9 -u minigpt4_video_demo.py