#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=mistral_videoInstruction_tvqa_no_subtitles%j
#SBATCH --output=mistral_videoInstruction_tvqa_no_subtitles%j.out
#SBATCH --error=mistral_videoInstruction_tvqa_no_subtitles%j.err
#SBATCH --time=0-59:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="mistral_videoInstruction_no_subtitles" # Name of the experiment
DATASET="tvqa" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif 
BATCH_SIZE=4  # batch size
CKPT_PATH="checkpoints/video_mistral_checkpoint_last.pth" # path to the checkpoint
cfg_path="test_configs/224_v2_mistral_video.yaml" # path to the config file
cd ../../
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path