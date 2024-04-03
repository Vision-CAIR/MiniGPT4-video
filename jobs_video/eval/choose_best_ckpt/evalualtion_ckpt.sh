#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=val%j
#SBATCH --output=val%j.out
#SBATCH --error=val%j.err
#SBATCH --time=0-10:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME=$2 # Name of the experiment
DATASET="dataset_name" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif,video_chatgpt_generic,video_chatgpt_temporal,video_chatgpt_consistency 
BATCH_SIZE=2  # batch size
CKPT_PATH=$1 # path to the checkpoint
cfg_path="test_configs/mistral_test_config.yaml" # path to the config file
cd ../../../
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --add_subtitles