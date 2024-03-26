#!/bin/bash
#SBATCH --partition=batch
#SBATCH --mail-user=kirolos.ataallah@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=llama2_stage_3_best%j
#SBATCH --output=llama2_stage_3_best%j.out
#SBATCH --error=llama2_stage_3_best%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="llama2_stage_3_best" # Name of the experiment
DATASET="msvd" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif ,video_chatgpt_generic,video_chatgpt_temporal,video_chatgpt_consistency
BATCH_SIZE=8 
CKPT_PATH="checkpoints/video_llama_checkpoint_best.pth" # path to the checkpoint
cfg_path="test_configs/llama2_test_config.yaml" # path to the config file
cd ../../

# without subtitles
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path

# with subtitles
# python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --add_subtitles