#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=llama2%j
#SBATCH --output=llama2%j.out
#SBATCH --error=llama2%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="llama2" # Name of the experiment
BATCH_SIZE=8 
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth" # path to the checkpoint

DATASET="msvd" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif ,video_chatgpt_generic,video_chatgpt_temporal,video_chatgpt_consistency
# set the paths to the dataset files
videos_path="" # path to the videos file
subtitles_path="" # path to the subtitles file
ann_path="" # path to the annotations file

cfg_path="test_configs/llama2_test_config.yaml" # path to the config file 
# # if the number of samples are too large you can specify the start and end index to evaluate on several machines
# pass the start and end index as arguments
start=$1 # start index
end=$2 # end index
# if start and end are not provided, then use the whole dataset
if [ -z "$start" ]
then
      start=0
fi
if [ -z "$end" ]
then
      end=10000000
fi
echo "Start: $start"
echo "End: $end"


# with subtitles
python evaluation/eval_minigpt4_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --videos_path $videos_path --subtitles_path $subtitles_path --ann_path $ann_path --ckpt $CKPT_PATH  --cfg-path=$cfg_path --start $start --end $end --add_subtitles

# without subtitles
# python evaluation/eval_minigpt4_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --videos_path $videos_path --subtitles_path $subtitles_path --ann_path $ann_path --ckpt $CKPT_PATH  --cfg-path=$cfg_path --start $start --end $end 

