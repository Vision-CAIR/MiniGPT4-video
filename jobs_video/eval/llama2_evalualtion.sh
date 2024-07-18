#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=llama2_best%j
#SBATCH --output=llama2_best%j.out
#SBATCH --error=llama2_best%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="llama2_best" # Name of the experiment
DATASET="tvqa" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif ,video_chatgpt_generic,video_chatgpt_temporal,video_chatgpt_consistency
BATCH_SIZE=8 
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth" # path to the checkpoint
cfg_path="test_configs/llama2_test_config.yaml" # path to the config file
# # if the number of samples are too large you can specify the start and end index to evaluate on several machines
# pass the start and end index as arguments
start=$1 # start index
end=$2 # end index
# if start and end are not provided, then use the whole dataset
if [ -z "$START" ]
then
      START=0
fi
if [ -z "$END" ]
then
      END=10000000
fi
echo "Start: $START"
echo "End: $END"

cd ../../
# without subtitles
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --start $start --end $end

# with subtitles
# python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --add_subtitles --start $start --end $end