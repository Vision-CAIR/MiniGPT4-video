#!/bin/bash
#SBATCH --partition=batch
#SBATCH --mail-user=kirolos.ataallah@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=mistral_best%j
#SBATCH --output=mistral_best%j.out
#SBATCH --error=mistral_best%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="mistral_best" # Name of the experiment
DATASET="tvqa" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif,video_chatgpt_generic,video_chatgpt_temporal,video_chatgpt_consistency
BATCH_SIZE=4 # batch size for A100 by using subtiles is 2 and without subtitles is 4 
CKPT_PATH="checkpoints/video_mistral_checkpoint_best.pth" # path to the checkpoint
cfg_path="test_configs/mistral_test_config.yaml" # path to the config file
# # if the number of samples are large you can specify the start and end index to evaluate on several machines
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
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path  --start $start --end $end

# with subtitles
# python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --add_subtitles --start $start --end $end