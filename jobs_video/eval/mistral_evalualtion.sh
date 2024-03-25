#!/bin/bash
#SBATCH --partition=batch
#SBATCH --mail-user=kirolos.ataallah@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=mistral_videoInstruction_best_activitynet_sub%j
#SBATCH --output=mistral_videoInstruction_best_activitynet_sub%j.out
#SBATCH --error=mistral_videoInstruction_best_activitynet_sub%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
## run the application:
NAME="mistral_videoInstruction_best" # Name of the experiment
DATASET="activitynet" # available datasets: tvqa, msrvtt, msvd, activitynet,tgif 
BATCH_SIZE=2 # batch size A100 by using subtiles is 2 and without subtitles is 4 
CKPT_PATH="checkpoints/video_mistral_checkpoint_best.pth" # path to the checkpoint
cfg_path="test_configs/mistral_test_config.yaml" # path to the config file
# # if the number of samples are large you can specify the start and end index to evaluate on several machines
start=$1 # start index
end=$2 # end index
cd ../../
python eval_video.py --dataset $DATASET --batch_size $BATCH_SIZE --name $NAME --ckpt $CKPT_PATH  --cfg-path=$cfg_path --start $start --end $end --add_subtitles