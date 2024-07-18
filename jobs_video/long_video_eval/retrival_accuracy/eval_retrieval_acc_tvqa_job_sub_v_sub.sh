#!/bin/bash
#SBATCH --partition=batch


#SBATCH --job-name=Retrieval_acc_3_%j
#SBATCH --output=Retrieval_acc_3_%j.out
#SBATCH --error=Retrieval_acc_3_%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1


## run the application:
cd ../../../
NAME="ckpt_92"
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
START=$1
END=$2
BATCH_SIZE=8

# if start and end are not provided, then use the whole dataset
if [ -z "$START" ]
then
      START=0
fi
if [ -z "$END" ]
then
      END=100000
fi
echo "Start: $START"
echo "End: $END"
echo "Batch size: $BATCH_SIZE"

NEIGHBOURS=1
# exp_name="vision"

# /ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name

# /ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --vision_only --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name

# /ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --subtitles_only --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name



exp_name="subtitles"
/ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name

# /ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --vision_only --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name

# /ibex/ai/home/ataallka/miniforge-pypy3/envs/minigptlv/bin/python3.9 -u eval_retrieval_acc_tvqa.py  --subtitles_only --start=$START --end=$END --neighbours=$NEIGHBOURS --batch_size=$BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH --exp_name=$exp_name
