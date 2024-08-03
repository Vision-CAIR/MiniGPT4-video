#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=MC_RAG_general_summary_all_%j
#SBATCH --output=MC_RAG_general_summary_all_%j.out
#SBATCH --error=MC_RAG_general_summary_all_%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1


## run the application:
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
START=$1
END=$2
BATCH_SIZE=4
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

NEIGHBOURS=-1 # use the whole neighbourhood for the global mode

dataset_path="path to the movies folder"
annotation_json_folder="path to the jsons folder"
# if you want to use openai embedding, then you need to set the OPENAI_API_KEY
use_openai_embedding=True
export OPENAI_API_KEY="your_openai_key"



exp_name="model_summary_and_subtitle"
fps=2

# use general summary
python evaluation/eval_goldfish_movie_chat.py --fps=$fps --neighbours_global=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
      --dataset_videos_path $dataset_path --annotation_json_folder $annotation_json_folder --use_openai_embedding $use_openai_embedding
