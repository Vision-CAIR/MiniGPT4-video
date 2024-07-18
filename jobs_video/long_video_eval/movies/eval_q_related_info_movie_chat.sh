#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=job_name%j
#SBATCH --output=job_name%j.out
#SBATCH --error=job_name%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1


## run the application:
cd ../../../
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
BATCH_SIZE=4
START=$1
END=$2
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


# dataset_path='/ibex/project/c2090/datasets/MovieChat/extracted_movies'
# annotation_json_folder="/ibex/project/c2090/datasets/MovieChat/jsons"
# # if you want to use openai embedding, then you need to set the OPENAI_API_KEY
# use_openai_embedding=True
# export OPENAI_API_KEY="your_openai_key"


exp_name="model_summary_and_subtitle"
fps=2

# use this for both info and general summary --v_sum_and_info

python eval_long_videos_movie_chat.py  --fps=$fps --neighbours_global=$NEIGHBOURS --batch_size=$BATCH_SIZE --start=$START --end=$END  --use_clips_for_info --ckpt $CKPT_PATH   --exp_name=$exp_name --dataset_path $dataset_path --annotation_json_folder $annotation_json_folder --use_openai_embedding $use_openai_embedding