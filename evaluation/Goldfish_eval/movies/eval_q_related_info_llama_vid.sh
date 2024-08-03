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
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
BATCH_SIZE=4
START=$1
END=$2

NEIGHBOURS=3

# Dataset paths
videos_path="path to the videos"
subtitle_path="path to the subtitles"
video_clips_saving_path="path to save the video clips"
annotation_file="path to the annotation file"
movienet_annotations_dir="path to the movienet annotations directory"
# if you want to use openai embedding, then you need to set the OPENAI_API_KEY
use_openai_embedding=True
export OPENAI_API_KEY="your_openai_key"


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

# # Vision + subtitles
exp_name="Vsion_subtitles_model_summary_subtitle"
echo $exp_name 
python evaluation/eval_goldfish_llama_vid.py --use_clips_for_info  --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
      --videos_path $videos_path --subtitle_path $subtitle_path --video_clips_saving_path $video_clips_saving_path --annotation_path $annotation_path --movienet_annotations_dir $movienet_annotations_dir --use_openai_embedding $use_openai_embedding


# vision only 
# exp_name="vision_only"
# echo $exp_name 
# python evaluation/eval_goldfish_llama_vid.py --use_clips_for_info --vision_only --model_summary_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --videos_path $videos_path --subtitle_path $subtitle_path --video_clips_saving_path $video_clips_saving_path --annotation_path $annotation_path --movienet_annotations_dir $movienet_annotations_dir --use_openai_embedding $use_openai_embedding

# # subtiltes only  (eliminate the vision)
# it is only from summaries no need to run it with clips 
