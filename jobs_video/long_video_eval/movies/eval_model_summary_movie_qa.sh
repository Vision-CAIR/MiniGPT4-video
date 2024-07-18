#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=M_RAG_general_summary_1_subtitles_together_%j
#SBATCH --output=M_RAG_general_summary_1_subtitles_together_%j.out
#SBATCH --error=M_RAG_general_summary_1_subtitles_together_%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1


## run the application:
cd ../../../
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
START=$1
END=$2
BATCH_SIZE=4

NEIGHBOURS=3
## Dataset paths
videos_path="path to the videos"
subtitle_path="path to the subtitles"
video_clips_saving_path="path to save the video clips"
annotation_file="path to the annotation file"
movienet_annotations_dir="path to the movienet annotations directory"
# if you want to use openai embedding, then you need to set the OPENAI_API_KEY
use_openai_embedding=True
export OPENAI_API_KEY="your_openai_key"


# videos_path="/ibex/project/c2106/kirolos/MovieNet/240_frames"
# subtitle_path="/ibex/project/c2106/kirolos/Movie_QA/subtitle"
# video_clips_saving_path="/ibex/project/c2106/kirolos/MovieNet/240_clips"
# annotation_path="datasets/evaluation_datasets/goldfish_eval_datasets/movie_qa/movie_qa_edited.json"
# movienet_annotations_dir="/ibex/ai/reference/videos/MoiveNet/MovieNet/raw/files/annotation"
# if you want to use openai embedding, then you need to set the OPENAI_API_KEY
use_openai_embedding=True
# export OPENAI_API_KEY="your_openai_key"


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


# # Vision + subtitles
exp_name="Vsion_subtitles_model_summary_subtitle"
echo $exp_name 
python eval_long_videos_movie_qa.py --add_unknown --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE --ckpt $CKPT_PATH  --exp_name=$exp_name\
      --videos_path $videos_path --subtitle_path $subtitle_path --video_clips_saving_path $video_clips_saving_path --annotation_path $annotation_path --movienet_annotations_dir $movienet_annotations_dir --use_openai_embedding $use_openai_embedding


# vision only 
# exp_name="vision_only"
# echo $exp_name 
# python eval_long_videos_movie_qa.py --add_unknown --vision_only --model_summary_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --videos_path $videos_path --subtitle_path $subtitle_path --video_clips_saving_path $video_clips_saving_path --annotation_path $annotation_path --movienet_annotations_dir $movienet_annotations_dir --use_openai_embedding $use_openai_embedding

# subtiltes only  (eliminate the vision)
# exp_name="subtitles_only"
# echo $exp_name 
# python eval_long_videos_movie_qa.py --add_unknown --index_subtitles_together --subtitles_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --videos_path $videos_path --subtitle_path $subtitle_path --video_clips_saving_path $video_clips_saving_path --annotation_path $annotation_path --movienet_annotations_dir $movienet_annotations_dir --use_openai_embedding $use_openai_embedding

