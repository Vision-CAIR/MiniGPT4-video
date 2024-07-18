#!/bin/bash
#SBATCH --partition=batch


#SBATCH --job-name=RAG_clips_info_1_vision_%j
#SBATCH --output=RAG_clips_info_1_vision_%j.out
#SBATCH --error=RAG_clips_info_1_vision_%j.err
#SBATCH --time=0-23:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1


## run the application:
cd ../../../
START=$1
END=$2

BATCH_SIZE=4
NEIGHBOURS=3
CKPT_PATH="checkpoints/video_llama_checkpoint_last.pth"
# tvqa_json_subtitles="path to the tvqa json subtitles file"
# tvqa_clips_subtitles="path to the tvqa clips subtitles"
# videos_frames="path to the video frames"
# annotation_path="path to the TVQA-Long annotation file"


tvqa_json_subtitles="datasets/evaluation_datasets/goldfish_eval_datasets/tvqa/tvqa_preprocessed_subtitles.json"
tvqa_clips_subtitles="/ibex/project/c2090/datasets/TVR_dataset/videos/tvqa_subtitles"
videos_frames="/ibex/project/c2090/datasets/TVR_dataset/videos/video_files/frames_hq/"
annotation_path="datasets/evaluation_datasets/goldfish_eval_datasets/tvqa/tvqa_val_edited.json"

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
python eval_long_videos_tvqa_cleaned.py --add_unknown --use_clips_for_info --use_choices_for_info --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
      --tvqa_json_subtitles $tvqa_json_subtitles --tvqa_clips_subtitles $tvqa_clips_subtitles --videos_frames $videos_frames --annotation_path $annotation_path


# exp_name="Vsion_subtitles_info_only"
# echo $exp_name 
# python eval_long_videos_tvqa_cleaned.py --add_unknown --info_only --use_clips_for_info --use_choices_for_info --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --tvqa_json_subtitles $tvqa_json_subtitles --tvqa_clips_subtitles $tvqa_clips_subtitles --videos_frames $videos_frames --annotation_path $annotation_path


# exp_name="info_sub_after_retrieval"
# echo $exp_name 
# python eval_long_videos_tvqa_cleaned.py --add_unknown --subtitles_only_after_retrieval --use_clips_for_info --use_choices_for_info --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --tvqa_json_subtitles $tvqa_json_subtitles --tvqa_clips_subtitles $tvqa_clips_subtitles --videos_frames $videos_frames --annotation_path $annotation_path





# vision only 
# exp_name="vision_only"
# echo $exp_name 
# python eval_long_videos_tvqa_cleaned.py --add_unknown --use_clips_for_info --use_choices_for_info --vision_only --model_summary_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
#       --tvqa_json_subtitles $tvqa_json_subtitles --tvqa_clips_subtitles $tvqa_clips_subtitles --videos_frames $videos_frames --annotation_path $annotation_path
