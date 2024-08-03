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
START=$1
END=$2

BATCH_SIZE=4
NEIGHBOURS=3

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
exp_name="Vsion_subtitles_model_summary_subtitle_videoLLM"
echo $exp_name 
python eval_goldfish_tvqa_long.py --add_unknown --index_subtitles_together --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --ckpt $CKPT_PATH  --exp_name=$exp_name\
      --tvqa_json_subtitles $tvqa_json_subtitles --tvqa_clips_subtitles $tvqa_clips_subtitles --videos_frames $videos_frames --annotation_path $annotation_path


# vision only 
# exp_name="vision_only"
# echo $exp_name 
# python eval_goldfish_tvqa_long.py --add_unknown --vision_only --model_summary_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH  --exp_name=$exp_name

# # subtiltes only  (eliminate the vision)
# exp_name="subtitles_only"
# echo $exp_name 
# python eval_goldfish_tvqa_long.py --add_unknown --index_subtitles_together --subtitles_only --neighbours=$NEIGHBOURS  --start=$START --end=$END --batch_size $BATCH_SIZE  --name $NAME --ckpt $CKPT_PATH  --exp_name=$exp_name
