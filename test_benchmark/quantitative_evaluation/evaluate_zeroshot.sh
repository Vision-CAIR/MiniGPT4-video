#!/bin/bash
#SBATCH --partition=batch


#SBATCH --job-name=mistral_all%j
#SBATCH --output=mistral_all%j.out
#SBATCH --error=mistral_all%j.err
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
## run the application:

PRED="./../../results/video_instruct_checkpoint_3_Video_validation_Dataset_no_subtitles.json"
OUTPUT_DIR="./../output/Video_validation_Dataset/video_instruct_checkpoint_3_Video_validation_Dataset_no_subtitles"
rm -rf $OUTPUT_DIR
API_KEY="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"
NUM_TASKS=128


python evaluate_activitynet_qa.py \
    --pred_path ${PRED} \
    --output_dir "${OUTPUT_DIR}/fewshot_accuracy" \
    --output_json "${OUTPUT_DIR}/fewshot_accuracy_results.json"\
    --api_key $API_KEY \
    --num_tasks $NUM_TASKS

echo pred_path: $PRED