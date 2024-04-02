#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=zeroshot_eval%j
#SBATCH --output=zeroshot_eval%j.out
#SBATCH --error=zeroshot_eval%j.err
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1

## run the application:

# PRED="pred_path"
# OUTPUT_DIR="output_dir"
# API_KEY="api_key"
# NUM_TASKS=128


python evaluate_activitynet_qa.py \
    --pred_path ${PRED} \
    --output_dir "${OUTPUT_DIR}/fewshot_accuracy" \
    --output_json "${OUTPUT_DIR}/fewshot_accuracy_results.json"\
    --api_key $API_KEY \
    --num_tasks $NUM_TASKS

echo pred_path: $PRED