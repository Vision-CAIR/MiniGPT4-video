#!/bin/bash

# Define common arguments for all scripts
# Without subtitles
PRED_GENERIC=""
PRED_TEMPORAL="test_benchmark/Other models/video_llava_temporal.json"
PRED_CONSISTENCY="test_benchmark/Other models/video_llava_consistency.json"
OUTPUT_DIR="/ibex/ai/home/ataallka/minigpt_video_results/quantitative_evaluation/ckpt_52_without_subtitles"
rm -rf $OUTPUT_DIR
# # With subtitles
# PRED_GENERIC="results/ckpt_52_video_chatgpt_generic_subtitles.json"
# PRED_TEMPORAL="results/ckpt_52_video_chatgpt_temporal_subtitles.json"
# PRED_CONSISTENCY="results/ckpt_52_video_chatgpt_consistency_subtitles.json"
#  OUTPUT_DIR="/ibex/ai/home/ataallka/minigpt_video_results/quantitative_evaluation/ckpt_52_with_subtitles"
# rm -rf $OUTPUT_DIR

API_KEY="open_ai_key"
NUM_TASKS=64

# Run the "correctness" evaluation script
python evaluate_benchmark_1_correctness.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "detailed orientation" evaluation script
python evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "contextual understanding" evaluation script
python evaluate_benchmark_3_context.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "temporal understanding" evaluation script
python evaluate_benchmark_4_temporal.py \
  --pred_path "${PRED_TEMPORAL}" \
  --output_dir "${OUTPUT_DIR}/temporal_eval" \
  --output_json "${OUTPUT_DIR}/temporal_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "consistency" evaluation script
python evaluate_benchmark_5_consistency.py \
  --pred_path "${PRED_CONSISTENCY}" \
  --output_dir "${OUTPUT_DIR}/consistency_eval" \
  --output_json "${OUTPUT_DIR}/consistency_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS


echo "All evaluations completed!"
