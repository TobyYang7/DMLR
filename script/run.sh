#!/bin/bash
set -euo pipefail

export HF_HOME=.cache
export CUDA_VISIBLE_DEVICES=0,1

NUM_WORKERS=2
DATA_POINT=4
DATASET=scienceqa
DATASET_FILE=data/${DATASET}.json
OUTPUT_DIR=./output_vis/v8/${DATASET}/${DATA_POINT}/qwen_initpatch_passk

MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct

uv run python main.py \
    --dataset $DATASET_FILE \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --seed 42 \
    --max_new_tokens 2048 \
    --max_num_steps 15 \
    --num_thought_tokens 2 \
    --sigma 25.0 \
    --sigma_decay 0.95 \
    --lr 0.01 \
    --verbose 0 \
    --min_pixels 128 \
    --max_pixels 256 \
    --start_data_idx 0 \
    --end_data_idx $DATA_POINT \
    --use_llm_verify \
    --num_workers $NUM_WORKERS \
    --worker_device_round_robin \
    --num_selected_patches 16 \
    --initial_patch_count 1\
    --patch_increment 1 \
    --visual_insert_stride 1 \
    --visual_injection_start_step 0 \
    --visual_injection_interval 1 \
