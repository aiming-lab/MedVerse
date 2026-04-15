#!/usr/bin/env bash
# Launch MedVerse fine-tuning on Qwen2.5-7B-Instruct
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/path/to/huggingface_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/MedVerse-Qwen2.5-7B}"

echo "[$(date)] Starting Qwen2.5-7B training → $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate medverse

torchrun \
    --nproc-per-node 4 \
    --master_port 12305 \
    sft_medverse.py \
    --model_name "$MODEL_PATH" \
    --train_file_path ../data/datasets/MedVerse14k \
    --block_size 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --bf16 True \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_config configs/fsdp_qwen_cpu.json \
    --eval_strategy no \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model True \
    --gradient_checkpointing True \
    --output_dir "$OUTPUT_DIR" \
    --push_to_hub false \
    --report_to none \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[$(date)] Qwen2.5-7B training finished."
