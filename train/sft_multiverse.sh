# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
#!/usr/bin/env bash
set -euo pipefail

export DATA_ROOT=/playpen-raid/cjw

export TMPDIR="${DATA_ROOT}/tmp"
export TMP="${DATA_ROOT}/tmp"
export TEMP="${DATA_ROOT}/tmp"
export PYARROW_TEMP_DIR="${DATA_ROOT}/tmp"

export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"

mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "${DATA_ROOT}/ckpts"
chmod 1777 "$TMPDIR"

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,4,5,7

export WANDB_API_KEY=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
export WANDB_NAME="MedVerse14k_qwen2.5-14b_sft_3epochs"
export WANDB_DIR="${DATA_ROOT}/wandb_dir"

uid="$(date +%Y%m%d_%H%M%S)"
base_model=Qwen/Qwen2.5-14B-Instruct
resume_ckpt=/playpen-raid/cjw/qwen2.5-14b/MedVerse-20260217_100649/checkpoint-97
resume_dir=/playpen-raid/cjw/qwen2.5-14b/MedVerse-20260217_100649
lr=2e-5
min_lr=0
epochs=3
weight_decay=0.05 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=32 # requires more GPU memory
max_steps=-1
gpu_count=4
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    ./sft_multiverse.py \
    --block_size=8192 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="MedVerse14k" \
    --model_name=${base_model} \
    --warmup_ratio=0.01 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="./fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="epoch" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${resume_dir}" \
    --resume_from_checkpoint "${resume_ckpt}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=False \
    --gradient_checkpointing=True \
    --use-liger=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'

# python /home/ubuntu/Med-Reason/Multiverse/train/eval.py \
#   --model_path ${base_model} \
#   --dataset_dir "Med_Reason" \
#   --split test \
#   --text_key text \
#   --block_size 32768 \
#   --per_device_eval_batch_size 1 \
#   --bf16 \
#   --save_json