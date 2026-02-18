eval_file=/home/ubuntu/Med-Reason/Multiverse/eval_data/medbullets_op4.jsonl
task_floder=Qwen-7B-results
model_path=/data/qwen2.5-7b/Multiverse-20250918_235502

python ./src/evaluation/eval.py \
  --model_name "$model_path" \
  --eval_file "$eval_file" \
  --batch_size 64 \
  --max_new_tokens 2000 \
  --task_floder "$task_floder" \
  --strict_prompt \
  --dtype bfloat16
