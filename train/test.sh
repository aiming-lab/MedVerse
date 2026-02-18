export DATA_ROOT=/data
export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

find "$HF_HOME" -type d -name "models--Qwen--Qwen2.5-7B-Instruct" -print -exec rm -rf {} +

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False --force-download

head -n 40 /data/models/Qwen2.5-7B-Instruct/config.json