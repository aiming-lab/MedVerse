cd MedVerse/inference/inference_engine
conda activate medverse-eval

CUDA_VISIBLE_DEVICES=3,4,5,7 python -m vllm.entrypoints.openai.api_server \
    --model /playpen-raid/cjw/qwen2.5-14b/MedVerse-20260217_100649/checkpoint-291 \
    --served-model-name MedVerse \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 262144 \
    --port 8010


CUDA_VISIBLE_DEVICES=3,4,5,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --served-model-name qwen2.5-14b \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 4 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 262144 \
    --port 8010


python benchmark_engines.py