# Training

Fine-tuning scripts for MedVerse medical reasoning models using topology-aware attention SFT.

## Directory Structure

```
train/
├── sft_medverse.py        # Training entry point
├── utils.py               # Topology-aware attention mask & data collator
├── eval.py                # Post-training evaluation
├── requirements.txt
├── configs/               # FSDP configs per model family
│   ├── fsdp_qwen.json
│   ├── fsdp_qwen_cpu.json
│   └── fsdp_llama.json
└── scripts/               # Launch scripts
    ├── launch_train_qwen.sh
    └── launch_train_llama.sh
```

## Requirements

- 2× GPU minimum (tested on NVIDIA RTX PRO 6000 Blackwell, 96 GB each)
- See `requirements.txt` for Python dependencies

## Quick Start

### 1. Prepare data

Generate or download the MedVerse14k dataset. See [`../data/README.md`](../data/README.md).

The training scripts expect the dataset at `../data/datasets/MedVerse14k/` (HuggingFace Dataset format, produced by `../data/preparation/prepare_train.py`).

### 2. Train

```bash
# Fine-tune Qwen2.5-7B-Instruct
bash scripts/launch_train_qwen.sh

# Fine-tune LLaMA-3.1-8B-Instruct
bash scripts/launch_train_llama.sh
```

Edit the variables at the top of each script to set your model path, output directory, and GPU selection.

### 3. Evaluate

```bash
python eval.py \
    --model_name /path/to/checkpoint \
    --eval_file ../eval_data/medqa_test.jsonl \
    --use_chat_template \
    --batch_size 4
```

## Configuration

Key training hyperparameters (paper settings):

| Parameter | Value |
|---|---|
| Learning rate | 1e-5 |
| Epochs | 3 |
| Global batch size | 128 (2 GPU × 1 × grad_accum 64) |
| Block size | 8192 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |

FSDP configs are in `configs/`:
- `fsdp_qwen_cpu.json` — Qwen with CPU offload (recommended for memory-constrained setups)
- `fsdp_qwen.json` — Qwen without CPU offload
- `fsdp_llama.json` — LLaMA-3.1
