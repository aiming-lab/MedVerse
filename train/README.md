# Training

Fine-tuning scripts for MedVerse medical reasoning models using topology-aware attention SFT.

## Directory Structure

```
train/
├── sft_medverse.py        # Training entry point
├── utils.py               # Topology-aware attention mask & data collator
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

**Recommended GPU configuration:**


| Setting       | Value                                                |
| ------------- | ---------------------------------------------------- |
| GPUs          | 4× NVIDIA GPU                                        |
| VRAM per GPU  | ≥ 48 GB                                              |
| FSDP strategy | `full_shard auto_wrap offload` (CPU offload enabled) |


For lower-memory setups, reduce `--gradient_accumulation_steps` and increase CPU offload. The scripts default to 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`) — adjust as needed.

- See `requirements.txt` for Python dependencies

## Quick Start

### 1. Prepare data

Generate or download the MedVerse14k dataset. See [../data/README.md](../data/README.md).

The training scripts expect the dataset in HuggingFace Dataset format produced by the preparation scripts:

- Qwen2.5: `../data/datasets/MedVerse14k/` — produced by `../data/preparation/prepare_train.py`
- LLaMA-3: `../data/datasets/MedVerse14k-LLaMA/` — produced by `../data/preparation/prepare_train_llama.py`

### 2. Train

```bash
# Fine-tune Qwen2.5-7B-Instruct
bash scripts/launch_train_qwen.sh

# Fine-tune LLaMA-3.1-8B-Instruct
bash scripts/launch_train_llama.sh
```

Edit the variables at the top of each script to set your model path, output directory, and GPU selection.

## Configuration

Key training hyperparameters (paper settings):


| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Learning rate     | 1e-5                            |
| Epochs            | 3                               |
| Global batch size | 128 (4 GPU × 1 × grad_accum 32) |
| Block size        | 8192                            |
| LR scheduler      | Cosine                          |
| Warmup ratio      | 0.03                            |


FSDP configs are in `configs/`:

- `fsdp_qwen_cpu.json` — Qwen with CPU offload (recommended for memory-constrained setups)
- `fsdp_qwen.json` — Qwen without CPU offload
- `fsdp_llama.json` — LLaMA-3.1

