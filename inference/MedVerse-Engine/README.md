# MedVerse Inference Engine

This repository contains the official implementation of MedVerse Inference Engine, which is built upon the [SGLang](https://github.com/sgl-project/sglang) codebase to support inference for MedVerse models. For more details, please refer to our research paper:

**MedVerse: Efficient and Reliable Medical Reasoning via DAG-Structured Parallel Execution**

---

## Installation

```bash
bash install.sh
```

The install script runs `pip install -e "python[all]"` inside the bundled SGLang fork.

---

## Starting the Server

```bash
python -m sglang.srt.entrypoints.medverse_server \
    --model-path /path/to/MedVerse-7B \
    --tp-size 1 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

**Multi-GPU example (tensor parallelism across 2 GPUs):**

```bash
python -m sglang.srt.entrypoints.medverse_server \
    --model-path /path/to/MedVerse-7B \
    --tp-size 2 \
    --port 30000 \
    --trust-remote-code
```

Wait for `Server is ready` in the logs before sending requests.

---

## Quick Start Example

```bash
cd example
python example.py \
    --server_url http://localhost:30000 \
    --prompts_dir ./prompt
```

The prompts directory contains sample medical questions. You can add your own `.txt` files — one question per file.

---

## API

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint. MedVerse routing is triggered automatically when the prompt contains the `<Think>` token.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="MedVerse-7B",
    messages=[
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user",   "content": "Question:\nA 45-year-old woman presents with..."},
    ],
    max_tokens=3072,
    temperature=0.6,
)
print(response.choices[0].message.content)
```

