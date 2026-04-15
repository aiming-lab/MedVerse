# MedVerse Inference Engine

This repository contains the official implementation of MedVerse Inference Engine, which is built upon the [SGLang](https://github.com/sgl-project/sglang) codebase to support inference for MedVerse models. For more details, please refer to our research paper:

**MedVerse: Efficient and Reliable Medical Reasoning via DAG-Structured Parallel Execution**

---

## 🤖 Key Features

- **Two-phase DAG execution** — Phase I generates a structured `<Plan>` with an `<Outline>` dependency graph; Phase II dispatches all reasoning steps in parallel on the GPU, then joins outputs into a final conclusion.
- **`MedVerseTokenizerManager`** — injects a `</Plan>` stop token so the model halts after plan generation and before parallel step execution begins.
- **`MedVerseScheduler`** — detects plan completion, forks child requests for every step in the DAG, and merges outputs into a single conclusion request.
- **`outline_parser` + `petri_net`** — parse `<Outline>` tags into a dependency DAG and track step completion via Petri nets to coordinate parallel execution.
- **Shared KV-cache prefix** — all parallel step branches reuse the Phase I radix-cache prefix, minimizing redundant computation across steps.
- **OpenAI-compatible API** — drop-in `/v1/chat/completions` endpoint; MedVerse routing is triggered automatically when the prompt contains the `<Think>` token.

---

## Quick Start

```bash
conda activate medverse-engine
```

**Starting the Server:**

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

## Try an Example

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
