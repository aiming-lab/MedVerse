# MedVerse Inference Engine

**SGLang-based inference server for MedVerse medical reasoning models**



This directory contains the MedVerse inference engine — a modified [SGLang](https://github.com/sgl-project/sglang) server that implements two-phase DAG-structured parallel execution for medical reasoning.

---

## Architecture

The engine adds three components on top of stock SGLang:


| Component                      | File                                         | Role                                                                                                  |
| ------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `MedVerseTokenizerManager`     | `srt/managers/medverse_tokenizer_manager.py` | Injects `</Plan>` stop token so Phase I halts after plan generation                                   |
| `MedVerseScheduler`            | `srt/managers/medverse_scheduler.py`         | Detects plan end, forks child requests for every step, joins outputs into a single conclusion request |
| `outline_parser` + `petri_net` | `srt/medverse/`                              | Parse `<Outline>` tags into a dependency DAG; track step completion via Petri nets                    |


### Execution flow

```
Client → /generate (MedVerse prompt)
    │
    ▼  Phase I
MedVerseTokenizerManager injects </Plan> stop
SGLang generates: preamble → <Plan><Outline>...</Outline></Plan>
    │
    ▼  fork (MedVerseScheduler._scan_and_fork_plans)
outline_parser extracts StepDef list
PetriNet built; all steps pre-fired (speculative parallel)
Child requests dispatched simultaneously for every step
    │
    │  Phase II — all steps run in parallel on GPU
    ├── Step 1 child → generates until </Step>
    ├── Step 2 child → generates until </Step>
    └── Step N child → generates until </Step>
    │
    ▼  join (MedVerseScheduler.merge_zombie_batch_to_run)
Radix-cache prefix from Phase I shared across all branches
Step outputs concatenated → conclusion_header appended
Single join request dispatched for Phase III
    │
    ▼
Client ← complete response (Plan + Steps + Conclusion)
```

---

## Installation

```bash
git clone https://github.com/aiming-lab/MedVerse.git
cd MedVerse/inference/MedVerse-Engine

conda create -n medverse-engine python=3.11 -y
conda activate medverse-engine

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

---

## Known Limitations

- **KV cache eviction** is not yet supported. For stability, keep concurrent requests ≤ 50.
- **Very long plans** (> 8 steps) may exceed KV memory; consider setting `--max-total-tokens` conservatively.

---

## License

MIT. See [LICENSE](LICENSE).