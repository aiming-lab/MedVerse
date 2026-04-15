# MedVerse Data Generation

This directory contains the pipeline for generating the **MedVerse14k** training dataset — 13,904 medical QA samples annotated with knowledge-grounded DAG reasoning paths.

The full dataset is available on [🤗 HuggingFace](https://huggingface.co/datasets/Jianwen/MedVerse14k). Run this pipeline only if you want to generate data from scratch or extend it with new sources.

---

## Pipeline Overview

```
Raw medical QA (JSONL)
        │
        ▼
[1] Generate_Reasoning_Path.py    Filter & refine reasoning chains via LLM
        │
        ▼
[2] Generate_Initial_Plan.py      Convert chains into DAG <Plan> structure
        │
        ▼
[3] Generate_Transient_Data.py    Generate step-by-step reasoning & conclusion
        │
        ▼ ◄─────────────────────────────────────┐
[4a] check_plan_execution.py      Validate XML structure (<Plan>/<Execution>)   │
        │                                                                        │
        ▼                                                                        │
[4b] check_conclusion.py          Validate conclusion consistency via LLM        │
        │                                                                        │
        └──── regenerate failed samples ─────────────────────────────────────────┘
```

---

## Input Format

The pipeline starts from a **JSONL file** (one JSON object per line) where each entry is a medical multiple-choice question with pre-existing reasoning chains:

```jsonl
{"id": "0", "question": "A 58-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and dyspnea. ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?", "answer": "Inferior STEMI", "options": {"A": "Stable angina", "B": "Inferior STEMI", "C": "Pulmonary embolism", "D": "Aortic dissection"}, "original_reasoning": "1: Chest pain->ST elevation->Myocardial infarction\n2: Inferior leads (II,III,aVF)->Right coronary artery territory->Inferior STEMI"}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique sample identifier |
| `question` | string | Medical question text |
| `answer` | string | Correct answer text |
| `options` | dict | Answer choices, e.g. `{"A": "...", "B": "...", "C": "...", "D": "..."}` |
| `original_reasoning` | string | Pre-existing reasoning chains, one per line, format: `"N: A->B->C"` |

### Where to get the input data

The source dataset is [MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason) on HuggingFace, which provides medical QA with graph-structured reasoning paths. Download and convert to the JSONL format above, or use MedXpertQA as an additional source.

The `original_reasoning` field contains graph-structured reasoning chains extracted from medical knowledge graphs. For the generation method, refer to [MedReason](https://github.com/UCSC-VLAA/MedReason) — MedVerse uses the same approach to construct `A->B->C` reasoning chains from medical QA pairs before feeding them into this pipeline.

### Generating `original_reasoning` from scratch

The code in [`medreason/`](medreason/) is adapted from [MedReason](https://github.com/UCSC-VLAA/MedReason) and can be called directly to generate `original_reasoning` chains from raw medical QA data. See [Generating `original_reasoning`](#generating-original_reasoning) below for setup and usage.

---

## Intermediate Formats

**After Step 1** — `Generate_Reasoning_Path.py` adds a `new_reasoning_path` field to each JSONL entry. This is a filtered and LLM-refined version of `original_reasoning`:

```
1: Chest pain->Coronary ischemia->ST elevation->Inferior STEMI
2: Diaphoresis->Sympathetic activation->Acute MI
```

**After Step 2** — `Generate_Initial_Plan.py` converts reasoning chains into a topologically-sorted DAG and produces a JSON dict:

```json
{
  "0": {
    "Plan Prompt": "<Plan>\n<Outline> 1. Chest pain; Dependency: [] </Outline>\n<Outline> 2. Coronary ischemia; Dependency: [1] </Outline>\n<Outline> 3. ST elevation; Dependency: [2] </Outline>\n<Outline> 4. Inferior STEMI; Dependency: [1, 3] </Outline>\n</Plan>",
    "Original Reason Path": "1: Chest pain->...",
    "Goal": "A 58-year-old male... Inferior STEMI",
    "Question": "A 58-year-old male...",
    "Options": {"A": "...", "B": "...", "C": "...", "D": "..."}
  }
}
```

**After Step 3** — `Generate_Transient_Data.py` generates the full structured training sample:

```json
{
  "0": {
    "Question": "...",
    "Options": {...},
    "Transient Plan Prompt": "<Plan>\n<Outline> Transient Step 1: Chest pain; Dependency: [] </Outline>\n...</Plan>",
    "Transient Execution Prompt": "<Execution>\n<Step> Transient Step 1: Chest pain presents as crushing, substernal pressure...\n</Step>\n...</Execution>",
    "Conclusion": "Explanation: The inferior ST-segment elevation in leads II, III, aVF combined with the clinical presentation localizes occlusion to the right coronary artery territory, consistent with an inferior STEMI.\nAnswer: B",
    "Eligible": 0
  }
}
```

The `Eligible` field tracks validation status: `0` = needs validation, `2` = fully validated.

---

## Quick Start

### 1. Install dependencies

```bash
conda activate medverse
pip install openai pandas lxml
```

### 2. Prepare input data

Place your input JSONL at `./jsonl/reasoning_path_raw.jsonl` following the format above.

### 3. Run the pipeline

```bash
export OPENAI_API_KEY="sk-..."
bash generate_data.sh
```

Override defaults with environment variables:

```bash
INPUT_JSONL=./jsonl/my_data.jsonl \
DATA_AMOUNT=5000 \
MAX_ITER=3 \
bash generate_data.sh
```

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `INPUT_JSONL` | `./jsonl/reasoning_path_raw.jsonl` | Input JSONL file |
| `DATA_AMOUNT` | `14000` | Target number of training samples |
| `MAX_ITER` | `5` | Max validation-regeneration iterations |

### 4. Run scripts individually

```bash
SCRIPTS="./generation"

# Step 1: Generate reasoning paths
python "$SCRIPTS/Generate_Reasoning_Path.py" \
    --input  ./jsonl/reasoning_path_raw.jsonl \
    --output ./jsonl/reasoning_path_raw.jsonl \
    --API_KEY "$OPENAI_API_KEY"

# Step 2: Build DAG plans
python "$SCRIPTS/Generate_Initial_Plan.py" \
    --input  ./jsonl/reasoning_path_raw.jsonl \
    --output ./json/MedVerse_Plan.json

# Step 3: Generate transient step data
python "$SCRIPTS/Generate_Transient_Data.py" \
    --input_json_path  ./json/MedVerse_Plan.json \
    --output_json_path ./json/MedVerse_transient.json \
    --data_amount 1000 \
    --API_KEY "$OPENAI_API_KEY"

# Step 4a: Validate XML structure
python "$SCRIPTS/check_plan_execution.py" \
    --input_json_path  ./json/MedVerse_transient.json \
    --output_json_path ./json/MedVerse_xml_checked.json

# Step 4b: Validate conclusions
python "$SCRIPTS/check_conclusion.py" \
    --input_json_path  ./json/MedVerse_xml_checked.json \
    --output_json_path ./json/MedVerse_validated.json \
    --API_KEY "$OPENAI_API_KEY"
```

---

## Output

The final validated dataset is a JSON dict where each value contains the structured training sample with `Transient Plan Prompt`, `Transient Execution Prompt`, and `Conclusion`. These three fields form the DAG-structured reasoning output that MedVerse is trained to produce.

The dataset is used directly by the training script:

```bash
bash train/scripts/launch_train_qwen.sh   # Qwen2.5-7B-Instruct

bash train/scripts/launch_train_llama.sh  # LLaMA-3.1-8B-Instruct
```

See [`train/README.md`](../train/README.md) for training details.

---

## Generating `original_reasoning`

The [`medreason/`](medreason/) directory contains code adapted from [MedReason](https://github.com/UCSC-VLAA/MedReason) to generate knowledge-grounded `A->B->C` reasoning chains from raw medical QA pairs. These chains become the `original_reasoning` field fed into Step 1 of this pipeline.

### How it works

1. Entity extraction — an LLM extracts medical entities from the question and answer
2. KG path finding — entities are matched to nodes in [PrimeKG](https://github.com/mims-harvard/PrimeKG), and shortest paths between question entities and answer entities are enumerated
3. Reasoning generation — an LLM uses the KG paths as scaffolding to write step-by-step reasoning chains in `A->B->C` format
4. Quality filtering — a second LLM judges whether the generated reasoning leads to the correct answer

### Setup

```bash
conda activate medverse
```

**Set your OpenAI API key** in `medreason/utils.py` or via environment variable:

```python
clients = {
    "gpt-5.2": {
        'api_key':  "YOUR_API_KEY",
    },
}
```

**Download PrimeKG** from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) and save as `primeKG.csv`.

**Generate node embeddings** (one-time, ~10 min on GPU):

```bash
cd medreason
python -c "import utils; utils.generate_node_embeddings(knowledge_graph_path='primeKG.csv')"
# Saves node_embeddings.pt in the current directory
```

### Usage

```bash
cd medreason

# Generate reasoning for medqa (50 samples)
python generate_reasoning.py --dataset medqa --sample 50

# Generate with parallel workers
python generate_reasoning.py --dataset medqa --sample 500 --batch_size 4

# Quality-filter the output
python quality_filtering.py --source_file medqa_0_50.jsonl
```

Output is written to `results/` (raw) and `results/quality_sampling/` (filtered). Each line is a JSONL record with `id`, `question`, `answer`, `options`, and `reasoning` — where `reasoning` maps to the `original_reasoning` field in the pipeline input format (reformat `N: A->B->C` chains as needed).

The dataset configurations in `configs/dataset_configs.yml` support: `medqa`, `MMLU`, `medmcqa`, `pubmedqa_artificial`, `pubmedqa_labeled`, `MedXpertQA`, `huatuo`, `LastHumanity`.

---

## Acknowledgements

The [`medreason/`](medreason/) directory contains code adapted from [**MedReason**](https://github.com/UCSC-VLAA/MedReason) (Wu et al., UCSC VLAA). MedReason introduces the methodology for constructing knowledge-grounded medical reasoning chains by linking medical QA entities through the PrimeKG knowledge graph. We gratefully acknowledge their contribution.

```
@article{wu2025medreason,
  title={MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graph},
  author={Wu, Juncheng and others},
  year={2025}
}
```
