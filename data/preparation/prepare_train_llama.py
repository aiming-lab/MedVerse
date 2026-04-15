"""Prepare MedVerse14k in LLaMA-3 chat format for fine-tuning."""
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

RAW_JSON_PATH = "./json/MedVerse14k.json"
OUTPUT_DS_DIR = "MedVerse14k-LLaMA"

SYSTEM_PROMPT = "You are a helpful medical assistant."
BRIDGE_SENTENCE = (
    "First find a reasoning path, then transform that path into an outlined plan, "
    "then execute the plan, and finally synthesize a concise conclusion."
)


def build_llama_chat(item: Dict[str, Any]) -> str:
    Q = ("Question: \n" + item.get("Question", "") or "").strip()
    OPT = (item.get("Options", "") or "").strip()
    PATH = (item.get("Original Reason Path", "") or "").strip()
    PLAN = (item.get("Transient Plan Prompt", "") or "").strip()
    EXEC = (item.get("Transient Execution Prompt", "") or "").strip()
    CONC = (item.get("Conclusion", "") or "").strip()
    final_answer = (item.get("Goal", "") or "").strip()

    user_block = "\n".join([x for x in [Q, OPT] if x])

    assistant_parts = [
        "<Think>",
        BRIDGE_SENTENCE,
        "Finding Reasoning Path:",
        PATH,
        PLAN,
        EXEC,
        "<Conclusion>",
        CONC,
        "</Conclusion>",
        "</Think>",
        f"Answer: {final_answer}",
    ]
    assistant_block = "\n".join([p for p in assistant_parts if p])

    # LLaMA-3 chat format
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_block}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_block}<|eot_id|>"
    )
    return text


def main():
    raw = json.loads(Path(RAW_JSON_PATH).read_text(encoding="utf-8"))
    samples = list(raw.values()) if isinstance(raw, dict) else raw

    rows: List[Dict[str, str]] = []
    for i, item in enumerate(samples):
        text = build_llama_chat(item)
        rows.append({"id": str(i), "text": text})

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dsd = DatasetDict(train=split["train"], test=split["test"])
    dsd.save_to_disk(OUTPUT_DS_DIR)
    print(dsd)
    print(f"Saved to {OUTPUT_DS_DIR}")


if __name__ == "__main__":
    main()
