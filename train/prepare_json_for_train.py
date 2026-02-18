import json
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

RAW_JSON_PATH = "./json/MedVerse14k.json"
OUTPUT_DS_DIR = "MedVerse14k"
OUTPUT_JSONL = "./json/MedVerse14k.jsonl"

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
BRIDGE_SENTENCE = (
    "First find a reasoning path, then transform that path into an outlined plan, then execute the plan, and finally synthesize a concise conclusion."
)

def build_chatml(item: Dict[str, Any]) -> str:
    """Convert one JSON object into the ChatML training format."""
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

    chatml = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_block}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_block}\n"
        "<|im_end|>\n"
    )
    return chatml

def main():
    raw = json.loads(Path(RAW_JSON_PATH).read_text(encoding="utf-8"))

    # Top-level may be dict (id->obj) or list[obj]
    samples = list(raw.values()) if isinstance(raw, dict) else raw

    rows: List[Dict[str, str]] = []
    for i, item in enumerate(samples):
        text = build_chatml(item)
        rows.append({"id": str(i), "text": text})

    # Save as Hugging Face Dataset
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dsd = DatasetDict(train=split["train"], test=split["test"])
    dsd.save_to_disk(OUTPUT_DS_DIR)

    # Optional preview
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows[:5000]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(dsd)

if __name__ == "__main__":
    main()