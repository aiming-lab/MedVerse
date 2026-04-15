"""Prepare MedVerse14k in LLaMA-3 chat format for fine-tuning."""
from datasets import load_dataset, Dataset, DatasetDict

HF_DATASET = "Jianwen/MedVerse14k"
OUTPUT_DS_DIR = "datasets/MedVerse14k-LLaMA"

SYSTEM_PROMPT = "You are a helpful medical assistant."


def messages_to_llama(messages: list) -> str:
    """Convert HF messages list to LLaMA-3 chat format."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    return "".join(parts)


def main():
    print(f"Loading dataset from HuggingFace: {HF_DATASET} ...")
    hf_ds = load_dataset(HF_DATASET)

    rows = []
    for split_name in hf_ds:
        for item in hf_ds[split_name]:
            messages = item["messages"]
            # Ensure system prompt is set correctly
            if messages and messages[0]["role"] != "system":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            elif messages and messages[0]["role"] == "system":
                messages[0]["content"] = SYSTEM_PROMPT
            text = messages_to_llama(messages)
            rows.append({"id": str(item["id"]), "text": text})

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dsd = DatasetDict(train=split["train"], test=split["test"])
    dsd.save_to_disk(OUTPUT_DS_DIR)
    print(dsd)
    print(f"Saved to {OUTPUT_DS_DIR}")


if __name__ == "__main__":
    main()
