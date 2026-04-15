"""Prepare MedVerse14k in Qwen2.5 ChatML format for fine-tuning."""
from datasets import load_dataset, Dataset, DatasetDict

HF_DATASET = "Jianwen/MedVerse14k"
OUTPUT_DS_DIR = "datasets/MedVerse14k"

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def messages_to_chatml(messages: list) -> str:
    """Convert HF messages list to Qwen2.5 ChatML format."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}\n<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}\n<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}\n<|im_end|>")
    return "\n".join(parts) + "\n"


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
            text = messages_to_chatml(messages)
            rows.append({"id": str(item["id"]), "text": text})

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dsd = DatasetDict(train=split["train"], test=split["test"])
    dsd.save_to_disk(OUTPUT_DS_DIR)
    print(dsd)
    print(f"Saved to {OUTPUT_DS_DIR}")


if __name__ == "__main__":
    main()
