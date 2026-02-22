import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


def list_jsonl_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".jsonl")
    )


def render_options(options: Any) -> Tuple[str, List[str]]:
    labels: List[str] = []
    rendered: List[str] = []

    if isinstance(options, dict):
        for key, value in options.items():
            label = str(key).strip()[:1].upper()
            if not label:
                continue
            labels.append(label)
            rendered.append(f"{label}. {value}")
        return "\n".join(rendered), labels

    if isinstance(options, list):
        for idx, item in enumerate(options):
            label = chr(ord("A") + idx)
            text = item
            if isinstance(item, dict):
                item_label = str(item.get("label", "")).strip()[:1].upper()
                if item_label:
                    label = item_label
                text = item.get("text", item.get("content", ""))
            labels.append(label)
            rendered.append(f"{label}. {text}")
        return "\n".join(rendered), labels

    return "", labels


def load_samples(jsonl_path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(data.get("question", "")).strip()
            options = data.get("options") or data.get("choices", {})
            options_text, option_labels = render_options(options)
            answer_label = str(data.get("answer_idx", "")).strip()[:1].upper()
            if not answer_label:
                continue

            full_question = question
            if options_text:
                full_question = f"{question}\n\nAnswer Choices:\n{options_text}"

            sample_id = data.get("id", data.get("idx", i))
            samples.append(
                {
                    "id": sample_id,
                    "question": full_question,
                    "answer_idx": answer_label,
                    "valid_labels": option_labels,
                    "raw": data,
                }
            )
    return samples


def extract_prediction(text: str, valid_labels: List[str]) -> str:
    if not text:
        return ""
    labels_pattern = "|".join(re.escape(x) for x in valid_labels) if valid_labels else "[A-Z]"
    patterns = [
        rf"Answer\s*[:：]\s*(?:Option\s*)?({labels_pattern})\b",
        rf"Final\s+Answer\s*[:：]\s*({labels_pattern})\b",
        rf"^\s*({labels_pattern})[\s\.\)\]:：-]",
        rf"\b({labels_pattern})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return ""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a medical multiple-choice question assistant. "
                "Reply with only one uppercase option letter, such as A."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Please output only the single best option letter (A/B/C/...)."
            ),
        },
    ]


async def infer_one(
    client: AsyncOpenAI,
    model_name: str,
    sample: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=build_messages(sample["question"]),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output_text = response.choices[0].message.content or ""
        pred = extract_prediction(output_text, sample["valid_labels"])
        gold = sample["answer_idx"]
        correct = pred == gold
        return {
            "id": sample["id"],
            "prediction": pred,
            "gold": gold,
            "correct": correct,
            "output": output_text,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": sample["id"],
            "prediction": "",
            "gold": sample["answer_idx"],
            "correct": False,
            "output": "",
            "error": str(exc),
        }


async def evaluate_file(
    client: AsyncOpenAI,
    model_name: str,
    samples: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    if not samples:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "elapsed_sec": 0.0, "samples": []}

    start = time.perf_counter()
    details: List[Dict[str, Any]] = []
    correct = 0
    effective_batch = max(1, min(batch_size, len(samples)))

    for i in range(0, len(samples), effective_batch):
        batch = samples[i : i + effective_batch]
        tasks = [
            infer_one(client, model_name, sample, temperature, max_tokens)
            for sample in batch
        ]
        outputs = await asyncio.gather(*tasks)
        for item in outputs:
            if item["correct"]:
                correct += 1
            details.append(item)

    elapsed = time.perf_counter() - start
    total = len(samples)
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_sec": elapsed,
        "samples": details,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/Qwen2.5-14B-Instruct on medical benchmark JSONL files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jimchen/MedVerse/eval_data",
        help="Directory containing benchmark jsonl files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Model name served by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8010/v1",
        help="OpenAI-compatible endpoint URL.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the endpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of requests to send concurrently.",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional sample cap for each file.",
    )
    parser.add_argument(
        "--max_samples_total",
        type=int,
        default=None,
        help="Optional sample cap across all files.",
    )
    parser.add_argument(
        "--include_pattern",
        type=str,
        default="",
        help="Only evaluate files whose name contains this substring.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen25_benchmark_results",
        help="Directory to save per-file and overall results.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Maximum generation tokens per question.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key)

    files = list_jsonl_files(args.data_dir)
    if args.include_pattern:
        files = [x for x in files if args.include_pattern in os.path.basename(x)]
    if not files:
        print(f"No matched jsonl files found in {args.data_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    remaining = args.max_samples_total
    overall_total = 0
    overall_correct = 0
    overall_elapsed = 0.0
    file_summaries: List[Dict[str, Any]] = []

    print("Starting evaluation...")
    for file_path in files:
        if remaining is not None and remaining <= 0:
            break

        samples = load_samples(file_path, limit=args.max_samples_per_file)
        if remaining is not None:
            samples = samples[:remaining]
            remaining -= len(samples)
        if not samples:
            continue

        file_name = os.path.basename(file_path)
        print(f"Evaluating {file_name} ({len(samples)} samples)")
        result = await evaluate_file(
            client=client,
            model_name=args.model_name,
            samples=samples,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        total = result["total"]
        correct = result["correct"]
        accuracy = result["accuracy"]
        elapsed = result["elapsed_sec"]
        overall_total += total
        overall_correct += correct
        overall_elapsed += elapsed

        print(
            f"{file_name}: {correct}/{total} ({accuracy * 100:.2f}%), "
            f"time={elapsed:.2f}s"
        )

        payload = {
            "model_name": args.model_name,
            "base_url": base_url,
            "file": file_name,
            "result": result,
        }
        output_file = os.path.join(
            args.output_dir, f"{os.path.splitext(file_name)[0]}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        file_summaries.append(
            {
                "file": file_name,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "elapsed_sec": elapsed,
                "output_file": output_file,
            }
        )

    overall_accuracy = overall_correct / overall_total if overall_total else 0.0
    overall = {
        "model_name": args.model_name,
        "base_url": base_url,
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_accuracy,
        "overall_elapsed_sec": overall_elapsed,
        "files": file_summaries,
    }
    overall_path = os.path.join(args.output_dir, "overall_summary.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n===== Overall =====")
    print(
        f"Accuracy: {overall_correct}/{overall_total} "
        f"({overall_accuracy * 100:.2f}%), time={overall_elapsed:.2f}s"
    )
    print(f"Summary saved to: {overall_path}")


if __name__ == "__main__":
    asyncio.run(main())
