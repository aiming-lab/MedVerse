import argparse
import asyncio
import json
import os
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

import inference_engine_api as api


def iter_jsonl_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(
        [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.endswith(".jsonl")
        ]
    )


def format_options(options: object) -> str:
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])
    if isinstance(options, list):
        # Attempt to render list of options cleanly.
        rendered = []
        for idx, item in enumerate(options):
            if isinstance(item, dict):
                label = item.get("label")
                text = item.get("text", item.get("content", ""))
                if label:
                    rendered.append(f"{label}. {text}")
                else:
                    rendered.append(f"{chr(ord('A') + idx)}. {text}")
            else:
                rendered.append(f"{chr(ord('A') + idx)}. {item}")
        return "\n".join(rendered)
    return ""


def load_samples(jsonl_path: str, limit: Optional[int]) -> List[Dict[str, str]]:
    samples = []
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
            question_text = data.get("question", "")
            options = data.get("options") or data.get("choices", {})
            options_str = format_options(options)
            formatted_question = (
                f"{question_text}\nAnswer Choices:\n{options_str}"
                if options_str
                else question_text
            )
            samples.append(
                {
                    "id": data.get("idx", i),
                    "question": formatted_question,
                    "answer_idx": data.get("answer_idx", ""),
                }
            )
    return samples


def check_correctness(model_output: str, correct_label: str) -> bool:
    if not correct_label:
        return False
    conclusion_match = re.search(
        r"<Conclusion>(.*?)</Conclusion>",
        model_output,
        re.DOTALL | re.IGNORECASE,
    )
    search_content = conclusion_match.group(1) if conclusion_match else model_output
    pattern = re.compile(rf"Answer:\s*(?:Option\s*)?({correct_label})\b", re.IGNORECASE)
    if pattern.search(search_content):
        return True
    return bool(re.search(rf"\b{correct_label}\b", search_content[-200:]))


async def run_engine(engine, question: str) -> str:
    output = await engine.run(question)
    if isinstance(output, tuple):
        return output[0]
    return output


async def evaluate_file(
    file_path: str,
    hybrid_engine,
    serial_engine,
    max_samples: Optional[int],
    max_total_remaining: Optional[int],
):
    samples = load_samples(file_path, limit=max_samples)
    if max_total_remaining is not None:
        samples = samples[: max_total_remaining]

    if not samples:
        return 0, 0, 0

    hybrid_correct = 0
    serial_correct = 0

    for sample in samples:
        question = sample["question"]
        answer_idx = sample["answer_idx"]

        try:
            hybrid_output = await run_engine(hybrid_engine, question)
            if check_correctness(hybrid_output, answer_idx):
                hybrid_correct += 1
        except Exception as e:
            print(f"[Hybrid] Error on {sample['id']}: {e}")

        try:
            serial_output = await run_engine(serial_engine, question)
            if check_correctness(serial_output, answer_idx):
                serial_correct += 1
        except Exception as e:
            print(f"[Serial] Error on {sample['id']}: {e}")

    return len(samples), hybrid_correct, serial_correct


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jimchen/Multiverse/eval_data",
        help="Directory containing jsonl evaluation files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=api.MODEL_NAME,
        help="Model name/path for vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default=api.VLLM_API_URL,
        help="vLLM API base URL.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default=api.VLLM_API_KEY,
        help="API key for vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Limit samples per file (default: all).",
    )
    parser.add_argument(
        "--max_samples_total",
        type=int,
        default=None,
        help="Global cap across all files (default: all).",
    )
    args = parser.parse_args()

    api.MODEL_NAME = args.model_name
    api.VLLM_API_URL = args.vllm_url
    api.VLLM_API_KEY = args.vllm_api_key
    api.aclient = AsyncOpenAI(base_url=api.VLLM_API_URL, api_key=api.VLLM_API_KEY)

    hybrid_engine = api.HybridInferenceEngine()
    serial_engine = api.SerialBaselineEngine()

    files = iter_jsonl_files(args.data_dir)
    if not files:
        print(f"No jsonl files found in {args.data_dir}")
        return

    total_samples = 0
    total_hybrid_correct = 0
    total_serial_correct = 0
    remaining = args.max_samples_total

    print("Evaluating files:")
    for file_path in files:
        if remaining is not None and remaining <= 0:
            break
        count, h_corr, s_corr = await evaluate_file(
            file_path,
            hybrid_engine,
            serial_engine,
            args.max_samples_per_file,
            remaining,
        )
        remaining = None if remaining is None else remaining - count
        total_samples += count
        total_hybrid_correct += h_corr
        total_serial_correct += s_corr

        if count > 0:
            print(
                f"{os.path.basename(file_path)}: "
                f"Hybrid {h_corr}/{count}, Serial {s_corr}/{count}"
            )

    if total_samples == 0:
        print("No samples evaluated.")
        return

    hybrid_acc = total_hybrid_correct / total_samples
    serial_acc = total_serial_correct / total_samples

    print("\n===== Final Accuracy =====")
    print(f"Total Samples: {total_samples}")
    print(f"HybridInferenceEngine: {total_hybrid_correct}/{total_samples} ({hybrid_acc:.4f})")
    print(f"SerialBaselineEngine:  {total_serial_correct}/{total_samples} ({serial_acc:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
