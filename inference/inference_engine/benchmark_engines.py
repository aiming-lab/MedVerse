import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

import inference_engine_api as api


OUTLINE_PATTERN = re.compile(
    r"<Outline>\s*Transient Step\s+(\d+):.*?Dependency:\s*\[(.*?)\]\s*</Outline>",
    re.DOTALL | re.IGNORECASE,
)


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


async def run_engine_with_timing(engine, question: str):
    start = time.perf_counter()
    raw_output = await engine.run(question)
    latency_sec = time.perf_counter() - start
    phase_timings = None

    if isinstance(raw_output, tuple):
        output_text = raw_output[0]
        if len(raw_output) > 1 and isinstance(raw_output[1], dict):
            phase_timings = raw_output[1]
    else:
        output_text = raw_output

    return output_text, latency_sec, phase_timings


def extract_hybrid_dag_width(output: str) -> Optional[int]:
    if not output:
        return None

    # Use "Finding Reasoning Path" entries as topology branches.
    # Example:
    # Finding Reasoning Path:
    # 1: xxx
    # 2: yyy
    # <Plan>
    path_block_match = re.search(
        r"Finding\s+Reasoning\s+Path:\s*(.*?)(?:<Plan>|$)",
        output,
        re.DOTALL | re.IGNORECASE,
    )
    if not path_block_match:
        return None

    block = path_block_match.group(1).strip()
    if not block:
        return None

    numbered_paths = re.findall(r"^\s*\d+\s*[:：]\s*(.+)$", block, re.MULTILINE)
    if numbered_paths:
        valid_paths = [x for x in numbered_paths if x.strip()]
        if not valid_paths:
            return None
        if len(valid_paths) == 1:
            return 1

        # If multiple paths have no repeated intermediate entities,
        # treat them as effectively linear (w=1).
        # Path format example: a -> b -> c
        intermediate_seen = set()
        has_repeated_intermediate = False
        for path in valid_paths:
            nodes = [n.strip().lower() for n in path.split("->") if n.strip()]
            if len(nodes) < 3:
                continue
            intermediates = nodes[1:-1]
            for ent in intermediates:
                if ent in intermediate_seen:
                    has_repeated_intermediate = True
                    break
                intermediate_seen.add(ent)
            if has_repeated_intermediate:
                break

        return len(valid_paths) if has_repeated_intermediate else 1

    # Fallback for some generations that omit numeric prefixes.
    arrow_lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and "->" in line and not line.strip().startswith("<")
    ]
    if arrow_lines:
        if len(arrow_lines) == 1:
            return 1

        intermediate_seen = set()
        has_repeated_intermediate = False
        for line in arrow_lines:
            nodes = [n.strip().lower() for n in line.split("->") if n.strip()]
            if len(nodes) < 3:
                continue
            for ent in nodes[1:-1]:
                if ent in intermediate_seen:
                    has_repeated_intermediate = True
                    break
                intermediate_seen.add(ent)
            if has_repeated_intermediate:
                break

        return len(arrow_lines) if has_repeated_intermediate else 1

    return None


def compute_hybrid_topology_stats(samples: List[Dict[str, object]]) -> Dict[str, object]:
    def summarize(bucket: List[Dict[str, object]]) -> Dict[str, object]:
        total = len(bucket)
        if total == 0:
            return {
                "count": 0,
                "accuracy": None,
                "avg_latency_sec": None,
                "execution_first_char_avg_sec": None,
            }
        corr = sum(1 for x in bucket if bool(x.get("correct", False)))
        total_latency = sum(float(x.get("latency_sec", 0.0)) for x in bucket)
        first_char_values = [
            float(x.get("phase_timings", {}).get("execution_first_char_avg_sec"))
            for x in bucket
            if isinstance(x.get("phase_timings"), dict)
            and x.get("phase_timings", {}).get("execution_first_char_avg_sec") is not None
        ]
        return {
            "count": total,
            "accuracy": corr / total,
            "avg_latency_sec": total_latency / total,
            "execution_first_char_avg_sec": (
                sum(first_char_values) / len(first_char_values) if first_char_values else None
            ),
        }

    width_known = [x for x in samples if x.get("dag_width") is not None]
    width_unknown = [x for x in samples if x.get("dag_width") is None]
    width_eq_1 = [x for x in width_known if x.get("dag_width") == 1]
    width_ge_2 = [x for x in width_known if int(x.get("dag_width", 0)) >= 2]

    linear = width_eq_1
    parallel = width_ge_2
    known_total = len(width_known)

    return {
        "parsed_samples": known_total,
        "unparsed_samples": len(width_unknown),
        "linear_vs_parallel_ratio": {
            "linear_count": len(linear),
            "parallel_count": len(parallel),
            "linear_ratio": (len(linear) / known_total) if known_total else None,
            "parallel_ratio": (len(parallel) / known_total) if known_total else None,
        },
        "by_width_bucket": {
            "width_eq_1": summarize(width_eq_1),
            "width_ge_2": summarize(width_ge_2),
        },
    }


def compute_hybrid_phase_time_stats(samples: List[Dict[str, object]]) -> Dict[str, object]:
    phase_samples = [x for x in samples if isinstance(x.get("phase_timings"), dict)]
    if not phase_samples:
        return {
            "count": 0,
            "totals_sec": {
                "planning_sec": 0.0,
                "parsing_scheduling_sec": 0.0,
                "execution_sec": 0.0,
            },
            "ratios": {
                "planning": None,
                "parsing_scheduling": None,
                "execution": None,
            },
            "avg_per_sample_sec": {
                "planning_sec": None,
                "parsing_scheduling_sec": None,
                "execution_sec": None,
                "execution_first_char_avg_sec": None,
            },
        }

    planning_total = 0.0
    parsing_sched_total = 0.0
    execution_total = 0.0
    first_char_values = []

    for item in phase_samples:
        timings = item["phase_timings"]
        planning_total += float(timings.get("planning_sec", 0.0))
        parsing_sched_total += float(timings.get("parsing_scheduling_sec", 0.0))
        execution_total += float(timings.get("execution_sec", 0.0))
        if timings.get("execution_first_char_avg_sec") is not None:
            first_char_values.append(float(timings["execution_first_char_avg_sec"]))

    grand_total = planning_total + parsing_sched_total + execution_total
    count = len(phase_samples)
    return {
        "count": count,
        "totals_sec": {
            "planning_sec": planning_total,
            "parsing_scheduling_sec": parsing_sched_total,
            "execution_sec": execution_total,
        },
        "ratios": {
            "planning": (planning_total / grand_total) if grand_total > 0 else None,
            "parsing_scheduling": (parsing_sched_total / grand_total) if grand_total > 0 else None,
            "execution": (execution_total / grand_total) if grand_total > 0 else None,
        },
        "avg_per_sample_sec": {
            "planning_sec": planning_total / count,
            "parsing_scheduling_sec": parsing_sched_total / count,
            "execution_sec": execution_total / count,
            "execution_first_char_avg_sec": (
                sum(first_char_values) / len(first_char_values) if first_char_values else None
            ),
        },
    }


def compute_serial_stats_by_hybrid_topology(
    hybrid_samples: List[Dict[str, object]], serial_samples: List[Dict[str, object]]
) -> Dict[str, object]:
    def summarize(bucket: List[Dict[str, object]]) -> Dict[str, object]:
        total = len(bucket)
        if total == 0:
            return {
                "count": 0,
                "accuracy": None,
                "avg_latency_sec": None,
                "execution_first_char_avg_sec": None,
            }
        corr = sum(1 for x in bucket if bool(x.get("correct", False)))
        latencies = [
            float(x["latency_sec"])
            for x in bucket
            if x.get("latency_sec") is not None
        ]
        avg_latency = (sum(latencies) / len(latencies)) if latencies else None
        first_char_latencies = [
            float(x["execution_first_char_avg_sec"])
            for x in bucket
            if x.get("execution_first_char_avg_sec") is not None
        ]
        avg_first_char = (
            sum(first_char_latencies) / len(first_char_latencies)
            if first_char_latencies
            else None
        )
        return {
            "count": total,
            "accuracy": corr / total,
            "avg_latency_sec": avg_latency,
            "execution_first_char_avg_sec": avg_first_char,
        }

    serial_by_id = {str(x.get("id")): x for x in serial_samples}
    paired_samples = []
    unpaired_count = 0
    for h in hybrid_samples:
        sid = str(h.get("id"))
        s = serial_by_id.get(sid)
        if s is None:
            unpaired_count += 1
            continue
        paired_samples.append(
            {
                "id": sid,
                "hybrid_dag_width": h.get("dag_width"),
                "hybrid_dag_topology": h.get("dag_topology"),
                "serial_correct": s.get("correct", False),
                "serial_latency_sec": s.get("latency_sec"),
                "hybrid_execution_first_char_avg_sec": (
                    h.get("phase_timings", {}).get("execution_first_char_avg_sec")
                    if isinstance(h.get("phase_timings"), dict)
                    else None
                ),
            }
        )

    width_known = [x for x in paired_samples if x.get("hybrid_dag_width") is not None]
    width_eq_1 = [x for x in width_known if x.get("hybrid_dag_width") == 1]
    width_ge_2 = [x for x in width_known if int(x.get("hybrid_dag_width", 0)) >= 2]

    linear = width_eq_1
    parallel = width_ge_2
    known_total = len(width_known)

    # Adapt key names for summarize function.
    def remap_for_summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return [
            {
                "correct": r.get("serial_correct", False),
                "latency_sec": r.get("serial_latency_sec"),
                "execution_first_char_avg_sec": r.get("hybrid_execution_first_char_avg_sec"),
            }
            for r in rows
        ]

    return {
        "paired_samples": len(paired_samples),
        "unpaired_samples": unpaired_count,
        "hybrid_width_parsed_samples": known_total,
        "serial_linear_vs_parallel_by_hybrid": {
            "linear_count": len(linear),
            "parallel_count": len(parallel),
            "linear_ratio": (len(linear) / known_total) if known_total else None,
            "parallel_ratio": (len(parallel) / known_total) if known_total else None,
            "linear_bucket_serial": summarize(remap_for_summarize(linear)),
            "parallel_bucket_serial": summarize(remap_for_summarize(parallel)),
        },
        "serial_by_hybrid_width_bucket": {
            "width_eq_1": summarize(remap_for_summarize(width_eq_1)),
            "width_ge_2": summarize(remap_for_summarize(width_ge_2)),
        },
    }


async def evaluate_engine(
    engine, samples: List[Dict[str, str]], batch_size: int, label: str, collect_hybrid_topology: bool = False
):
    if not samples:
        return 0, 0, 0.0, []

    start_time = time.perf_counter()
    correct = 0
    per_sample = []

    async def run_batch(questions):
        tasks = [run_engine_with_timing(engine, q) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=True)

    effective_batch_size = min(batch_size, len(samples))
    for start in range(0, len(samples), effective_batch_size):
        batch = samples[start : start + effective_batch_size]
        questions = [item["question"] for item in batch]
        outputs = await run_batch(questions)

        for item, out in zip(batch, outputs):
            answer_idx = item["answer_idx"]
            if isinstance(out, Exception):
                print(f"[{label}] Error on {item['id']}: {out}")
                per_sample.append(
                    {
                        "id": item["id"],
                        "output": None,
                        "correct": False,
                        "correct_answer": answer_idx,
                        "error": str(out),
                        "latency_sec": None,
                        "dag_width": None,
                        "dag_topology": None,
                    }
                )
            else:
                output_text, latency_sec, phase_timings = out
                is_correct = check_correctness(output_text, answer_idx)
                dag_width = (
                    extract_hybrid_dag_width(output_text) if collect_hybrid_topology else None
                )
                dag_topology = (
                    ("linear" if dag_width == 1 else "parallel") if dag_width is not None else None
                )
                if is_correct:
                    correct += 1
                per_sample.append(
                    {
                        "id": item["id"],
                        "output": output_text,
                        "correct": is_correct,
                        "correct_answer": answer_idx,
                        "latency_sec": latency_sec,
                        "dag_width": dag_width,
                        "dag_topology": dag_topology,
                        "phase_timings": phase_timings,
                    }
                )

    elapsed = time.perf_counter() - start_time
    return len(samples), correct, elapsed, per_sample


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jimchen/MedVerse/eval_data",
        help="Directory containing jsonl evaluation files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MedVerse",
        help="Model name/path for vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8010/v1",
        help="vLLM API base URL.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to evaluate concurrently.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to write per-dataset JSON results.",
    )
    args = parser.parse_args()

    api.MODEL_NAME = args.model_name
    base_url = args.vllm_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api.VLLM_API_URL = base_url
    api.VLLM_API_KEY = args.vllm_api_key
    api.aclient = AsyncOpenAI(base_url=api.VLLM_API_URL, api_key=api.VLLM_API_KEY)

    hybrid_engine = api.HybridInferenceEngine()
    serial_engine = api.SerialBaselineEngine()

    files = iter_jsonl_files(args.data_dir)
    if not files:
        print(f"No jsonl files found in {args.data_dir}")
        return

    remaining = args.max_samples_total
    print("Evaluating files:")
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in files:
        if file_path.find('HLE_biomed') == -1:
            continue
        if remaining is not None and remaining <= 0:
            break

        samples = load_samples(file_path, limit=args.max_samples_per_file)
        samples = samples[:20]
        if remaining is not None:
            samples = samples[:remaining]
        remaining = None if remaining is None else remaining - len(samples)

        if not samples:
            continue

        print(f"\nRunning HybridInferenceEngine on {os.path.basename(file_path)}...")
        total, hybrid_correct, hybrid_time, hybrid_samples = await evaluate_engine(
            hybrid_engine, samples, args.batch_size, "Hybrid", collect_hybrid_topology=True
        )
        h_acc = hybrid_correct / total if total else 0.0
        hybrid_topology_stats = compute_hybrid_topology_stats(hybrid_samples)
        hybrid_phase_time_stats = compute_hybrid_phase_time_stats(hybrid_samples)

        print(f"Running SerialBaselineEngine on {os.path.basename(file_path)}...")
        _, serial_correct, serial_time, serial_samples = await evaluate_engine(
            serial_engine, samples, args.batch_size, "Serial"
        )
        s_acc = serial_correct / total if total else 0.0
        serial_by_hybrid_stats = compute_serial_stats_by_hybrid_topology(
            hybrid_samples, serial_samples
        )

        print(
            f"{os.path.basename(file_path)}: Hybrid {hybrid_correct}/{total} ({h_acc * 100:.2f}%) "
            f"[{hybrid_time:.2f}s] | Serial {serial_correct}/{total} ({s_acc * 100:.2f}%) "
            f"[{serial_time:.2f}s]"
        )
        ratio = hybrid_topology_stats["linear_vs_parallel_ratio"]
        buckets = hybrid_topology_stats["by_width_bucket"]
        print(
            "Hybrid topology ratio: "
            f"linear={ratio['linear_count']}, parallel={ratio['parallel_count']}, "
            f"linear_ratio={ratio['linear_ratio']}, parallel_ratio={ratio['parallel_ratio']}"
        )
        phase_ratio = hybrid_phase_time_stats["ratios"]
        print(
            "Hybrid phase time ratio: "
            f"planning={phase_ratio['planning']}, "
            f"parsing+scheduling={phase_ratio['parsing_scheduling']}, "
            f"execution={phase_ratio['execution']}"
        )
        phase_avg = hybrid_phase_time_stats["avg_per_sample_sec"]
        print(
            "Hybrid phase avg per sample: "
            f"planning={phase_avg['planning_sec']}, "
            f"parsing+scheduling={phase_avg['parsing_scheduling_sec']}, "
            f"execution={phase_avg['execution_sec']}, "
            f"execution_first_char_avg_sec={phase_avg['execution_first_char_avg_sec']}"
        )
        print(
            "Hybrid by width: "
            f"w=1(acc={buckets['width_eq_1']['accuracy']},lat={buckets['width_eq_1']['avg_latency_sec']},"
            f"ttfc={buckets['width_eq_1']['execution_first_char_avg_sec']}), "
            f"w>=2(acc={buckets['width_ge_2']['accuracy']},lat={buckets['width_ge_2']['avg_latency_sec']},"
            f"ttfc={buckets['width_ge_2']['execution_first_char_avg_sec']})"
        )
        serial_buckets = serial_by_hybrid_stats["serial_by_hybrid_width_bucket"]
        print(
            "Serial by Hybrid width: "
            f"w=1(acc={serial_buckets['width_eq_1']['accuracy']},lat={serial_buckets['width_eq_1']['avg_latency_sec']},"
            f"ttfc={serial_buckets['width_eq_1']['execution_first_char_avg_sec']}), "
            f"w>=2(acc={serial_buckets['width_ge_2']['accuracy']},lat={serial_buckets['width_ge_2']['avg_latency_sec']},"
            f"ttfc={serial_buckets['width_ge_2']['execution_first_char_avg_sec']})"
        )
        results.append(
            {
                "file": os.path.basename(file_path),
                "total": total,
                "hybrid_correct": hybrid_correct,
                "hybrid_accuracy": h_acc,
                "hybrid_time_sec": hybrid_time,
                "hybrid_topology_stats": hybrid_topology_stats,
                "hybrid_phase_time_stats": hybrid_phase_time_stats,
                "hybrid_samples": hybrid_samples,
                "serial_correct": serial_correct,
                "serial_accuracy": s_acc,
                "serial_time_sec": serial_time,
                "serial_by_hybrid_topology_stats": serial_by_hybrid_stats,
                "serial_samples": serial_samples,
            }
        )

        output_payload = {
            "model_name": args.model_name,
            "vllm_url": api.VLLM_API_URL,
            "batch_size": args.batch_size,
            "max_samples_per_file": args.max_samples_per_file,
            "max_samples_total": args.max_samples_total,
            "result": results[-1],
        }
        output_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
        output_path = os.path.join(args.output_dir, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON results to {output_path}")
        break


if __name__ == "__main__":
    asyncio.run(main())
