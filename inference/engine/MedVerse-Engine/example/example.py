"""
MedVerse quick-start example.

Sends medical questions from a directory of .txt files to a running
MedVerse server and prints the structured Plan → Steps → Conclusion output.

Usage:
    # 1. Start the server (in a separate terminal):
    #    python -m sglang.srt.entrypoints.medverse_server \
    #        --model-path /path/to/MedVerse-14B \
    #        --tp-size 1 --port 30000 --trust-remote-code

    # 2. Run this script:
    python example.py \
        --server_url http://localhost:30000 \
        --prompts_dir ./prompt
"""

import os
import json
import argparse
import urllib.request

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MEDVERSE_THINK_PREFIX = "<Think>\n"


def build_prompt(question: str) -> str:
    """Construct the MedVerse prompt format (matches MedVerse14k training format)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuestion:\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{MEDVERSE_THINK_PREFIX}"
    )


def generate(server_url: str, prompt: str, max_new_tokens: int = 3072) -> str:
    payload = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
            "stop": ["<|im_end|>"],
        },
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result.get("text", "")


def main(args):
    prompts_dir = args.prompts_dir
    files = sorted(f for f in os.listdir(prompts_dir) if f.endswith(".txt"))

    if not files:
        print(f"No .txt files found in {prompts_dir}")
        return

    for fname in files:
        path = os.path.join(prompts_dir, fname)
        with open(path) as f:
            question = f.read().strip()

        print("=" * 70)
        print(f"[{fname}] Question:\n{question}\n")

        prompt = build_prompt(question)
        output = generate(args.server_url, prompt, max_new_tokens=args.max_new_tokens)

        print(f"[{fname}] MedVerse output:\n{output}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedVerse quick-start example")
    parser.add_argument(
        "--server_url", default="http://localhost:30000",
        help="URL of the running MedVerse server",
    )
    parser.add_argument(
        "--prompts_dir", default="./prompt",
        help="Directory containing .txt prompt files (one question per file)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=3072,
        help="Maximum tokens to generate per question",
    )
    main(parser.parse_args())
