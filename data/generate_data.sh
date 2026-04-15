#!/usr/bin/env bash
# MedVerse Data Generation Pipeline
#
# Generates the MedVerse14k training dataset in four stages:
#   1. Generate reasoning paths from raw medical QA
#   2. Convert reasoning paths into DAG plans
#   3. Generate transient step data via LLM
#   4. Iterative validation loop: XML check -> conclusion check -> regenerate
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   bash generate_data.sh
#
# Override defaults with environment variables:
#   INPUT_JSONL=./jsonl/my_data.jsonl DATA_AMOUNT=5000 MAX_ITER=3 bash generate_data.sh

set -euo pipefail

API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/generation"

# ── File paths ────────────────────────────────────────────────────────────────
INPUT_JSONL="${INPUT_JSONL:-./jsonl/reasoning_path_raw.jsonl}"
PLAN_JSON="${PLAN_JSON:-./json/MedVerse_Plan.json}"
TRANSIENT_JSON="${TRANSIENT_JSON:-./json/MedVerse_transient.json}"
XML_CHECKED_JSON="${XML_CHECKED_JSON:-./json/MedVerse_xml_checked.json}"
VALIDATED_JSON="${VALIDATED_JSON:-./json/MedVerse_validated.json}"

# ── Config ────────────────────────────────────────────────────────────────────
DATA_AMOUNT="${DATA_AMOUNT:-14000}"
MAX_ITER="${MAX_ITER:-5}"

mkdir -p ./jsonl ./json

# ── Step 1: Generate reasoning paths ─────────────────────────────────────────
echo "[1/4] Generating reasoning paths..."
python "$SCRIPTS/Generate_Reasoning_Path.py" \
    --input  "$INPUT_JSONL" \
    --output "$INPUT_JSONL" \
    --API_KEY "$API_KEY"

# ── Step 2: Build initial DAG plans ──────────────────────────────────────────
echo "[2/4] Building initial DAG plans..."
python "$SCRIPTS/Generate_Initial_Plan.py" \
    --input  "$INPUT_JSONL" \
    --output "$PLAN_JSON"

# ── Step 3: Generate transient step data ─────────────────────────────────────
echo "[3/4] Generating transient step data (target: $DATA_AMOUNT samples)..."
python "$SCRIPTS/Generate_Transient_Data.py" \
    --input_json_path  "$PLAN_JSON" \
    --output_json_path "$TRANSIENT_JSON" \
    --data_amount      "$DATA_AMOUNT" \
    --API_KEY          "$API_KEY"

# ── Step 4: Validation + regeneration loop ───────────────────────────────────
echo "[4/4] Starting validation loop (max $MAX_ITER iterations)..."
for i in $(seq 1 "$MAX_ITER"); do
    echo "=== Iteration $i / $MAX_ITER ==="

    # 4a. Filter out malformed XML (Plan / Execution structure)
    python "$SCRIPTS/check_plan_execution.py" \
        --input_json_path  "$TRANSIENT_JSON" \
        --output_json_path "$XML_CHECKED_JSON"

    # 4b. Filter out inconsistent conclusions
    python "$SCRIPTS/check_conclusion.py" \
        --input_json_path  "$XML_CHECKED_JSON" \
        --output_json_path "$VALIDATED_JSON" \
        --API_KEY          "$API_KEY"

    # 4c. Regenerate samples that failed validation
    output=$(python "$SCRIPTS/Generate_Transient_Data.py" \
        --input_json_path  "$VALIDATED_JSON" \
        --output_json_path "$TRANSIENT_JSON" \
        --data_amount      "$DATA_AMOUNT" \
        --API_KEY          "$API_KEY" 2>&1)
    echo "$output"

    if echo "$output" | grep -q "ALL DATA ARE ELIGIBLE."; then
        echo "All data eligible. Stopping early."
        break
    fi
done

echo "Done. Final dataset: $VALIDATED_JSON"
