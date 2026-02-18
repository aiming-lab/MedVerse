#!/bin/bash

# Step 1: run Generate_Transient_Data.py once

# LOG_FILE="logs/run_$(date '+%Y-%m-%d_%H-%M-%S').log"
# exec > >(tee -a "$LOG_FILE") 2>&1

# conda create -n medreason-test python=3.10

# conda activate medreason-test
# pip install openai pandas lxml

API_KEY="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"
# python3 code/Generate_Reasoning_Path.py --input="./jsonl/new_reasoning_path.jsonl" --output="./jsonl/new_reasoning_path_1.jsonl" --API_KEY=$API_KEY

# python ./code/Generate_Initial_plan.py --input="./jsonl/new_reasoning_path_1.jsonl" --output="./json/Med_Reason_Plan_medqa.json"

# python ./code/Generate_Transient_Data.py --input_json_path="./json/Med_Reason_Plan_medqa.json" --output_json_path="./json/Med_Reason_medqa_10_1.json" --data_amount=10000 --last_json="./json/Med_Reason_medqa_3k.json" --API_KEY=$API_KEY

python ./code/Generate_Transient_Data.py --input_json_path="./json/Med_Reason_Final_10_3.json" --output_json_path="./json/Med_Reason_Final_10_1.json" --data_amount=15000 --API_KEY=$API_KEY

# Step 2: loop up to 5 times
for i in {1}
do
    echo "=== Iteration $i ==="

    # python ./code/check_plan_execution.py --input_json_path="./json/Med_Reason_Final_10_3.json" --output_json_path="./json/Med_Reason_Final_10_2.json"

    python ./code/check_conclusion.py --input_json_path="./json/Med_Reason_Final_10_1.json" --output_json_path="./json/Med_Reason_Final_10_3.json" --API_KEY=$API_KEY

    # python ./code/check_plan_execution.py --input_json_path="./json/Med_Reason_medqa_10_1.json" --output_json_path="./json/Med_Reason_medqa_10_2.json"

    # python ./code/check_conclusion.py --input_json_path="./json/Med_Reason_medqa_10_2.json" --output_json_path="./json/Med_Reason_medqa_10_3.json" --API_KEY=$API_KEY

    # Capture output of Generate_Transient_Data.py
    # python ./code/Generate_Transient_Data.py --input_json_path="./json/Med_Reason_medqa_10_3.json" --output_json_path="./json/Med_Reason_medqa_10_1.json"  --data_amount=1 --API_KEY=$API_KEY
    # output=$(python ./code/Generate_Transient_Data.py --input_json_path="./json/Med_Reason_Final_10_3.json" --output_json_path="./json/Med_Reason_Final_10_1.json"  --data_amount=3500)

    # echo "$output"

    # # If program prints ALL DATA ARE ELIGIBLE., stop loop
    # if echo "$output" | grep -q "ALL DATA ARE ELIGIBLE."; then
    #     echo "All data eligible, stopping loop."
    #     break
    # fi
done