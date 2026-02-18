from openai import OpenAI
import json
import argparse

parser = argparse.ArgumentParser(description="Check Eligibility of Conclusion")
parser.add_argument("--input_json_path", default="./Med_Reason_Final_10_2.json")
parser.add_argument("--output_json_path", default="./Med_Reason_Final_10_3.json")
parser.add_argument("--API_KEY")
args = parser.parse_args()
json_path = args.input_json_path
output_path = args.output_json_path
api_key = args.API_KEY

API_KEY = api_key
client = OpenAI(api_key=API_KEY)
total_input_tokens, total_output_tokens = 0, 0

def run_llm(model="gpt-4o", temperature=0, messages=""):
    global total_input_tokens, total_output_tokens
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages, 
            temperature=0
            )
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    except Exception as e:
        return '**FAIL**'
    return response.choices[0].message.content

with open(json_path, 'r') as f:
    data=json.load(f)

k_list, v_list = [], []
for idx, item in data.items():
    if "Eligible" not in item:
        item["Eligible"] = 0
    elif item["Eligible"] == 2:# or item["Eligible"] == 0:
        k_list.append(idx)
        v_list.append(item)
        continue

    goal = item['Goal']
    question = item['Question']
    options = item['Options']
    conclusion = item['Conclusion']
    reasoning_steps = item['Final Step']
    messages_check_conclusion = [
        {
            "role": "system",
            "content": """You are an expert evaluator of medical reasoning consistency.

            You will be given:
            - A **goal**, describing the correct answer to the medical question.
            - A **question** and its **options**.
            - A **conclusion**, which includes a final answer and an explanation.
            - A set of **reasoning steps**, representing the chain of thought used to derive the answer.

            Your tasks:
            1. Verify whether the conclusion's final answer matches the correct answer specified in the goal.
            2. Verify whether the explanation in the conclusion is logically consistent with the reasoning steps:
            - The explanation must be directly derivable from the given reasoning steps.
            - It must not rely on external facts or background knowledge not present in the reasoning steps.

            **Output requirements:**
            - Output "Consistent" if (a) the conclusion's answer matches the goal's correct answer AND (b) the explanation is logically consistent with the reasoning steps.
            - Output "Inconsistent" otherwise.
            - Do not output anything else."""
        },
        {
            "role": "user",
            "content": f"""Goal (correct answer): {goal}

            Question: {question}

            Options: {options}

            Conclusion (explanation + answer): {conclusion}

            Reasoning Steps:
            {reasoning_steps}"""
        }
    ]

    answer = run_llm(model="gpt-4o", temperature=0, messages=messages_check_conclusion)
    print(answer)
    if answer == '**FAIL**':
        break

    if answer.find("Inconsistent") != -1:
        item["Eligible"] = 0
        print(f"Data {idx} Validation failed: Conclusion is inconsistent.")
    else:
        item["Eligible"] = 2
        print(f"Data {idx} Validation passed: Conclusion is consistent.")
    # print(goal)
    # print(conclusion)

    k_list.append(idx)
    v_list.append(item)

result={k:v for k,v in zip(k_list,v_list)}
with open(output_path,'w') as file:
    json.dump(result, file, indent=2, ensure_ascii=False)