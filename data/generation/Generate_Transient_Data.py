from openai import OpenAI
import json
import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Generate Transient Step Data")
parser.add_argument("--last_json", default="None")
parser.add_argument("--input_json_path", default="./MedVerse_Plan.json")
parser.add_argument("--output_json_path", default="./MedVerse_transient.json")
parser.add_argument("--data_amount")
parser.add_argument("--API_KEY")
args = parser.parse_args()
json_path = args.input_json_path
output_path = args.output_json_path
last_json_path = args.last_json
api_key = args.API_KEY
number = int(args.data_amount)

API_KEY = api_key
client = OpenAI(api_key=API_KEY)
PRICE_INPUT = 0.0025 / 1000
PRICE_OUTPUT = 0.0100 / 1000

with open(json_path, 'r') as f:
    data=json.load(f)
if last_json_path=='None':
    data1={}
else:
    with open(last_json_path, 'r') as f:
        data1=json.load(f)

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

cnt, eli_cnt = 0, 0
k_list, v_list = [], []
# number = 2
print(number)
for idx, item in data.items():
    if idx in data1:
        item=data1[idx]
    if cnt == number:
        break
    print(idx)
    cnt += 1
    # if cnt<=2000:
    #     # k_list.append(idx)
    #     # v_list.append(item)
    #     continue
    if "Eligible" not in item:
        item["Eligible"] = 0
    elif item["Eligible"] == 2:
        eli_cnt += 1
        k_list.append(idx)
        v_list.append(item)
        continue
    else:
        item["Eligible"] = 0
    # if item.get("Transient Execution Prompt", "<Execution>\n</Execution>") != "<Execution>\n</Execution>":
    #     eli_cnt += 1
    #     k_list.append(idx)
    #     v_list.append(item)
    #     continue
    # else:
    #     item["Eligible"] = 0
    goal = item['Goal']
    plan = item['Plan Prompt']

    executed = {}
    titles = {}
    multistep_reasoning=""
    step_pattern = re.compile(r"<Outline>\s*(\d+)\.\s*(.*?);\s*Dependency:\s*\[(.*?)\]\s*</Outline>")
    transient_step = 0
    transient_first = defaultdict(set)
    transient_second = defaultdict(int)
    transient_full = {}


    ##############################
    ### step by step reasoning (first prompt)
    for match in step_pattern.finditer(plan):
        step_number = int(match.group(1))
        title = match.group(2).strip()
        dependencies_raw = match.group(3).strip()
        dependencies = [int(dep) for dep in dependencies_raw.split(',')] if dependencies_raw else []
        titles[step_number] = title
        # print(step_number, title, dependencies)

        if dependencies == []:
            executed[step_number]='Step '+str(step_number)+': '+title
            continue 
        else:
            executed_step=""
            for prev_step in dependencies:
                executed_step+=executed[prev_step]+'\n'
            current_step='Step '+str(step_number)+': '+title
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert in the medical domain. Given a medical goal, a plan consisting of a list of steps with their dependencies and associated entity keywords, your task is to reason **step by step**, performing **CoT (Chain-of-Thought) reasoning for each step independently**, as if you are deriving the final goal without prior knowledge of the answer.

                        Your focus is to reason through **only one step at a time**. Each step’s reasoning must rely solely on:
                        * The **CoT results from the directly dependent previous steps**, and
                        * The **entity keywords specific to the current step**.

                        **Output only a single concise CoT paragraph focused strictly on factual reasoning with minimal inference.**

                        Do **not**:
                        * Explain the goal.
                        * Speculate about future steps.
                        * Mention dependencies or reference previous steps explicitly.
                        * Introduce extraneous information like definitions, purpose, or clinical relevance.
                        * **State functions, importance, or implications of anatomical structures.**

                        Just reason through the current step using **only necessary factual inference and brief, anatomy-based logic**, grounded in structural relationships.

                        **Guidelines:**
                        * Identify the **dependencies** of the current step from the plan.
                        * Use the **CoT results** of those dependencies as your prior knowledge.
                        * Combine this information with the **current step’s entity keywords** to form a **short and concise** reasoning paragraph.
                        * Ensure your output is only **one short paragraph**, clearly articulating **only what is necessary** to move forward.
                        * Do **not** mention the existence of predefined reasoning paths, provided answers, or assumptions about correctness.
                        * The reasoning should be derived solely from the **current context**.

                        **Requirements:**
                        * All entities mentioned in the current step and in its directly dependent executed steps MUST have their factual details explicitly and concretely included in the reasoning paragraph. 
                        * Do not omit or generalize these details; each entity must be described with its specific factual attributes as given in the inputs, which may include structural, anatomical, temporal, pathological, or disease-related characteristics.
                        * The reasoning will be considered incorrect if any entity detail is missing or only vaguely referred to.

                        **Return only the CoT paragraph. Do not include headings or labels.**"""
                },
                {
                    "role": "user",
                    "content": f"""**Input:**

                    1. **Goal** – The final objective to be achieved:
                    {goal}

                    2. **Plan** – A list of steps with associated goals and their dependencies:
                    {plan}

                    3. **Executed Steps** – CoT results from only those steps that the current step depends on:
                    {executed_step}

                    4. **Current Step** – The step you are currently reasoning about:
                    {current_step}

                    **Output:**
                    **CoT Paragraph:**
                    (Provide **only one short paragraph** of reasoning for the current step. Stick to **facts** and brief **logical inference**. Avoid elaboration. **Do not reference the answer or future steps.**)"""
                }
            ]

            raw_output = run_llm(model="gpt-4o", temperature=0, messages=messages)
            if raw_output == '**FAIL**':
                break
            
            ##############################
            ### refine reasoning process (second prompt)
            messages_ = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical domain expert. "
                        "You are given a paragraph describing anatomical structures. "
                        "Your task is to revise the paragraph by removing any sentence that mentions importance, purpose, usefulness, significance, "
                        "or general statements like 'this is essential' or 'understanding X helps'. "
                        "Only include structural or anatomical facts. "
                        "Do not explain, summarize, or mention reasoning. "
                        "Output should be a single paragraph."
                    )
                },
                {
                    "role": "user",
                    "content": f"Input:\n{raw_output}\n\nOutput:"
                }
            ]

            final_output = run_llm(model="gpt-4", temperature=0, messages=messages_)
            if final_output == '**FAIL**':
                break

            transient_step += 1
            current_transient = ""
            current_transient += "Transient Step "+str(transient_step)+": "
            for i, prev_step in enumerate(dependencies):
                if i:
                    current_transient += ', '
                current_transient += titles[prev_step]
                transient_first[transient_step].add(prev_step)
            current_transient += ' -> '+title+'\n'
            multistep_reasoning += current_transient
            transient_second[transient_step] = step_number
            transient_full[transient_step] = current_transient[:-1]
            # print(current_transient)

            executed[step_number]='Step '+str(step_number)+': '+final_output
            multistep_reasoning+=final_output+'\n'+'\n'
            # print(final_output)
            # print(messages)
    
    # print(multistep_reasoning)
    # print('-------------------')

    ##############################
    ### refine overall reasoning in two steps
    messages_dedup = [
        {
            "role": "system",
            "content": """You are an expert in extracting concise, non-redundant, factual reasoning from multi-step medical analyses.

            You will be given:
            - A **goal**, describing the final medical question to be answered.
            - A series of **step-wise reasoning outputs**, where each step includes:
            1. A label in the format: `Entity A -> Entity B`, and
            2. A detailed reasoning paragraph.

            Your task:
            - Eliminate any redundant content that has already appeared in previous steps.
            - Keep only factual details necessary for reasoning toward the goal.
            - Contain only medical facts (do not include purpose, significance, or usefulness).
            - Exclude general background or unrelated definitions.
            - Must retain the logical relationship between Entity A and Entity B.

            Instructions:
            - For each step, return the original step label followed by your revised concise reasoning.
            - Do not include explanations, meta-comments, or summaries.
            - Maintain the original order of steps.
            - Do not modify the goal.
            - Do not add overall conclusions — only revise each step individually."""
        },
        {
            "role": "user",
            "content": f"""Goal: {goal}

            Stepwise Reasoning:
            {multistep_reasoning}"""
        }
    ]
    pending_final = run_llm(model="gpt-4o", temperature=0, messages=messages_dedup)
    if pending_final == '**FAIL**':
        break

    total_final = pending_final
    messages_find_missing = [
        {
            "role": "system",
            "content": """You are an expert in analyzing medical reasoning completeness.

            You will be given:
            - A goal (answer target).
            - A series of step-wise reasoning outputs (each has a label `Entity A -> Entity B` and a reasoning paragraph).

            Task:
            1) Identify the single most important entity from the goal that is NOT explicitly present in the stepwise reasoning.  
            - “Most important” = the single missing entity whose absence most seriously undermines the ability to reach or justify the goal’s answer.  
            2) Choose exactly ONE transient step that is the most appropriate place to add this missing entity.  
            - The appropriate step is usually **not one of the very first transient steps**, but a later step where the reasoning structure better supports the insertion.  
            - “Most appropriate” = the one point in the reasoning where inserting the entity makes the overall logic most reasonable and coherent for reaching the final goal.  
            - The chosen step must involve entities from its outline (Entity A -> Entity B) that are directly related to the missing entity.  
            3) Do not propose duplicates or multiple steps.

            Output format:
            - If a missing entity exists, return in this format:

            Detail: <the single most important missing entity>  
            Insert_into_step: Transient Step <Step Number>: <Step Label>

            - If no entity is missing, return exactly:

            No missing detail.
            """
        },
        {
            "role": "user",
            "content": f"""Goal:
            {goal}

            Stepwise Reasoning:
            {pending_final}"""
        }
    ]
    missing_detail_output = run_llm(model="gpt-4o", temperature=0, messages=messages_find_missing)
    if missing_detail_output == '**FAIL**':
        break

    if "No missing detail" in missing_detail_output:
        print("No missing detail, returning original reasoning.")
        pass
    else:
        print(missing_detail_output)
        messages_insert_missing = [
            {
                "role": "system",
                "content": """You are an expert in revising medical reasoning to ensure completeness.

                You will be given:
                - A goal.
                - The current stepwise reasoning (ordered steps with labels and paragraphs).
                - ONE missing goal entity (if any) and the single transient step label where it should be inserted.

                Task:
                - If a missing entity is provided, you MUST add this entity so that it is explicitly represented in the reasoning.
                - Insert it into the indicated step’s reasoning paragraph in a way that is aligned with that step’s outline (Entity A -> Entity B). 
                Do not merely append the entity; instead, COMPOSE ONE NEW SENTENCE that explicitly includes the missing entity and states its relationship to the step’s outline entities (A → B), then insert this single sentence at a natural position within that step.
                - Do not add the same entity to other steps.
                - If the input says "No missing detail.", return the reasoning unchanged.
                - Preserve all existing content and step order.

                Output:
                - For each step, return the original step label followed by the revised reasoning.
                - Do not include explanations, meta-comments, or summaries.
                - Do not modify the goal.
                - Do not add overall conclusions — only revise each step individually."""
            },
            {
                "role": "user",
                "content": f"""Goal:
                {goal}

                Stepwise Reasoning (before update):
                {pending_final}

                Missing Detail:
                {missing_detail_output}"""
            }
        ]
        total_final = run_llm(model="gpt-4o", temperature=0, messages=messages_insert_missing)
        if total_final == '**FAIL**':
            break
    # print(pending_final)
    # print('-----------------------')

    # print(pending_final)
    print(total_final)
    print('-------------------')
    item['Pending step'] = pending_final
    item['Final Step'] = total_final
    # print(total_final)

    ##############################
    ### transient graph construction
    edges = []
    dependencies_transient = defaultdict(list)
    for trans1 in transient_first:
        for trans2 in  transient_second:
            if transient_second[trans2] in transient_first[trans1]:
                edges.append((trans2, trans1))
                # print(str(trans2)+' -> '+str(trans1))
                dependencies_transient[trans1].append(trans2)
    adj = defaultdict(list)
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
        outdeg[u] += 1
    for x in range(transient_step):
        indeg.setdefault(x+1, 0)
        outdeg.setdefault(x+1, 0)
        adj.setdefault(x+1, [])
    sources = [x+1 for x in range(transient_step) if indeg[x+1] == 0]
    sinks = {x+1 for x in range(transient_step) if outdeg[x+1] == 0}
    res = []

    def dfs(u, path, seen):
        if u in sinks:
            real_path = []
            for node in path:
                real_path.append('('+transient_full[node]+')')

            res.append(real_path)
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            path.append(v)
            dfs(v, path, seen)
            path.pop()
            seen.remove(v)

    for s in sources:
        dfs(s, [s], {s})
    path_output = ""
    for i, path in enumerate(res):
        path_output += str(i+1)+'. '
        for i1, path_step in enumerate(path):
            if i1:
                path_output += " -> "
            path_output += path_step 
        path_output += '\n'
    # print(path_output)
    item['Transient Path'] = path_output

    ##############################
    ### generate transient plan
    outlines = []
    for x in range(transient_step):
        node = x+1
        entity = transient_full[node]
        dep = sorted([d for d in dependencies_transient[node]]) if node in dependencies_transient else []
        if node not in dependencies_transient:
            outlines.append(f"<Outline> {entity}; Dependency: {dep} </Outline>")
        else:
            outlines.append(f"<Outline> {entity}; Dependency: {dep} </Outline>")
    transient_plan = "<Plan>\n" + "\n".join(outlines) + "\n</Plan>"
    item['Transient Plan Prompt'] = transient_plan
    # print(transient_plan)

    ##############################
    ### generate transient execution
    def to_execution_block_from_plain(text: str) -> str:
        """
        从纯文本中提取以 'Transient Step N:' 开头的段落，生成标准的<Execution><Step>...</Step>...</Execution> 结构。
        """
        header = r'(?:\s*\d+\.\s+)?(?:Transient Step\s*\d+\s*:[^\n]*|[^\n]*?->[^\n]*)'
        pattern = re.compile(
            rf'(?m)(?s)'           # 多行 + DOTALL
            rf'^\s*(?P<head>{header})\s*$'   # 标题行（单行）
            rf'(?P<body>.*?)'                 # 正文（可跨行）
            rf'(?=^\s*(?:{header})\s*$|\Z)'   # 截止到下一条标题或文本结束
        )

        steps = []
        for m in pattern.finditer(text):
            head = m.group(1).strip()
            body = m.group(2).strip('\n ')
            content = head if not body else f"{head}\n{body}"
            steps.append(content)

        if not steps:
            return "<Execution>\n</Execution>"

        parts = ["<Execution>"]
        for s in steps:
            parts.append(f"<Step> {s}\n</Step>")
        parts.append("</Execution>")
        return "\n".join(parts)

    transient_execution = to_execution_block_from_plain(total_final)
    item['Transient Execution Prompt'] = transient_execution
    print(transient_execution)
    # if transient_execution == "<Execution>\n</Execution>":
    #     break

    # ##############################
    # ### judgement of generated prompt
    # messages2 = [
    #     {
    #         "role": "system",
    #         "content": """You are an evaluator of reasoning completeness. 
    # Your task is to check whether the reasoning process fully covers the goal.

    # Rules:
    # - Factual details include entities, attributes, relationships, time references, pathological features, or disease-related information.
    # - If every factual detail in the Goal is explicitly mentioned in the Reasoning Process, output "Complete".
    # - If any detail is missing, output "Incomplete".
    # - Do not output anything else."""
    #     },
    #     {
    #         "role": "user",
    #         "content": f"""Reasoning Process:
    # {reasoning_process}

    # Goal:
    # {goal}"""
    #     }
    # ]
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=messages2, 
    #     temperature=0
    # )
    # judge = response.choices[0].message.content
    # print(judge)

    ##############################
    ### generate conclusion
    question = item['Question']
    option =  item['Options']
    messages3 = [
        {
            "role": "system",
            "content": (
                """
                You are an expert in medical reasoning.
                You will be given three inputs:
                1) Reasoning Paragraphs (a sequence of 'Transient Step N: ...' logical steps),
                2) Question (a medical multiple-choice question),
                3) Options (possible answers).
                Your task:
                - Use ONLY the Reasoning Paragraphs to determine the correct answer.
                - First, output the final answer to the Question.
                - Then, output one concise paragraph explaining why this is the correct answer, referencing the relevant steps.
                - Do not add external knowledge.
                Output format must be:
                Explanation: <one paragraph justification>
                Answer: <the correct option>
                """
            )
        },
        {
            "role": "user",
            "content": (
                f"""
                Reasoning Paragraphs:{total_final}
                Question: {question}
                Options: {option}
                """
            )
        }
    ]
    conclusion = run_llm(model="gpt-4o", temperature=0, messages=messages3)
    if conclusion == '**FAIL**':
        break
    item['Conclusion'] = conclusion
    if int(idx)%10==0:
        cost_total = total_input_tokens * PRICE_INPUT + total_output_tokens * PRICE_OUTPUT
        print(f"current API cost: {cost_total}")
    # print(item['Goal'])
    # print(conclusion)

    k_list.append(idx)
    v_list.append(item)

result={k:v for k,v in zip(k_list,v_list)}
with open(output_path,'w') as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

if eli_cnt == number:
    print("ALL DATA ARE ELIGIBLE.")