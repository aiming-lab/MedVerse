import pandas as pd
import re
from collections import defaultdict, deque
import json
import argparse

parser = argparse.ArgumentParser(description="Generate Initial Path")
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()
input_json = args.input
output_json = args.output

def extract_section(text, start, end):
    pattern = rf"\*{{0,}}{start}:?\*{{0,}}(.*?)(\*{{0,}}{end}:?\*{{0,}})"
    match = re.search(pattern, text, re.S | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# df = pd.read_json("hf://datasets/UCSC-VLAA/MedReason/ours_quality_33000.jsonl", lines=True)
df = pd.read_json(input_json, lines=True)

k_list, v_list = [], []
for index in range(len(df)):
    goal = df.loc[index, 'question']+" "+df.loc[index, 'answer']
    option = df.loc[index, 'options']
    question = df.loc[index, 'question']
    text = df.loc[index, 'new_reasoning_path']
    # text=extract_section(df.loc[index, 'reasoning'], "Finding reasoning paths", "Reasoning Process")
    # if text.find('->')==-1:
    #     # print(df.loc[index, 'reasoning'])
    #     continue
    # # print(text)
    paths = []
    for line in text.splitlines():
        line = re.sub(r"^\d+[.:]\s*", "", line)
        entities = [e.strip() for e in line.split("->")]
        paths.append(entities)

    entity_to_id = {}
    id_to_entity = {}
    counter = 1
    for path in paths:
        for entity in path:
            if entity not in entity_to_id:
                entity_to_id[entity] = counter
                id_to_entity[counter] = entity
                counter += 1
    
    dependencies = defaultdict(set)
    graph = defaultdict(set)
    indegree = defaultdict(int)

    for path in paths:
        for i in range(1, len(path)):
            prev_entity = path[i-1]
            curr_entity = path[i]
            prev_id = entity_to_id[prev_entity]
            curr_id = entity_to_id[curr_entity]
            if curr_id not in dependencies:
                dependencies[curr_id] = set()
            dependencies[curr_id].add(prev_id)
            if curr_id not in graph[prev_id]:
                graph[prev_id].add(curr_id)
                indegree[curr_id] += 1

    for idx in entity_to_id.values():
        indegree.setdefault(idx, 0)

    queue = deque([node for node, deg in indegree.items() if deg == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    idx_to_seq = {idx: i for i, idx in enumerate(topo_order, 1)}
    outlines = []
    for i, idx in enumerate(topo_order, 1):
        entity = id_to_entity[idx]
        dep = sorted([idx_to_seq[d] for d in dependencies[idx]]) if idx in dependencies else []
        if idx not in dependencies:
            outlines.append(f"<Outline> {i}. {entity}; Dependency: {dep} </Outline>")
        else:
            outlines.append(f"<Outline> {i}. {entity}; Dependency: {dep} </Outline>")

    plan = "<Plan>\n" + "\n".join(outlines) + "\n</Plan>"

    item={}
    item['Plan Prompt'] = plan
    item['Original Reason Path'] = text
    item['Goal'] = goal
    item['Question'] = question
    item['Options'] = option
    k_list.append(index)
    v_list.append(item)
    
    # print(plan)
    # print("------------------------")


result={k:v for k,v in zip(k_list,v_list)}
with open(output_json,'w') as file:
    json.dump(result, file, indent=2, ensure_ascii=False)
