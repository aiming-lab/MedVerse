from openai import OpenAI
import os
import time
import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import random
import logging

# Global variable to hold API cost across multiple LLM calls
api_total_cost = 0.0

_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")

clients = {
    "gpt-5.2": {
        'api_key': _api_key,
        'name': 'gpt-5.2',
        'input_price': 1.75 / 10 ** 6,
        'output_price': 14.0 / 10 ** 6,
    }
}


def init_logger(name=''):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs('logs', exist_ok=True)
    handler = logging.FileHandler('logs/{name}-{time}.log'.format(name=name, time=time.strftime("%Y%m%d-%H%M%S")))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_json_from_generated_text(text):
    start = text.find("{")
    end = text.rfind("}")
    json_str = text[start:end+1]
    return json.loads(json_str)


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h.lower(), t.lower(), relation=r.lower().strip())
    return G


def get_topk_similar_entities(entity, knowledge_graph, emb_model, nodeemb_dict, k=5, filter_threshold=0.8):
    entity_type = entity["type"]
    node_entities_with_type = knowledge_graph.query('x_type=="{}"'.format(entity_type))['x_name'].unique()
    embeddings_for_node_entities = nodeemb_dict[entity_type]
    entity_embedding = emb_model.encode(entity["name"])
    similarity = emb_model.similarity(entity_embedding, embeddings_for_node_entities)
    val, idx = torch.topk(similarity, k)
    topk_similarity = similarity[0][idx]
    top1_similarity = topk_similarity[0][0].item()
    idx = idx[similarity[0][idx] > filter_threshold]
    if len(idx) == 0:
        return [], top1_similarity
    elif len(idx) == 1:
        return [node_entities_with_type[idx[0]]], top1_similarity
    else:
        return node_entities_with_type[idx].tolist(), top1_similarity


def find_all_path_KG(question_entities, result_entities, G):
    path_all = []
    for q_entity in question_entities:
        for a_entity in result_entities:
            path_all += list(nx.all_shortest_paths(G, q_entity.lower(), a_entity.lower()))
    return path_all


def generate_node_embeddings(knowledge_graph_path='/path/to/primeKG.csv', emb_model_name='abhinand/MedEmbed-large-v0.1'):
    knowledge_graph = pd.read_csv(knowledge_graph_path, low_memory=False)
    emb_model = SentenceTransformer(emb_model_name).to('cuda')
    types = knowledge_graph['x_type'].unique()
    nodeemb_dict = {}
    for t in types:
        print("generating embeddings for type: ", t)
        entities_in_types = knowledge_graph.query('x_type=="{}"'.format(t))['x_name'].unique()
        type_embeddings = emb_model.encode(list(entities_in_types))
        nodeemb_dict[t] = type_embeddings
    torch.save(nodeemb_dict, 'node_embeddings.pt')


def compute_usage(response, engine):
    usage = response.usage.to_dict()
    input_tokens = usage["prompt_tokens"]
    reasoning = 0 if "completion_tokens_details" not in usage else usage["completion_tokens_details"]["reasoning_tokens"]
    output_tokens = usage["completion_tokens"] - reasoning
    cost = {
        "input": input_tokens * clients[engine]['input_price'],
        "output": output_tokens * clients[engine]['output_price'],
    }
    cost["total"] = sum(cost.values())
    return cost


def run_llm(prompt, temperature=0.0, max_tokens=3000, engine="gpt-5.2", max_attempt=10):
    global api_total_cost
    client = OpenAI(api_key=clients[engine]['api_key'])
    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt},
    ]

    flag = 0
    while flag == 0 and max_attempt > 0:
        max_attempt -= 1
        try:
            response = client.chat.completions.create(
                model=clients[engine]['name'],
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                frequency_penalty=0)
            result = response.choices[0].message.content
            api_total_cost += compute_usage(response, engine)["total"]
            flag = 1
        except Exception as e:
            print(e)
            result = "openai error, retry"
            time.sleep(2)
    return result


def coarse_entity_extraction(text, temperature=0.0, max_tokens=3000, engine="gpt-5.2"):
    Extract_prompt = """ You are a helpful, pattern-following medical assistant.
Given the text in a medical or biomedical context, precisely extract all entities from the text.

### Output Format
Strictly follow the JSON structure below.
The type of each entity MUST STRICTLY BELONG to one type from:
1. gene/protein
2. drug
3. effect/phenotype
4. disease
5. biological_process
6. molecular_function
7. cellular_component
8. exposure
9. pathway
10. anatomy

```json
{{
"Entity": [
    {{"id": "1", "type": "some_type", "name": "entity_name"}},
    {{"id": "2", "type": "some_type", "name": "entity_name"}},
]
}}
```

### Input
text:
{text}

output:
"""
    prompt = Extract_prompt.format(text=text)
    return run_llm(prompt, temperature, max_tokens, engine)


def most_correlated_enetity_selection(question, query_entity, similar_entities, temperature=0.0, max_tokens=3000, engine="gpt-5.2"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
    Given a medical question and corresponding answer, an query entity which is extracted from the question, and a list of similar entities.
    Select ONE most correlated entity from the list of similar entities based on the question and query entity.
    SELECTED ENTITY MUST BE IN THE SIMILAR ENTITIES.
    IF there is not suitable entity in the similar entities, directly return the NONE.

    ### Output Format
    Strictly follow the JSON structure below:
    ```json
    {{
        "selected_entity": {{
            "name": "selected_entity_name",
            "id": a int number, the index of the selected entity in the similar entities list, from 0 to N-1
            "reason": "reason for choosing this entity"
        }}
    }}
    ```

    if there is no suitable entity, return:
    ```json
    {{
        "selected_entity": {{
            "name": "NONE",
            "id": "NONE",
            "reason": "reason for not choosing any entity"
        }}
    }}
    ```

    ### Input:
    question: {question}
    query entity: {query_entity}
    similar entities: {similar_entities}

    output:
    """
    similar_entities_str = ', '.join("{}.{}".format(idx, ent) for idx, ent in enumerate(similar_entities))
    prompt = Reformat_prompt.format(question=question, query_entity=query_entity, similar_entities=similar_entities_str)
    return run_llm(prompt, temperature, max_tokens, engine)


def QA_reformat_based_on_entity(question, answer, entity_list_text, temperature=0.0, max_tokens=5000, engine="gpt-5.2"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
Given a medical question and answer, and all a list of entities.
You need to reformat the question and answer into a pair of description and conclusion.

MUST MAKE SURE the conclusion and description paragraphs contain the entities in the entity list.
You can reallocate information from the question to the description and conclusion paragraphs, to make sure the entities in the entity list are included in the description and conclusion paragraphs.
However, you CAN NOT ADD ANY INFORMATION that is not in the question or answer.

### Output Format
Strictly follow the JSON structure below.

```json
{{
"description": {{
    "text" : "The description of the medical question.",
    "entities": [list of entities in the description, should not be empty]
    }},
"conclusion": {{
    "text" : "The conclusion of the medical question.",
    "entities": [list of entities in the conclusion, should not be empty]
    }}
}}
```

### Input
question:
{question}

answer:
{answer}

entity list:
{entity_list_text}

output:
"""
    prompt = Reformat_prompt.format(question=question, answer=answer, entity_list_text=entity_list_text)
    return run_llm(prompt, temperature, max_tokens, engine)


def QA_reformat_with_entity_extraction(question, answer, knowledge_graph, emb_model, nodeemb_dict,
                                        temperature=0.0, max_tokens=5000, engine="gpt-5.2"):
    QA_text = 'Question: {question}\nAnswer: {answer}'.format(question=question, answer=answer)
    all_entities = coarse_entity_extraction(QA_text, temperature, max_tokens, engine)
    all_entities = get_json_from_generated_text(all_entities)
    type_set = set(knowledge_graph['x_type'].unique())

    result_entities = []
    for entity in all_entities["Entity"]:
        if entity["type"] not in type_set:
            continue
        similar_entities, top1_similarity = get_topk_similar_entities(
            entity, knowledge_graph, emb_model, nodeemb_dict, k=10, filter_threshold=0.7)
        if not similar_entities:
            continue
        selected_entity = None
        for ent in similar_entities:
            if entity["name"].lower() == ent.lower():
                selected_entity = {"name": ent, "id": str(similar_entities.index(ent))}
                break
        if top1_similarity > 0.85 and selected_entity is None:
            selected_entity = {"name": similar_entities[0], "id": "0"}
        if selected_entity is None:
            selected_entity = most_correlated_enetity_selection(QA_text, entity["name"], similar_entities)
            selected_entity = get_json_from_generated_text(selected_entity)["selected_entity"]
        if selected_entity["name"] != "NONE":
            result_entities.append(similar_entities[int(selected_entity["id"])])

    result_entities = list(set(result_entities))
    entities_text = '\n'.join(['{}.{}'.format(idx+1, ent) for idx, ent in enumerate(result_entities)])
    return QA_reformat_based_on_entity(question, answer, entities_text, temperature, max_tokens, engine)


def path_sampling(path_all, question, answer, topK_reasoning_paths, max_path_number_per_group=50,
                  engine="gpt-5.2", logger=None):
    path_groups = {}
    for path in path_all:
        if len(path) < 2:
            continue
        path_key = (path[0], path[-1])
        if path_key not in path_groups:
            path_groups[path_key] = []
        path_groups[path_key].append(path)

    sampled_paths = []
    for path_key in path_groups:
        if logger is not None:
            logger.info(f"Sampling for Path group: {path_key}")
        if len(path_groups[path_key]) > max_path_number_per_group:
            path_groups[path_key] = random.sample(path_groups[path_key], max_path_number_per_group)
        text_for_group_paths = '\n'.join([str(idx+1) + ':' + '->'.join(p)
                                          for idx, p in enumerate(path_groups[path_key])])
        result_for_group = most_correlated_path_selection(question, text_for_group_paths, answer,
                                                          topK=topK_reasoning_paths, engine=engine)
        for path_i in result_for_group["Paths"]:
            sampled_paths.append(path_i["path"].split('->'))
    return sampled_paths


def most_correlated_path_selection(question, paths_text, answer, topK=2, temperature=0.0,
                                    max_tokens=4000, engine="gpt-5.2"):
    Reformat_prompt = """ You are a helpful, pattern-following medical assistant.
    Given a medical question and possible relation paths that link to the answer.
    Select up to {topK} most correlated entity from the relation paths list based on the question and the answer.
    If total number of paths is less than {topK}, select all of them.

    ### Output Format
    Strictly follow the JSON structure below.
    ```json
    {{
    "Paths": [
        {{"ranking": "1", "path": "sample_path_1", "reason": "reason for choosing this path"}},
        .....
    ]
    }}
    ```

    ### Input:
    question: {question}
    answer: {answer}
    paths: {paths}

    output:
    """
    prompt = Reformat_prompt.format(question=question, answer=answer, paths=paths_text, topK=topK)
    result = run_llm(prompt, temperature, max_tokens, engine)
    return get_json_from_generated_text(result)


def llm_generate_answer_with_reasoning(question, options, reasoning, engine='gpt-5.2'):
    prompt = f"""
You are an expert in the medical domain. You need to answer the following question based on the provided reasoning.
YOU MUST USE THE PROVIDED REASONING TO ANSWER THE QUESTION.
If the answer choices are provided, please choose ONE answer from the answer choices.

Question:
{question}
{options}
Reasoning:
{reasoning}
"""
    return run_llm(prompt, engine=engine)


def llm_judge_answer(llm_output, answer, engine='gpt-5.2'):
    prompt = f"""
You are an expert in the medical domain. Given a correct answer, and the answer from medical student.
You need to judge whether the answer from medical student is correct, by comparing the answer from medical student with the correct answer.
Your response must be 'True' or 'False'.
If the answer is correct, please respond with 'True'.
If the answer is wrong, please respond with 'False'.
Correct answer:
{answer}
Answer from medical student:
{llm_output}
"""
    return run_llm(prompt, engine=engine)
