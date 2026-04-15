from openai import OpenAI
import json
import re
import argparse

parser = argparse.ArgumentParser(description="Generate Reasoning Path")
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--API_KEY")
args = parser.parse_args()
input_json = args.input
output_json = args.output
api_key = args.API_KEY

API_KEY = api_key
client = OpenAI(api_key=API_KEY)
PRICE_INPUT = 0.0025 / 1000
PRICE_OUTPUT = 0.0100 / 1000

total_input_tokens = 0
total_output_tokens = 0
data = []

file_path = input_json
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            sss = json.loads(line)
            data.append(sss)

def validate_reasoning_chains(chains_text: str) -> bool:
    if chains_text == "**FAIL**":
        return False
    lines = [ln.strip() for ln in chains_text.splitlines() if ln.strip()]
    if not lines:
        return False

    expected_index = 1
    for i, line in enumerate(lines):
        match = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if not match:
            return False

        index, content = int(match.group(1)), match.group(2)
        if index != expected_index:
            return False
        expected_index += 1

        if "->" not in content:
            return False

        parts = [p.strip() for p in content.split("->")]
        if any(p == "" for p in parts):
            return False

    return True

def deduplicate_reasoning_chains(text_block):
    """
    输入: 包含多行 "ID: Chain" 的字符串
    输出: 去重并重新编号后的字符串
    """
    if not text_block:
        return ""

    lines = text_block.strip().split('\n')
    seen_content = set()
    unique_chains = []
    new_index = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            _, content = line.split(':', 1)
            content = content.strip()
        else:
            content = line

        if content not in seen_content:
            seen_content.add(content)
            unique_chains.append(f"{new_index}: {content}")
            new_index += 1

    return '\n'.join(unique_chains)

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

def change_format(filter_reasoning_path):
    global total_input_tokens, total_output_tokens
    judge = validate_reasoning_chains(filter_reasoning_path)
    if judge==False:
        messages2 = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning chain format correction assistant.\n"
                    "Your only job is to fix formatting issues in the reasoning chains, "
                    "without changing any of their textual content.\n\n"
                    "=== STRICT RULES ===\n"
                    "1. Do NOT change, rewrite, or paraphrase any part of a reasoning chain.\n"
                    "   - Keep every token, word, and symbol exactly as in the original text.\n"
                    "2. Only fix **formatting**, including:\n"
                    "   - Whitespace or indentation errors.\n"
                    "   - Non-sequential or missing indices.\n"
                    "   - Incorrect ':' separator formatting.\n"
                    "   - Missing or redundant line breaks.\n"
                    "3. Each reasoning chain must be formatted as '<index>: A->B->C->...'\n"
                    "   - Indices start from 1 and increase sequentially (1, 2, 3, ...).\n"
                    "   - Preserve the original order of the reasoning chains.\n"
                    "4. Do NOT modify or reorder reasoning chain contents.\n"
                    "5. Do NOT add, delete, merge, or split reasoning chains.\n"
                    "6. If a line cannot be fixed without altering its content, keep it unchanged.\n"
                    "7. If the input is completely empty or contains only whitespace, output an empty string (no text, no comment).\n"
                    "8. Output only the reasoning chains in corrected format — no explanations, no extra text."
                )
            },
            {
                "role": "user",
                "content": (
                    "Input reasoning chains:\n"
                    f"{filter_reasoning_path}\n\n"
                    "Output the same reasoning chains with only formatting corrected, keeping all text exactly identical.\n"
                    "If the input is empty, output nothing."
                )
            }
        ]

        filter_reasoning_path = run_llm(model="gpt-4o", temperature=0, messages=messages2)
        judge = validate_reasoning_chains(filter_reasoning_path)
        
    if judge==False:
        return ""
    return filter_reasoning_path

jsonl_path = input_json
count = 0
data = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        question, answer, original_reasoning = obj['question'], obj['answer'], obj['original_reasoning']

        messages1 = [
            {
                "role": "system",
                "content": """You are a strict reasoning chain filter.
                    Given a question, a list of candidate reasoning chains (original_reasoning), and the correct answer,
                    select only the reasoning chains that are directly relevant for deriving the answer from the question.

                    Filtering Rules (Follow All Exactly)
                    1) Relevance: Keep only chains that directly or critically contribute to deriving the answer from the question.
                    Discard any chain that is unrelated or unnecessary for reaching the answer.
                    2) Consistency: Remove chains that contradict the facts stated in the question or that lead to conclusions conflicting with the answer.
                    3) Duplicate Removal: If multiple chains are textually identical, keep only the first occurrence.
                    4) Order & Priority: Preserve the original order of appearance in original_reasoning.
                    If more than 10 chains remain, output the 10 most useful ones for deriving the answer (strongest/direct connection).
                    5) Text Integrity: Do not modify any retained reasoning chain text.
                    Each chain must remain exactly identical to its original text (from 'A->B->C->...' to the end).
                    Only reassign new indices starting from 1 in ascending order.
                    6) Empty Case: If no chains satisfy the rules, output nothing (no text, no comment).
                    """
            },
            {
                "role": "user",
                "content": f"""
                    Input:
                    * question:
                    {question}

                    * answer:
                    {answer}

                    * original_reasoning (each line formatted as "<index>: A->B->C->..."):
                    {original_reasoning}

                    Output:
                    Return only the filtered reasoning chains in this exact format and nothing else:
                    1: reasoning chain text (identical to original)
                    2: reasoning chain text (identical to original)
                    ...
                    (Up to 10 lines total.)
                    """
            }
        ]
        
        filter_reasoning_path = run_llm(model="gpt-4o", temperature=0, messages=messages1)
        if filter_reasoning_path == "**FAIL**":
            continue
        
        filter_reasoning_path = change_format(filter_reasoning_path)
        if filter_reasoning_path == "":
            continue

        messages3 = [
            {
                "role": "system",
                "content": (
                    "You are an expert in the medical domain.\n"
                    "Goal: Generate reasoning chains where the **Start Node** is strongly correlated with a key entity in the Question, "
                    "and the **End Node** is strongly correlated with the Answer entity.\n\n"
                    "STRONG PRIORITY ON USING PROVIDED PATHS:\n"
                    "- Maximize reuse of entities and links that appear in the provided Paths.\n"
                    "- Prefer chains composed entirely of nodes from the Paths.\n"
                    "- If you reuse nodes from the Paths, keep their strings EXACTLY as written (same casing, no edits).\n"
                    "- Select Start/End nodes from the provided Paths that have the highest semantic or clinical correlation to the Question/Answer context.\n\n"
                    "STRICT OUTPUT RULES:\n"
                    "1) Each line: '<index>: A->B->C->...'\n"
                    "   - <index> starts at 1 and increases by 1; exactly one space after the colon.\n"
                    "2) Use '->' as the ONLY delimiter with no spaces around it.\n"
                    "3) First node: Must have a STRONG CORRELATION (semantic or clinical) to a key entity in the Question (does NOT need to be an exact string match).\n"
                    "4) Final node: Must have a STRONG CORRELATION to the Answer entity (does NOT need to be the exact Answer string).\n"
                    "5) Each link A->B must be a medically valid causal/inferential relation.\n"
                    "6) No headers, no explanations, no extra text.\n"
                    "7) Do not rewrite/rename nodes taken from the Paths.\n"
                    "8) Output up to 6 chains; output nothing if no valid chain can be formed.\n"
                    "9) If multiple valid options exist, prefer chains that:\n"
                    "    (a) maximize coverage of nodes from the provided Paths,\n"
                    "    (b) require zero new nodes; if unavoidable, use at most one new medically sound bridge.\n"
                    "10) Avoid producing only a single reasoning chain unless only one medically valid path exists; "
                    "whenever possible, output two or more distinct valid reasoning chains."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    "Paths (filtered reasoning paths to reuse):\n"
                    f"{filter_reasoning_path}\n\n"
                    "Output:\n"
                    "1: A->B->...\n"
                    "2: A->B->...\n"
                    "..."
                )
            }
        ]

        new_reasoning_path = run_llm(model="gpt-4o", temperature=0.2, messages=messages3)
        if new_reasoning_path == "**FAIL**":
            continue

        messages4 = [
            {
                "role": "system",
                "content": (
                    "You are a medical-domain reasoning chain editor.\n"
                    "Edit policy (hard constraints):\n"
                    "- Identity-by-default: If a chain is already complete and logically sound, output it UNCHANGED.\n"
                    "- Only-if-incomplete: Modify a chain ONLY when it is logically incomplete between Question-entity and Answer-entity/synonym.\n"
                    "- Preserve-original-entities: Do NOT alter, delete, paraphrase, or reorder ANY existing entities or links.\n"
                    "- Insert-only-new-bridge: When a fix is necessary, INSERT the FEWEST possible concise medical entities as bridges;\n"
                    "  do not modify existing tokens. Prefer ≤2 new entity per chain.\n"
                    "- Medical validity: Each hop A->B must be a clinically valid causal/inferential relation.\n"
                    "- If uncertain whether a change is required, leave the chain UNCHANGED."
                )
            },
            {
                "role": "user",
                "content": (
                    "Requirements:\n"
                    "1) Only add new entities to chains that are logically incomplete; keep ALL original entities exactly as given (no edits, no reordering, no deletions).\n"
                    "2) Use '->' only; no spaces around it. Output one line per chain as '<index>: A->B->C->...'; indices start at 1, increment by 1, and have exactly one space after ':'.\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    "new_reasoning_path:\n"
                    f"{new_reasoning_path}\n\n"
                    "Output:\n"
                    "1: A->B->...\n"
                    "2: A->B->...\n"
                    "..."
                )
            }
        ]

        new_reasoning_path = run_llm(model="gpt-4o", temperature=0.2, messages=messages4)
        if new_reasoning_path == "**FAIL**":
            continue
        new_reasoning_path = deduplicate_reasoning_chains(new_reasoning_path)


        messages5 = [
            {
                "role": "system",
                "content": (
                    "You are a board-certified physician and medical reasoning evaluator. "
                    "Judge whether the entire new_reasoning_path is adequate to connect the Question to the Answer. "
                    "Only mark FAIL when the chains are clearly unrelated or medically incorrect."
                )
            },
            {
                "role": "user",
                "content": (
                    "PASS if and only if ALL of the following hold:\n"
                    "1) The chain(s) overall move in the correct direction and can plausibly derive the Answer from the Question.\n"
                    "2) Minor omissions or shorthand steps are acceptable as long as the overall path is understandable.\n\n"
                    "FAIL only if ANY of the following occur (strong threshold):\n"
                    "- The chain(s) are largely unrelated to the Question or to the Answer (topic mismatch, irrelevant entities), or\n"
                    "- One or more steps contain medically invalid/contradictory relations, or\n"
                    "- The overall direction is wrong or cannot plausibly reach the Answer at all.\n\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    "new_reasoning_path:\n"
                    f"{new_reasoning_path}\n\n"
                    "Output strictly as ONE of the following (single line):\n"
                    "PASS\n"
                    "or\n"
                    "FAIL, <concise reason>\n"
                )
            }
        ]

        validation = run_llm(model="gpt-4o", temperature=0, messages=messages5)
        if validation == "**FAIL**":
            continue
        # print(validation)
        if validation.lower().find('pass')!=-1:
            new_reasoning_path = change_format(new_reasoning_path)
            if new_reasoning_path == "":
                continue
            obj['new_reasoning_path'] = new_reasoning_path
            data.append(obj)
            count += 1
        else:
            repair_prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Medical Reasoning Refiner.\n"
                        "Task: Repair the 'Defective Chain' based on the 'Failure Reason'. Additionally, if the defect implies weak logic, add 1-2 NEW valid chains.\n\n"
                        "STRICT OUTPUT RULES:\n"
                        "1. **Format**: Output a list of chains separated by a **SINGLE newline** only. **NO blank lines** between chains. Format: '<ID>: Node A->Node B->...'\n"
                        "2. **Deduplication**: Ensure NO two chains in the output are identical (exact same node sequence).\n"
                        "3. **Syntax**: Use '->' as the only delimiter.\n"
                        "4. **Entity Constraint (CRITICAL)**: Each node MUST be a concise **Medical Entity** (noun or noun phrase, e.g., 'Ischemia', 'Chest Pain'). **DO NOT** use full sentences, verb phrases, or long descriptions (e.g., Avoid 'The patient feels pain' or 'causes heart failure').\n\n"
                        "OPERATIONS (Execute BOTH if applicable):\n"
                        "1. **REPAIR (Primary)**: \n"
                        "   - Fix the specific 'Failure Reason' in the original chain using **MINIMAL edits** (Gap Fill, Swap, Redirect).\n"
                        "   - Ensure modified nodes are converted to concise entities if they were originally sentences.\n"
                        "   - Preserve valid nodes. Maintain the original ID for the repaired version.\n"
                        "2. **AUGMENT (Secondary)**: \n"
                        "   - If the original chain was significantly flawed, ADD 1-2 NEW, concise, and valid reasoning chains.\n"
                        "   - **New Start**: Must be a meaningful clinical entity (noun) from the **Question**.\n"
                        "   - **New End**: Must be a meaningful clinical entity (noun) from the **Answer**.\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Answer: {answer}\n"
                        f"Defective Chain: {new_reasoning_path}\n"
                        f"Failure Reason: {validation}\n\n"
                        "Output (Repaired Chain + Optional New Chains):"
                    )
                }
            ]
            new_reasoning_path = run_llm(model="gpt-4o", temperature=0, messages=repair_prompt_messages)
            if new_reasoning_path == "**FAIL**":
                continue
            new_reasoning_path = change_format(new_reasoning_path)
            if new_reasoning_path == "":
                continue
            obj['new_reasoning_path'] = new_reasoning_path
            data.append(obj)
            count += 1

        print(obj['id'])
        print(question)
        print(answer)
        print(validation)
        print(filter_reasoning_path)

        print(new_reasoning_path)

        cost_total = total_input_tokens * PRICE_INPUT + total_output_tokens * PRICE_OUTPUT
        print(f"current API cost: {cost_total}")
        print("-------------------")

print(count)
with open(output_json, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")