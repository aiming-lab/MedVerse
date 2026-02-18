import json
import re
import argparse
import os
from openai import OpenAI
from datasets import load_dataset

MODEL_NAME = "gpt-5.1" 
TEMPERATURE_GENERATE = 0.4
TEMPERATURE_STRICT = 0.0

PRICE_INPUT = 0.0025 / 1000
PRICE_OUTPUT = 0.0100 / 1000

parser = argparse.ArgumentParser(description="Generate Medical Reasoning Chains from Scratch")
parser.add_argument("--output", default="/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/MedXperQA_reasoning_path.jsonl", help="Output JSONL file path")
parser.add_argument("--API_KEY", default="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", help="OpenAI API Key")

args = parser.parse_args()

client = OpenAI(api_key=args.API_KEY)

total_input_tokens = 0
total_output_tokens = 0

# ================= 核心工具函数 =================

def run_llm(model, temperature, messages):
    global total_input_tokens, total_output_tokens
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        usage = response.usage
        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return '**FAIL**'

def validate_reasoning_chains(chains_text: str) -> bool:
    """验证格式: '1: A->B->C'"""
    if chains_text == "**FAIL**" or not chains_text.strip():
        return False
    lines = [ln.strip() for ln in chains_text.splitlines() if ln.strip()]
    if not lines:
        return False
    expected_index = 1
    for line in lines:
        match = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if not match: return False
        index, content = int(match.group(1)), match.group(2)
        if index != expected_index: return False
        expected_index += 1
        if "->" not in content: return False
        if any(p.strip() == "" for p in content.split("->")): return False
    return True

def deduplicate_reasoning_chains(text_block):
    """去重并重新编号"""
    if not text_block or text_block == "**FAIL**": return ""
    lines = text_block.strip().split('\n')
    seen_content = set()
    unique_chains = []
    new_index = 1
    for line in lines:
        line = line.strip()
        if not line: continue
        if ':' in line: _, content = line.split(':', 1)
        else: content = line
        content = content.strip()
        if content not in seen_content:
            seen_content.add(content)
            unique_chains.append(f"{new_index}: {content}")
            new_index += 1
    return '\n'.join(unique_chains)

def change_format(reasoning_text):
    """格式修正 Agent"""
    if validate_reasoning_chains(reasoning_text): return reasoning_text
    messages = [
        {"role": "system", "content": "Fix formatting to '<index>: Node A->Node B'. Keep text identical. Output only corrected chains."},
        {"role": "user", "content": f"Input:\n{reasoning_text}"}
    ]
    corrected = run_llm(MODEL_NAME, TEMPERATURE_STRICT, messages)
    return corrected if validate_reasoning_chains(corrected) else ""

# ================= 主处理逻辑 =================

def process_line(obj):
    question = obj.get('question', '')
    answer = obj.get('answer', '')

    # ---------------- Step 1: Generate from Scratch (无输入链，完全生成) ----------------
    msgs_gen_step1 = [
        {
            "role": "system",
            "content": (
                "You are an expert Medical Reasoning Generator.\n"
                "Goal: Brainstorm 3-7 distinct, scientifically valid reasoning pathways connecting the Question to the Answer.\n"
                "Strategy: Explore different medical angles (e.g., Pathophysiology, Anatomy, Pharmacology, Diagnostic Criteria) to maximize diversity.\n\n"
                "STRICT RULES:\n"
                "1. **Anchors (Critical)**: \n"
                "   - **Start Node**: Must be a specific key clinical entity (Symptom, Lab Result, Drug, Condition) extracted directly from the **Question**.\n"
                "   - **End Node**: Must be the specific key clinical entity from the **Answer**.\n"
                "2. **Format**: Output strictly as a numbered list using '->' as the delimiter.\n"
                "3. **Logic**: Ensure each step (A->B) represents a direct causal or inferential link. Avoid huge logical leaps.\n"
                "4. **Diversity**: Do NOT just repeat the same path with slightly different synonyms. Look for alternative mechanisms or intermediate steps.\n"
                "5. **Clean Output**: Output ONLY the reasoning chains. No introductory text, no explanations, no markdown notes."
            )
        },
        { 
            "role": "user", 
            "content": f"Question: {question}\nAnswer: {answer}\n\nOutput:" 
        }
    ]
    raw_chains = run_llm(MODEL_NAME, TEMPERATURE_GENERATE, msgs_gen_step1)
    if raw_chains == "**FAIL**": return None
    # print(raw_chains)
    msgs_gen_step2 = [
        {
            "role": "system",
            "content": (
                "You are a Medical Knowledge Graph Engineer.\n"
                "Task: Convert raw reasoning chains into a **Standardized, Dense Medical DAG (Directed Acyclic Graph)**.\n\n"
                "### CORE OBJECTIVES ###\n"
                "1. **Standardize Entities**: Convert descriptive text into concise, standard Medical Terms (Noun Phrases).\n"
                "   - *Raw*: 'Patient has low blood sugar levels' -> *Std*: 'Hypoglycemia'\n"
                "   - *Raw*: 'BP 160/95' -> *Std*: 'Hypertension'\n"
                "2. **Aggressive Convergence (The 'Zipper' Method)**:\n"
                "   - Identify shared intermediate states (e.g., 'Ischemia', 'Inflammation') across the raw chains.\n"
                "   - **FORCE** all paths to route through these shared nodes using the **EXACT SAME STRING**.\n"
                "3. **No Daisy-Chaining (Merge Connected Segments)**:\n"
                "   - **Forbidden Pattern**: Do NOT output split segments where the **End Node** of one chain is the **Start Node** of another (e.g., '1. A->B', '2. B->C').\n"
                "   - **Requirement**: If Chain 1 ends where Chain 2 starts, you MUST **MERGE** them into a single continuous line (e.g., '1. A->B->C').\n"
                "### OUTPUT FORMAT ###\n"
                "1. Output the optimized paths using '->' as delimiter.\n"
                "2. **Constraint**: Output strictly **NO MORE THAN 7** unique chains.\n"
                "3. Format:\n"
                "1: Term A -> SharedNode X -> Term B\n"
                "2: Term C -> SharedNode X -> Term D\n"
                "..."
            )
        },
        { 
            "role": "user", 
            "content": f"Context:\nQ: {question}\nA: {answer}\n\nRaw Brainstormed Chains:\n{raw_chains}\n\nOutput Optimized Dense Chains:" 
        }
    ]
    
    generated_path = run_llm(MODEL_NAME, TEMPERATURE_GENERATE, msgs_gen_step2)
    if generated_path == "**FAIL**": return None
    
    generated_path = change_format(generated_path)
    if not generated_path: return None

    # ---------------- Step 2: Edit (微调格式和实体) ----------------
    msgs_edit = [
        {
            "role": "system",
            "content": (
                "You are a strict Medical Entity Condenser.\n"
                "Task: Simplify reasoning chains into **Concise Abstract Concepts**.\n\n"
                "STRICT RULES:\n"
                "1. **NO NUMBERS or SPECIFIC VALUES (CRITICAL)**: \n"
                "   - **Remove ALL** specific measurements, degrees, dosages, percentages, dates, scores, or ages.\n"
                "   - **Convert to Concept**: Replace values with their qualitative clinical state.\n"
                "     * 'Fever > 38.5°C' -> **'High fever'**\n"
                "     * 'BP 140/90' -> **'Hypertension'**\n"
                "     * 'Retroversion of 15 degrees' -> **'Retroversion'**\n"
                "     * 'Stage III' -> **'Advanced stage'** (or just the disease name)\n"
                "2. **Extreme Conciseness**: Extract ONLY the core medical concept. Target 1-5 words per node.\n"
                "3. **Remove Meta-Language**: DELETE functional phrases like 'Indication for', 'Evidence of', 'Risk of', 'History of'. Keep only the entity.\n"
                "4. **Entity Normalization**: Use the same standard term for the same concept throughout (e.g., 'Renal failure' everywhere).\n\n"
                "**Transformation Examples**:\n"
                "   - Input: 'Mild glenoid retroversion (>10 deg)' -> Output: **'Mild retroversion'**\n"
                "   - Input: 'Patient aged 75' -> Output: **'Elderly'**\n"
                "   - Input: 'Indication for eccentric anterior glenoid reaming' -> Output: **'Eccentric reaming'**\n"
                "   - Input: 'Hemoglobin 8 g/dL' -> Output: **'Anemia'**"
            )
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer: {answer}\nChains:\n{generated_path}\n\nOutput:"
        }
    ]
    refined_path = run_llm(MODEL_NAME, 0.0, msgs_edit) # 温度调低，求稳
    if refined_path == "**FAIL**": return None
    
    refined_path = deduplicate_reasoning_chains(refined_path)

    # ---------------- Step 3: Validate (验证) ----------------
    msgs_valid = [
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
                    f"{refined_path}\n\n"
                    "Output strictly as ONE of the following (single line):\n"
                    "PASS\n"
                    "or\n"
                    "FAIL, <concise reason>\n"
                )
            }
    ]
    validation = run_llm(MODEL_NAME, TEMPERATURE_STRICT, msgs_valid)

    # ---------------- Step 4: Repair / Augment (修复) ----------------
    final_path = ""
    
    if "PASS" in validation.upper():
        final_path = change_format(refined_path)
    else:
        # 如果生成的不行，或者被判为 FAIL，进行修复
        msgs_repair = [
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
                        f"Defective Chain: {refined_path}\n"
                        f"Failure Reason: {validation}\n\n"
                        "Output (Repaired Chain + Optional New Chains):"
                    )
                }
        ]
        repaired_path = run_llm(MODEL_NAME, TEMPERATURE_STRICT, msgs_repair)
        if repaired_path != "**FAIL**":
            final_path = change_format(repaired_path)

    if final_path:
        obj['new_reasoning_path'] = final_path
        obj['status'] = "GENERATED_PASS" if "PASS" in validation.upper() else "REPAIRED"
        return obj
    
    return None

# ================= 主程序入口 =================

def main():
    existing_jsonl_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Plan_mednew1.jsonl"
    existing_questions = set()
    
    if os.path.exists(existing_jsonl_path):
        print(f"正在加载去重文件: {existing_jsonl_path} ...")
        with open(existing_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if 'question' in obj:
                            existing_questions.add(obj['question'].strip())
                    except:
                        continue
        print(f"已加载 {len(existing_questions)} 条历史问题用于去重。")
    else:
        print(f"警告: 找不到去重文件 {existing_jsonl_path}，将不进行去重。")

    # 2. 加载 HF 数据集 (Test split 的前 1000 条)
    print("正在加载 TsinghuaC3I/MedXpertQA (test[:1000])...")
    try:
        # 使用切片语法直接只下载/加载前1000条
        ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test[:1000]")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    processed_count = 0
    skipped_count = 0
    
    # 3. 处理数据并写入 args.output
    # 注意：这里不再需要打开 args.input读取，但我们仍然需要写入 args.output
    with open(args.output, "w", encoding="utf-8") as f_out:
        
        for idx, item in enumerate(ds):
            # item 已经是字典格式，例如 {'question': ..., 'answer': ..., 'id': ...}
            
            question_text = item.get('question', '').strip()
            
            # --- 核心修改：去重逻辑 ---
            if not question_text:
                continue
                
            if question_text in existing_questions:
                skipped_count += 1
                # 如果想看跳过了哪些，可以取消下面注释
                # print(f"Skipping index {idx}: Duplicate question found.")
                continue
            # -----------------------

            try:
                if 'label' not in item:
                    print(f"Skipping index {idx}: Missing answer.")
                    continue
                item['answer'] = item['options'][item['label']]
                # 调用处理函数
                result_obj = process_line(item)
                
                if result_obj:
                    f_out.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed_count += 1
                    
                    # MedXpertQA 可能有 'id' 字段，也可能没有，做个兼容
                    current_id = result_obj.get('id', f"idx_{idx}")
                    print(f"[{processed_count}] Processed (Source Idx: {idx}). ID: {current_id} | Status: {result_obj.get('status')}")
                    
                    print(result_obj['new_reasoning_path'])
                    cost_total = total_input_tokens * PRICE_INPUT + total_output_tokens * PRICE_OUTPUT
                    print(f"Cost So Far: ${cost_total:.4f}")
                    print("-" * 30)
                else:
                    print(f"Failed to generate valid chains for index {idx}.")

            except Exception as e:
                print(f"Error processing index {idx}: {e}")

    print(f"\nDone!")
    print(f"Total processed (New): {processed_count}")
    print(f"Total skipped (Duplicates): {skipped_count}")
    final_cost = total_input_tokens * PRICE_INPUT + total_output_tokens * PRICE_OUTPUT
    print(f"Final Estimated Cost: ${final_cost:.4f}")

if __name__ == "__main__":
    main()