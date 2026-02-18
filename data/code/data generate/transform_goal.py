import json
import re
from typing import Dict, Tuple, List, Union

json_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Plan_total.json"
output_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Plan_total_new.json"

# 将 "Answer Choices:\nA. foo\nB. bar\n..." 解析为 { "A": "foo", "B": "bar", ... }
def parse_options_block(options_str: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in options_str.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("answer choices"):
        lines = lines[1:]

    pat = re.compile(r"^\s*([A-Za-z])\.\s*(.+?)\s*$")
    parsed: Dict[str, str] = {}
    for ln in lines:
        m = pat.match(ln)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            parsed[letter] = text
    return parsed

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[ \t\r\n.,'’\"`-]+", "", s)
    return s

def find_option_letter(options_map: Dict[str, str], goal_text: str) -> Union[str, None]:
    norm_goal = normalize(goal_text)
    for letter, txt in options_map.items():
        if normalize(txt) == norm_goal:
            return letter
    for letter, txt in options_map.items():
        ntxt = normalize(txt)
        if norm_goal in ntxt or ntxt in norm_goal:
            return letter
    return None

with open(json_path, "r") as f:
    data = json.load(f)
k_list, v_list = [], []

modified = 0

for idx, item in data.items():
    if not all(k in item for k in ("Question", "Options", "Goal")):
        continue
    question = item["Question"]
    options_raw = item["Options"]
    goal_text = item["Goal"]
    item["Answer"] = item["Goal"]

    if isinstance(options_raw, list):
        options_str = "Answer Choices:\n" + "\n".join(options_raw)
    else:
        options_str = str(options_raw)

    options_map = parse_options_block(options_str)
    if not options_map:
        print(f"⚠️ 第 {idx} 题：无法解析 Options（不是 A./B./C./D. 格式？）")
        k_list.append(idx)
        v_list.append(item)
        continue

    m = re.match(r"^\s*([A-Za-z])\s*$", str(goal_text))
    if m:
        letter = m.group(1).upper()
        if letter in options_map:
            item["Goal"] = f"The answer is {letter}"
            modified += 1
            print(f"第 {idx} 题：Goal 已是字母，规范化为 -> {item['Goal']}")
            k_list.append(idx)
            v_list.append(item)
            continue

    letter = find_option_letter(options_map, str(goal_text))
    if letter is None:
        print(f"⚠️ 第 {idx} 题：未找到与 Goal='{goal_text}' 匹配的选项。")
        k_list.append(idx)
        v_list.append(item)
        continue

    item["Answer"] = f"The answer is {letter}"
    modified += 1
    print(f"第 {idx} 题修改完成：{item['Goal']}")
    k_list.append(idx)
    v_list.append(item)


result={k:v for k,v in zip(k_list,v_list)}
with open(output_path,'w') as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

print(f"完成：共修改 {modified} 条，已写入 {output_path}")
