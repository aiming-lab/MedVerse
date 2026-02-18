import json
import os

folder_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl"

data = []

for file in sorted(os.listdir(folder_path)):
    if file.find('merged_sorted')!=-1 or file.find('DS_Store')!=-1:
        continue
    file_path = os.path.join(folder_path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data.append(obj)

unique_data = {d["id"]: d for d in data}.values()

sorted_data = sorted(unique_data, key=lambda x: x["id"])

with open("/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/merged_sorted.jsonl", "w", encoding="utf-8") as f:
    for item in sorted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已按 id 排序合并完成，共 {len(data)} 条，输出文件：merged_sorted.jsonl")
