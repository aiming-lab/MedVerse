from datasets import load_dataset, load_from_disk
import json, random, re

random.seed(42)

# 1) 加载数据
ds = load_dataset("UCSC-VLAA/MedReason", split="train")
# ds1 = load_from_disk("/home/ubuntu/Med-Reason/Multiverse/train/Med_Reason")["train"]

# 可选上限：最多取多少条（你的需求是1000条）
CAP = 1000

def clean(s: str) -> str:
    return re.sub(r"\s+\n", "\n", (s or "")).strip()

ours_data = []
matched = 0

# 2) 匹配 + 转成 SFT.py 需要的 ours 格式
for ex in ds:
    q = clean(ex.get("question"))
    if not q:
        continue

    # 在本地 ds1 的 text 中查找 question（沿用你的逻辑）
    # found = False
    # for ex1 in ds1:
    #     text = ex1.get("text") or ""
    #     if text.find(q) != -1:
    #         found = True
    #         break
    # if not found:
    #     continue

    a = clean(ex.get("answer") or ex.get("output"))
    r = clean(ex.get("reasoning") or ex.get("rationale") or "")
    if not a:
        continue

    ours_data.append({
        "question": q,
        "reasoning": r,   # 可为空串
        "answer": a
    })
    matched += 1
    # if CAP and len(ours_data) >= CAP:
    #     break

print(f"Matched questions: {matched}")
print(f"Collected (ours schema): {len(ours_data)}")

# 3) 打乱并 9:1 切分
random.shuffle(ours_data)
n_train = int(len(ours_data) * 0.95)
train_set = ours_data[:n_train]
eval_set  = ours_data[n_train:]

# 4) 写文件（文件名包含“ours”，以触发 SFT.py 的 ours 分支）
with open("medreason_ours_all_30k.jsonl", "w", encoding="utf-8") as f:
    for ex in ours_data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open("medreason_ours_train_30k.jsonl", "w", encoding="utf-8") as f:
    for ex in train_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open("medreason_ours_eval_30k.jsonl", "w", encoding="utf-8") as f:
    for ex in eval_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Total {len(ours_data)} , Train {len(train_set)} , Evaluation {len(eval_set)}.")
