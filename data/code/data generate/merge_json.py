import json
import os

def merge_filter_split(file_path1, file_path2, clean_output_path, filtered_output_path):
    # --- 1. 读取文件 ---
    try:
        with open(file_path1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        print(f"成功加载文件 1: {len(data1)} 条数据")
    except Exception as e:
        print(f"读取文件 1 失败: {e}")
        return

    try:
        with open(file_path2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        print(f"成功加载文件 2: {len(data2)} 条数据")
    except Exception as e:
        print(f"读取文件 2 失败: {e}")
        return

    # --- 2. 准备容器和计数器 ---
    clean_data = {}
    clean_counter = 1
    
    filtered_data = {}
    filtered_counter = 1

    # 定义要过滤的“坏”字符串特征
    # 注意：这里假设 JSON 解析后的字符串中包含了真实的换行符
    BAD_PLAN_STR = "<Plan>\n\n</Plan>"
    BAD_EXEC_STR = "<Execution>\n</Execution>"

    def process_data(source_data, source_name):
        nonlocal clean_counter, filtered_counter
        
        # 尝试排序 Key 以保证处理顺序
        try:
            sorted_keys = sorted(source_data.keys(), key=lambda x: int(x))
        except:
            sorted_keys = source_data.keys()

        for key in sorted_keys:
            item = source_data[key]
            
            # 获取字段内容，使用 .get 防止字段不存在报错
            plan_prompt = item.get("Transient Plan Prompt", "")
            exec_prompt = item.get("Transient Execution Prompt", "")
            opt_prompt = item.get("Options", "")

            # --- 核心判断逻辑 ---
            # 如果 Plan 是空的 OR Execution 是空的
            # 有些时候可能包含空格，所以我加了 .strip() == ... 的容错，
            # 如果你需要严格匹配字符串，去掉 .strip() 即可。
            is_bad_plan = plan_prompt == BAD_PLAN_STR
            is_bad_exec = exec_prompt == BAD_EXEC_STR
            is_bad_options = opt_prompt == ""
            
            if is_bad_plan or is_bad_exec:
                # 放入被过滤的字典中，并重新编号
                new_key = str(filtered_counter)
                filtered_data[new_key] = item
                filtered_counter += 1
            else:
                # 放入干净的字典中，并重新编号
                new_key = str(clean_counter)
                clean_data[new_key] = item
                clean_counter += 1

    # --- 3. 执行处理 ---
    print("正在处理文件 1...")
    process_data(data1, "文件1")
    
    print("正在处理文件 2...")
    process_data(data2, "文件2")

    # --- 4. 保存两个文件 ---
    
    # 保存干净数据
    try:
        with open(clean_output_path, 'w', encoding='utf-8') as f_out:
            json.dump(clean_data, f_out, ensure_ascii=False, indent=4)
        print("-" * 30)
        print(f"[干净数据] 保存完成！")
        print(f"路径: {clean_output_path}")
        print(f"数量: {len(clean_data)}")
    except Exception as e:
        print(f"保存干净数据失败: {e}")

    # 保存被过滤的数据
    try:
        with open(filtered_output_path, 'w', encoding='utf-8') as f_out:
            for key, item in filtered_data.items():
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print("-" * 30)
        print(f"[被过滤数据] 保存完成 (JSONL格式)！")
        print(f"路径: {filtered_output_path}")
        print(f"数量: {len(filtered_data)}")
    except Exception as e:
        print(f"保存过滤数据失败: {e}")

# --- 配置路径 ---
file1 = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Final_12k_Clean.json"
file2 = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Final_10_1_1_1_1.json"

output_dir = os.path.dirname(file1)

# 输出文件 1：干净的数据
clean_file = os.path.join(output_dir, "Merged_Med_Reason_Clean.json")
# 输出文件 2：被过滤掉的数据 (包含空 Plan 或 空 Execution)
filtered_file = os.path.join(output_dir, "Merged_Med_Reason_Filtered.jsonl")

if __name__ == "__main__":
    merge_filter_split(file1, file2, clean_file, filtered_file)