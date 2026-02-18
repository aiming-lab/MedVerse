import json
import os

def remove_overlapping_questions(target_file, reference_file, output_file):
    print(f"正在加载参考文件 (黑名单源): {reference_file}")
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
    except Exception as e:
        print(f"读取参考文件失败: {e}")
        return

    seen_questions = set()
    
    if isinstance(ref_data, dict):
        iterable = ref_data.values()
    elif isinstance(ref_data, list):
        iterable = ref_data
    else:
        print("参考文件格式无法识别")
        return

    for item in iterable:
        # 使用 .strip() 避免因首尾空格导致的匹配失败
        q = item.get("Question", "").strip()
        if q:
            seen_questions.add(q)
            
    print(f"黑名单构建完成，包含 {len(seen_questions)} 个唯一问题。")

    # --- 2. 读取并过滤目标文件 ---
    print(f"正在加载待处理文件: {target_file}")
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
    except Exception as e:
        print(f"读取目标文件失败: {e}")
        return

    original_count = len(target_data)
    filtered_data = None
    removed_count = 0

    # 根据目标文件的结构 (Dict 或 List) 分别处理
    if isinstance(target_data, dict):
        filtered_data = {}
        # 尝试排序 key 处理 (可选)
        keys = target_data.keys()
        for key in keys:
            item = target_data[key]
            q = item.get("Question", "").strip()
            
            if q in seen_questions:
                removed_count += 1
            else:
                filtered_data[key] = item
                
    elif isinstance(target_data, list):
        filtered_data = []
        for item in target_data:
            q = item.get("Question", "").strip()
            
            if q in seen_questions:
                removed_count += 1
            else:
                filtered_data.append(item)
    else:
        print("待处理文件格式无法识别")
        return

    # --- 3. 保存结果 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(filtered_data, f_out, ensure_ascii=False, indent=4)
        
        print("-" * 30)
        print(f"处理完成！")
        print(f"原数据量: {original_count}")
        print(f"剔除重复数据: {removed_count}")
        print(f"剩余数据量: {len(filtered_data)}")
        print(f"文件已保存至: {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")

# --- 配置路径 ---
# 待处理的文件 (要从中删除数据)
target_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Plan_medqa.json"

# 参考文件 (如果在里面出现了，就删掉)
ref_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Final_8k.json"

# 输出文件 (自动加上 _Filtered 后缀)
output_dir = os.path.dirname(target_path)
output_filename = os.path.splitext(os.path.basename(target_path))[0] + "_Filtered.json"
output_path = os.path.join(output_dir, output_filename)

if __name__ == "__main__":
    remove_overlapping_questions(target_path, ref_path, output_path)