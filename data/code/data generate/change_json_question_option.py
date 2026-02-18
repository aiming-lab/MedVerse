import json
import os

def refine_questions(file_path, output_path):
    # 1. 读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载文件: {len(data)} 条数据")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    if not isinstance(data, dict):
        print("错误：文件必须是字典格式")
        return

    # 计数器
    options_extracted_count = 0  
    conclusion_cleaned_count = 0 
    skipped_split_count = 0      
    
    # 按照ID排序处理
    try:
        sorted_keys = sorted(data.keys(), key=lambda x: int(x))
    except:
        sorted_keys = data.keys()

    for key in sorted_keys:
        item = data[key]
        
        # ==========================================
        # 逻辑 1: 分离 Question 和 Options (保持不变)
        # ==========================================
        question_text = item.get("Question", "")
        split_marker = "Answer Choices:"
        
        if split_marker in question_text:
            part1, part2, part3 = question_text.partition(split_marker)
            item["Question"] = part1.strip()
            item["Options"] = (part2 + part3).strip()
            options_extracted_count += 1
        else:
            skipped_split_count += 1
            if "Options" not in item:
                item["Options"] = ""

        # ==========================================
        # 逻辑 2 (修改): 清洗 Conclusion，保留 Explanation
        # 条件: Options 为空 AND Conclusion 含有 "Explanation"
        # ==========================================
        current_options = item.get("Options", "")
        current_conclusion = item.get("Conclusion", "")
        
        if not current_options:
            keyword = "Explanation"
            if keyword in current_conclusion:
                # part1: Explanation 之前的内容 (丢弃)
                # part2: "Explanation" 这个词本身 (保留)
                # part3: Explanation 之后的内容 (保留)
                _, part2, part3 = current_conclusion.partition(keyword)
                
                # 拼接 part2 和 part3，并去除首尾空格
                item["Conclusion"] = (part2 + part3).strip()
                
                conclusion_cleaned_count += 1

    # 3. 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=4)
        
        print("-" * 30)
        print(f"处理完成！")
        print(f"总数据量: {len(data)}")
        print(f"成功分离 Options 的数据: {options_extracted_count} 条")
        print(f"成功清洗 Conclusion 的数据: {conclusion_cleaned_count} 条 (丢弃了Explanation之前的废话)")
        print(f"文件已保存至: {output_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")

# --- 配置路径 ---
input_file = "/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Final_10_1_1_1.json"
output_dir = os.path.dirname(input_file)
output_file = os.path.join(output_dir, "Med_Reason_Final_10_1_1_1_Refined.json")

if __name__ == "__main__":
    refine_questions(input_file, output_file)