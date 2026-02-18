import json
import os

# 输入文件路径
file_paths = [
    "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/LastHumanity_0_44.jsonl",
    "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/MedXpertQA_0_300.jsonl"
]

# 输出文件路径 (保存到同一目录下)
output_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/merged_cleaned_data.jsonl"

def merge_and_clean(input_files, output_file):
    total_count = 0
    kept_count = 0
    dropped_count = 0

    print(f"正在处理，结果将保存至: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"警告: 找不到文件 {file_path}，跳过。")
                continue
            
            print(f"正在读取: {os.path.basename(file_path)} ...")
            
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        total_count += 1
                        
                        # 检查 reasoning 字段
                        reasoning_content = data.get("reasoning", "")
                        
                        # 过滤逻辑：如果 reasoning 中包含指定错误信息，则丢弃
                        if reasoning_content and "No reasoning path found." in reasoning_content:
                            dropped_count += 1
                            continue
                        
                        # 写入保留的数据
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        kept_count += 1
                        
                    except json.JSONDecodeError:
                        print(f"跳过无法解析的行: {line[:50]}...")
                        continue

    print("-" * 30)
    print("处理完成！")
    print(f"原始总条数: {total_count}")
    print(f"删除条数 (包含错误): {dropped_count}")
    print(f"保留条数 (有效数据): {kept_count}")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    merge_and_clean(file_paths, output_path)