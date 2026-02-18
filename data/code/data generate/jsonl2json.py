import json

def jsonl_to_json_by_id(input_path, output_path):
    result = {}
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        record_id = str(data['id']) 
                        result[record_id] = data
                    else:
                        print(f"警告: 第 {line_number} 行缺少 'id' 字段，已跳过。")
                        
                except json.JSONDecodeError:
                    print(f"错误: 第 {line_number} 行 JSON 格式无效。")

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(result, f_out, ensure_ascii=False, indent=4)
            
        print(f"转换完成！共处理 {len(result)} 条数据，已保存至 {output_path}")

    except FileNotFoundError:
        print("错误: 找不到输入文件。")

input_filename = '/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/MedXperQA_reasoning_path.jsonl'
output_filename = '/Users/jimchen/Documents/Med-Cot/codes/datasets/json/MedXperQA_reasoning_path.json'

jsonl_to_json_by_id(input_filename, output_filename)