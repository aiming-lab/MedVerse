import json
import os

def process_med_reason_dataset(target_file, reference_file, output_file):
    # --- 1. 加载参考数据 (LastHumanity.jsonl) ---
    # 构建 "问题文本 -> 选项字典" 的映射表
    ref_map = {}
    print(f"正在加载参考文件: {reference_file}")
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    q = item.get('question', '').strip()
                    opts = item.get('options', {})
                    if q:
                        ref_map[q] = opts
                except:
                    continue
        print(f"参考库构建完成，包含 {len(ref_map)} 个问题。")
    except Exception as e:
        print(f"读取参考文件失败: {e}")
        return

    # --- 2. 处理目标文件 ---
    print(f"正在处理目标文件: {target_file}")
    processed_count = 0
    split_count = 0
    lookup_count = 0
    
    try:
        with open(target_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                except:
                    continue

                # 1. 删除不需要的字段 (使用 .pop 防止key不存在报错)
                item.pop('original_reasoning', None)
                item.pop('reasoning', None)

                # 获取原始问题文本
                raw_question = item.get('question', "")
                split_marker = "\nAnswer Choices:"

                # 2. 处理 Options 逻辑
                if split_marker in raw_question:
                    # --- 情况 A: 文本内包含 Answer Choices ---
                    # partition 返回 (分隔符前, 分隔符本身, 分隔符后)
                    part1, part2, part3 = raw_question.partition(split_marker)
                    
                    # 更新 Question (去除分割符前面的空格)
                    item['question'] = part1.strip()
                    
                    # 构造 Options (保留 "Answer Choices:" 前缀以保持格式统一)
                    # 题目要求 "后面的内容放到 options"，为了语义完整建议保留前缀
                    # 结果形式: "Answer Choices: (A) xxx (B) xxx"
                    item['options'] = (part2 + part3).strip()
                    
                    split_count += 1
                
                else:
                    # --- 情况 B: 文本内无选项，去参考库查找 ---
                    clean_q = raw_question.strip()
                    
                    if clean_q in ref_map:
                        source_options = ref_map[clean_q]
                        
                        # 将字典格式 {'A': 'xx', 'B': 'xx'} 转换为字符串格式
                        # 格式化为: "Answer Choices:\nA. xx\nB. xx"
                        formatted_lines = ["Answer Choices:"]
                        for label in sorted(source_options.keys()):
                            content = source_options[label]
                            formatted_lines.append(f"{label}. {content}")
                        
                        item['options'] = "\n".join(formatted_lines)
                        lookup_count += 1
                    else:
                        # 如果没找到，options 可能会是 None 或者空字符串，视情况保留原样
                        if 'options' not in item:
                            item['options'] = ""

                # 3. 写入新文件
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                processed_count += 1

        print("-" * 30)
        print(f"处理完成！")
        print(f"总处理行数: {processed_count}")
        print(f"通过分割提取 Options: {split_count}")
        print(f"通过查找补充 Options: {lookup_count}")
        print(f"输出文件: {output_file}")

    except Exception as e:
        print(f"处理过程中出错: {e}")

# --- 配置路径 ---
target_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/MedXperQA_reasoning_path.jsonl"
ref_path = "/Users/jimchen/Documents/Med-Cot/codes/datasets/jsonl/LastHumanity.jsonl"

# 输出文件路径 (自动添加 _Cleaned 后缀)
output_dir = os.path.dirname(target_path)
output_filename = os.path.splitext(os.path.basename(target_path))[0] + "_Cleaned.jsonl"
output_path = os.path.join(output_dir, output_filename)

if __name__ == "__main__":
    process_med_reason_dataset(target_path, ref_path, output_path)