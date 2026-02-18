#!/usr/bin/env python3
"""
将 Med_Reason.jsonl 中的 text 部分解析为结构化的可读格式
每个样本会保存为单独的文件，包含系统提示、用户问题、助手回复等部分
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def parse_chatml(text: str) -> Dict[str, str]:
    """
    解析 ChatML 格式的文本，提取各个部分
    """
    # 提取系统提示
    system_match = re.search(r'<\|im_start\|>system\n(.*?)\n<\|im_end\|>', text, re.DOTALL)
    system_prompt = system_match.group(1).strip() if system_match else ""
    
    # 提取用户输入
    user_match = re.search(r'<\|im_start\|>user\n(.*?)\n<\|im_end\|>', text, re.DOTALL)
    user_input = user_match.group(1).strip() if user_match else ""
    
    # 提取助手回复
    assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)\n<\|im_end\|>', text, re.DOTALL)
    assistant_response = assistant_match.group(1).strip() if assistant_match else ""
    
    # 解析助手回复中的各个部分
    think_section = ""
    plan_section = ""
    execution_section = ""
    conclusion_section = ""
    final_answer = ""
    
    if assistant_response:
        # 提取 <Think> 部分
        think_match = re.search(r'<Think>(.*?)</Think>', assistant_response, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            
            # 从 Think 部分提取 Plan
            plan_match = re.search(r'<Plan>(.*?)</Plan>', think_content, re.DOTALL)
            if plan_match:
                plan_section = plan_match.group(1).strip()
            
            # 从 Think 部分提取 Execution
            execution_match = re.search(r'<Execution>(.*?)</Execution>', think_content, re.DOTALL)
            if execution_match:
                execution_section = execution_match.group(1).strip()
            
            # 从 Think 部分提取 Conclusion
            conclusion_match = re.search(r'<Conclusion>(.*?)</Conclusion>', think_content, re.DOTALL)
            if conclusion_match:
                conclusion_section = conclusion_match.group(1).strip()
        
        # 提取最终答案（在 Think 标签外的部分）
        final_answer_match = re.search(r'</Think>\n(.*?)$', assistant_response, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
    
    return {
        "system_prompt": system_prompt,
        "user_input": user_input,
        "think_section": think_section,
        "plan_section": plan_section,
        "execution_section": execution_section,
        "conclusion_section": conclusion_section,
        "final_answer": final_answer,
        "full_assistant_response": assistant_response
    }

def format_readable_text(parsed_data: Dict[str, str], sample_id: str) -> str:
    """
    将解析后的数据格式化为可读的文本
    """
    output = []
    output.append("=" * 80)
    output.append(f"样本 ID: {sample_id}")
    output.append("=" * 80)
    output.append("")
    
    # 系统提示
    if parsed_data["system_prompt"]:
        output.append("🤖 系统提示:")
        output.append("-" * 40)
        output.append(parsed_data["system_prompt"])
        output.append("")
    
    # 用户输入
    if parsed_data["user_input"]:
        output.append("👤 用户问题:")
        output.append("-" * 40)
        output.append(parsed_data["user_input"])
        output.append("")
    
    # 思考过程
    if parsed_data["think_section"]:
        output.append("🧠 思考过程:")
        output.append("-" * 40)
        output.append(parsed_data["think_section"])
        output.append("")
    
    # 计划部分
    if parsed_data["plan_section"]:
        output.append("📋 推理计划:")
        output.append("-" * 40)
        output.append(parsed_data["plan_section"])
        output.append("")
    
    # 执行部分
    if parsed_data["execution_section"]:
        output.append("⚡ 执行步骤:")
        output.append("-" * 40)
        output.append(parsed_data["execution_section"])
        output.append("")
    
    # 结论部分
    if parsed_data["conclusion_section"]:
        output.append("🎯 结论:")
        output.append("-" * 40)
        output.append(parsed_data["conclusion_section"])
        output.append("")
    
    # 最终答案
    if parsed_data["final_answer"]:
        output.append("✅ 最终答案:")
        output.append("-" * 40)
        output.append(parsed_data["final_answer"])
        output.append("")
    
    return "\n".join(output)

def main():
    input_file = Path("./json/Med_Reason.jsonl")
    output_dir = Path("./parsed_samples")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    (output_dir / "individual").mkdir(exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)
    (output_dir / "full_text").mkdir(exist_ok=True)
    
    print(f"正在处理文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    all_samples = []
    sample_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                sample_id = data.get("id", f"sample_{line_num}")
                text = data.get("text", "")
                
                if not text:
                    print(f"警告: 第 {line_num} 行没有 text 内容")
                    continue
                
                # 解析 ChatML 格式
                parsed_data = parse_chatml(text)
                
                # 生成可读格式
                readable_text = format_readable_text(parsed_data, sample_id)
                
                # 保存单个样本文件（结构化格式）
                individual_file = output_dir / "individual" / f"sample_{sample_id}.txt"
                with open(individual_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(readable_text)
                
                # 保存完整text文件（不分块）
                full_text_file = output_dir / "full_text" / f"sample_{sample_id}_full.txt"
                with open(full_text_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(f"样本 ID: {sample_id}\n")
                    out_f.write("=" * 80 + "\n\n")
                    out_f.write(text)
                
                # 收集样本信息用于汇总
                all_samples.append({
                    "id": sample_id,
                    "parsed_data": parsed_data,
                    "readable_text": readable_text,
                    "full_text": text
                })
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"已处理 {sample_count} 个样本...")
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"错误: 处理第 {line_num} 行时出错: {e}")
                continue
    
    # 生成汇总文件
    print(f"\n正在生成汇总文件...")
    
    # 1. 所有样本的汇总
    summary_file = output_dir / "summary" / "all_samples_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"医学推理数据集解析汇总\n")
        f.write(f"总样本数: {sample_count}\n")
        f.write(f"处理时间: {Path().cwd()}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, sample in enumerate(all_samples, 1):
            f.write(f"样本 {i}/{sample_count} (ID: {sample['id']})\n")
            f.write("-" * 40 + "\n")
            
            # 只显示用户问题和最终答案
            user_input = sample['parsed_data']['user_input']
            final_answer = sample['parsed_data']['final_answer']
            
            f.write(f"问题: {user_input[:200]}{'...' if len(user_input) > 200 else ''}\n")
            f.write(f"答案: {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}\n")
            f.write("\n")
    
    # 2. 问题列表
    questions_file = output_dir / "summary" / "questions_list.txt"
    with open(questions_file, 'w', encoding='utf-8') as f:
        f.write("医学推理问题列表\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(all_samples, 1):
            user_input = sample['parsed_data']['user_input']
            f.write(f"{i}. {user_input}\n\n")
    
    # 3. 答案列表
    answers_file = output_dir / "summary" / "answers_list.txt"
    with open(answers_file, 'w', encoding='utf-8') as f:
        f.write("医学推理答案列表\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(all_samples, 1):
            final_answer = sample['parsed_data']['final_answer']
            f.write(f"{i}. {final_answer}\n\n")
    
    # 4. 完整text汇总文件
    full_text_summary = output_dir / "summary" / "all_full_texts.txt"
    with open(full_text_summary, 'w', encoding='utf-8') as f:
        f.write("医学推理数据集 - 完整Text汇总\n")
        f.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(all_samples, 1):
            f.write(f"样本 {i}/{len(all_samples)} (ID: {sample['id']})\n")
            f.write("=" * 80 + "\n")
            f.write(sample['full_text'])
            f.write("\n\n" + "=" * 80 + "\n\n")
    
    # 5. 统计信息
    stats_file = output_dir / "summary" / "statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集统计信息\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"总样本数: {sample_count}\n")
        f.write(f"成功解析: {len(all_samples)}\n")
        
        # 计算各部分长度统计
        plan_lengths = [len(s['parsed_data']['plan_section']) for s in all_samples if s['parsed_data']['plan_section']]
        execution_lengths = [len(s['parsed_data']['execution_section']) for s in all_samples if s['parsed_data']['execution_section']]
        conclusion_lengths = [len(s['parsed_data']['conclusion_section']) for s in all_samples if s['parsed_data']['conclusion_section']]
        
        f.write(f"\n计划部分平均长度: {sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0:.1f} 字符\n")
        f.write(f"执行部分平均长度: {sum(execution_lengths) / len(execution_lengths) if execution_lengths else 0:.1f} 字符\n")
        f.write(f"结论部分平均长度: {sum(conclusion_lengths) / len(conclusion_lengths) if conclusion_lengths else 0:.1f} 字符\n")
    
    print(f"\n✅ 处理完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 结构化样本文件: {output_dir}/individual/")
    print(f"📄 完整text文件: {output_dir}/full_text/")
    print(f"📊 汇总文件: {output_dir}/summary/")
    print(f"📈 总共处理了 {sample_count} 个样本")

if __name__ == "__main__":
    main()
