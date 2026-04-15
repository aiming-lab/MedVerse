import argparse
from tqdm import tqdm
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from jinja2 import Template


def get_results(log_save_path):
    """Compute accuracy by matching predicted answer letter against gold label."""
    import re
    with open(log_save_path) as f:
        results = json.load(f)
    correct, total = 0, 0
    for item in results:
        gold = str(item.get("answer_idx", item.get("answer", ""))).strip().upper()
        pred = item.get("output", "")
        # Extract last "The answer is X" or standalone letter near end
        match = re.search(r"(?:answer is|Answer:)\s*([A-E])", pred, re.IGNORECASE)
        if not match:
            match = re.search(r"\b([A-E])\b(?=[^A-E]*$)", pred)
        predicted = match.group(1).upper() if match else ""
        if predicted == gold:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    result_path = log_save_path.replace(".json", "_score.json")
    with open(result_path, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total}, f, indent=2)
    return accuracy


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    # if the file is json file, load it
    if input_fp.endswith('.json'):
        with open(input_fp, 'r') as f:
            data = json.load(f)
    elif input_fp.endswith('.jsonl'):
        data = []
        with open(input_fp, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {input_fp}")

    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k, v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=2000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template', action="store_true")
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--task_floder', type=str, default='anonymous_run')
    parser.add_argument('--dtype', type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    # dtype 选择
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    template = None
    if args.use_chat_template and getattr(tokenizer, "chat_template", None):
        template = Template(tokenizer.chat_template)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=0.9,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_data = load_file(args.eval_file)
    final_results = []

    if args.strict_prompt:
        query_prompt = ("Please answer the following multiple-choice questions. "
                        "Please answer the following multiple-choice questions, ensuring your response concludes "
                        "with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}\n")
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}\n"

    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx * args.batch_size:min((idx + 1) * args.batch_size, len(input_data))]
        if len(batch) == 0:
            break

        for item in batch:
            item['option_str'] = '\n'.join([f"{op}. {ans}" for op, ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)

        prompts = [item["input_str"] for item in batch]

        if args.use_chat_template and template is not None:
            prompts = [template.render(messages=[{"role": "user", "content": p}],
                                       bos_token=tokenizer.bos_token,
                                       add_generation_prompt=True)
                       for p in prompts]

        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids, skip_special_tokens=False))
                else:
                    new_prompts.append(prompt)
            prompts = new_prompts

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_cfg)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, item in enumerate(batch):
            pred = decoded[j]
            if args.strict_prompt and item["input_str"] in pred:
                pred = pred.split(item["input_str"], 1)[-1].strip()
            pred = postprocess_output(pred)
            if pred:
                item["output"] = pred
                final_results.append(item)

    task_floder = f'./results/{args.task_floder}'
    os.makedirs(f'{task_floder}/logs', exist_ok=True)
    os.makedirs(f'{task_floder}/result', exist_ok=True)

    task_name = os.path.split(args.model_name)[-1]
    task_name = task_name + os.path.basename(args.eval_file).replace('.json', '').replace('.jsonl', '') + "_hf" + (
        '_strict-prompt' if args.strict_prompt else '')
    log_save_path = f'{task_floder}/logs/{task_name}.json'
    with open(log_save_path, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 打分
    get_results(log_save_path)


if __name__ == "__main__":
    main()
