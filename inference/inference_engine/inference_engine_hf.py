import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import networkx as nx
import re
import copy
import time
import json
import os

MODEL_PATH = "/data/LLaMA-3.1-8B/Multiverse-20251204/checkpoint-100"
DATASET_PATH = "/home/ubuntu/Med-Reason/MedReason/eval_data/HLE_biomed.jsonl"
NUM_SAMPLES_TO_TEST = 5

class PlanParser:
    """Parses the reasoning plan into a directed graph."""
    def __init__(self):
        self.item_pattern = re.compile(
            r"<Outline>\s*(Transient Step\s+(\d+):\s*(.*?));\s*Dependency:\s*\[(.*?)\]\s*</Outline>", 
            re.DOTALL | re.IGNORECASE
        )

    def parse(self, text):
        graph = nx.DiGraph()
        plan_match = re.search(r"<Plan>(.*?)(?:</Plan>|$)", text, re.DOTALL)
        if not plan_match:
            return graph
            
        content_inner = plan_match.group(1)
        matches = self.item_pattern.findall(content_inner)
        
        for full_desc, step_id_str, step_content, dep_str in matches:
            step_id = int(step_id_str)
            deps = []
            if dep_str.strip():
                try:
                    deps = [int(d.strip()) for d in dep_str.split(',') if d.strip()]
                except ValueError: pass
            
            graph.add_node(step_id, description=full_desc, content=step_content)
            for dep_id in deps:
                if dep_id != step_id:
                    graph.add_edge(dep_id, step_id)
        return graph

class KVCacheManager:
    """
    Manages Hugging Face past_key_values.
    """
    def __init__(self, device):
        self.device = device

    def fork_cache(self, past_key_values, num_branches):
        """
        Forks a Batch=1 cache into Batch=N via physical copying.
        """
        if past_key_values is None:
            return None
        
        new_cache = DynamicCache()
        if hasattr(past_key_values, "key_cache"):
            num_layers = len(past_key_values.key_cache)
            for layer_idx in range(num_layers):
                # Shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
                k_expanded = k.repeat_interleave(num_branches, dim=0)
                v_expanded = v.repeat_interleave(num_branches, dim=0)
                new_cache.update(k_expanded, v_expanded, layer_idx)
        elif isinstance(past_key_values, tuple):
            for layer_idx, (k, v) in enumerate(past_key_values):
                k_expanded = k.repeat_interleave(num_branches, dim=0)
                v_expanded = v.repeat_interleave(num_branches, dim=0)
                new_cache.update(k_expanded, v_expanded, layer_idx)
                
        return new_cache

# ==========================================
# Topological Parallel Engine
# ==========================================
class HFTopologicalEngine:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=self.device
        )
        self.mem_manager = KVCacheManager(self.device)
        self.parser = PlanParser()

    def _generate_batch(self, input_texts, prefix_kv_cache, max_new_tokens=256):
        batch_size = len(input_texts)
        if batch_size == 0: return []
        current_batch_kv = self.mem_manager.fork_cache(prefix_kv_cache, batch_size)
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        if hasattr(current_batch_kv, "get_seq_length"):
            prefix_len = current_batch_kv.get_seq_length()
        else:
            prefix_len = current_batch_kv[0][0].shape[2] if current_batch_kv else 0
        prefix_mask = torch.ones((batch_size, prefix_len), device=self.device, dtype=inputs.attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=full_attention_mask,
                past_key_values=current_batch_kv, 
                max_new_tokens=max_new_tokens,
                stop_strings=["</Step>"], 
                tokenizer=self.tokenizer,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        input_len = inputs.input_ids.shape[1]
        generated_texts = []
        
        for i in range(batch_size):
            new_tokens = output_ids[i][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            text = text.split("</Step>")[0]
            generated_texts.append(text)
            
        return generated_texts

    def run(self, question):
        prompt_header = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"Question:\n{question}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"<Think>\n"
        )
        inputs = self.tokenizer(prompt_header, return_tensors="pt").to(self.device)
        
        full_plan_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512, 
            stop_strings=["</Plan>"], 
            tokenizer=self.tokenizer,
            do_sample=False
        )
        full_text_with_plan = self.tokenizer.decode(full_plan_ids[0], skip_special_tokens=False)
        
        if "</Plan>" in full_text_with_plan:
            plan_block = full_text_with_plan.split("<Think>\n")[1]
        else:
            plan_block = "<Plan>...</Plan>"
            
        context_so_far = prompt_header + plan_block
        inputs_root = self.tokenizer(context_so_far, return_tensors="pt").to(self.device)
        with torch.no_grad():
            root_outputs = self.model(**inputs_root, use_cache=True)
        root_kv_cache = root_outputs.past_key_values
        
        graph = self.parser.parse(plan_block)
        results = {}
        
        while len(results) < len(graph.nodes):
            current_wave = []
            for node_id in graph.nodes:
                if node_id in results: continue
                preds = list(graph.predecessors(node_id))
                if all(p in results for p in preds):
                    current_wave.append(node_id)
            
            if not current_wave:
                break

            wave_inputs = []
            for node_id in current_wave:
                step_data = graph.nodes[node_id]
                preds = sorted(list(graph.predecessors(node_id)))
                dep_context = ""
                if preds:
                    dep_context = "\n".join([results[p] for p in preds])
                input_text = f"\n<Execution>\n{dep_context}\n<Step> {step_data['description']}"
                wave_inputs.append(input_text)
            print(f"⚡ Running Wave: {current_wave} (Batch Size: {len(wave_inputs)})")
            wave_outputs = self._generate_batch(wave_inputs, root_kv_cache)
            
            for i, node_id in enumerate(current_wave):
                full_res = f"<Step> {graph.nodes[node_id]['description']}{wave_outputs[i]}</Step>"
                results[node_id] = full_res

        sorted_steps = [results[i] for i in sorted(results.keys())]
        full_execution = "\n".join(sorted_steps) + "\n</Execution>\n<Conclusion>"
        
        final_input_text = context_so_far + full_execution
        final_inputs = self.tokenizer(final_input_text, return_tensors="pt").to(self.device)
        
        final_ids = self.model.generate(
            **final_inputs, 
            max_new_tokens=256, 
            stop_strings=["</Think>"], 
            tokenizer=self.tokenizer,
            do_sample=False
        )
        final_output = self.tokenizer.decode(final_ids[0], skip_special_tokens=False)
        return final_output

def load_dataset(path, limit=None):
    samples = []
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            try:
                data = json.loads(line)
                options_str = "\n".join([f"{k}. {v}" for k, v in data.get('options', {}).items()])
                fmt_q = f"{data.get('question','')}\nAnswer Choices:\n{options_str}"
                samples.append({'id': i, 'question': fmt_q, 'answer_idx': data.get('answer_idx')})
            except: continue
    return samples

def main():
    engine = HFTopologicalEngine(MODEL_PATH)
    samples = load_dataset(DATASET_PATH, limit=NUM_SAMPLES_TO_TEST)
    
    print(f"\n{'ID':<5} | {'Time (s)':<10} | {'Result Preview'}")
    print("-" * 50)
    
    for sample in samples:
        start = time.time()
        try:
            output = engine.run(sample['question'])
            cost = time.time() - start
            
            ans_match = re.search(r"Answer:\s*(.*)", output)
            ans = ans_match.group(1) if ans_match else "N/A"
            
            print(f"{sample['id']:<5} | {cost:<10.2f} | {ans[:30]}...")
        except Exception as e:
            print(f"{sample['id']:<5} | Error | {e}")

if __name__ == "__main__":
    main()