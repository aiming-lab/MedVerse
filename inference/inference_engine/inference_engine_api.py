import asyncio
import re
import networkx as nx
from openai import AsyncOpenAI
from collections import deque
import time
import json
import os
from inference_engine_hf import HFTopologicalEngine
from inference_engine_hfs import HFSerialEngine

VLLM_API_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "/data/LLaMA-3.1-8B/Multiverse-20251204/checkpoint-100"
DATASET_PATH = "/home/ubuntu/Med-Reason/MedReason/eval_data/HLE_biomed.jsonl"
NUM_SAMPLES_TO_TEST = 2

aclient = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

# ==========================================
# Plan Parser
# ==========================================
class PlanParser:
    def __init__(self):
        # Strict match for training data format:
        # <Outline> Transient Step X: ...; Dependency: [...] </Outline>
        self.item_pattern = re.compile(
            r"<Outline>\s*(Transient Step\s+(\d+):\s*(.*?));\s*Dependency:\s*\[(.*?)\]\s*</Outline>", 
            re.DOTALL | re.IGNORECASE
        )

    def parse(self, text):
        graph = nx.DiGraph()
        # Extract content within <Plan> tags
        plan_match = re.search(r"<Plan>(.*?)(?:</Plan>|$)", text, re.DOTALL)
        if not plan_match:
            print("Warning: No <Plan> tags found in output.")
            return graph
            
        content_inner = plan_match.group(1)
        matches = self.item_pattern.findall(content_inner)
        
        for full_desc, step_id_str, step_content, dep_str in matches:
            step_id = int(step_id_str)
            deps = []
            if dep_str.strip():
                try:
                    # Parse dependencies like "1, 2"
                    deps = [int(d.strip()) for d in dep_str.split(',') if d.strip()]
                except ValueError:
                    pass
            
            graph.add_node(step_id, description=full_desc, content=step_content, result=None)
            for dep_id in deps:
                if dep_id != step_id:
                    graph.add_edge(dep_id, step_id)
        return graph

# ==========================================
# Async Graph Executor
# ==========================================
class AsyncGraphExecutor:
    def __init__(self, step_processor):
        self.processor = step_processor

    async def execute(self, graph):
        if len(graph.nodes) == 0:
            return {}

        in_degree = {u: d for u, d in graph.in_degree()}
        queue = deque([u for u in in_degree if in_degree[u] == 0])
        running_tasks = {} 
        results = {}       

        while queue or running_tasks:
            while queue:
                node_id = queue.popleft()
                node_data = graph.nodes[node_id]
                
                preds = list(graph.predecessors(node_id))
                preds.sort()
                context_results = [results[p] for p in preds]
                
                task = asyncio.create_task(
                    self.processor(node_id, node_data['description'], context_results)
                )
                running_tasks[node_id] = task

            if not running_tasks:
                break

            # Wait for any task to complete
            done, pending = await asyncio.wait(
                running_tasks.values(), 
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                finished_node_id = [k for k, v in running_tasks.items() if v == task][0]
                del running_tasks[finished_node_id]
                
                try:
                    res = task.result()
                    results[finished_node_id] = res
                    
                    # Unlock successors
                    for neighbor in graph.successors(finished_node_id):
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)
                except Exception as e:
                    print(f"Error executing Step {finished_node_id}: {e}")
                    results[finished_node_id] = f"<Step> Transient Step {finished_node_id}: Execution Failed. Error: {str(e)} </Step>"

        return results

# ==========================================
# Hybrid Inference Engine (vLLM Parallel)
# ==========================================
class HybridInferenceEngine:
    def __init__(self):
        self.parser = PlanParser()
        self.executor = AsyncGraphExecutor(self.step_processor)
        self.current_base_prompt = "" 
        self.current_plan_block = ""   

    def _build_header(self, question):
        """Construct LLaMA-3 formatted header."""
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"Question:\n{question}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"<Think>\n"
        )

    async def step_processor(self, node_id, full_step_desc, context_results):
        """
        Process a single step. 
        Constructs a prompt simulating sequential history for the model.
        """
        execution_context = "\n".join(context_results)
        
        # Construct Prompt: Header -> Plan -> Execution History -> Current Step
        prompt = (
            f"{self.current_base_prompt}"
            f"{self.current_plan_block}\n"
            f"<Execution>\n"
            f"{execution_context}\n"
            f"<Step> {full_step_desc}"
        )
        
        try:
            response = await aclient.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                stop=["</Step>"], 
                max_tokens=512,
                temperature=0.7,
                extra_body={"skip_special_tokens": False}
            )
            content = response.choices[0].text
            return f"<Step> {full_step_desc}{content}</Step>"
        except Exception as e:
            return f"<Step> {full_step_desc} Error: {str(e)}</Step>"

    async def run(self, question):
        self.current_base_prompt = self._build_header(question)
        
        # --- Phase 1: Planning ---
        # Generate the reasoning path and plan
        try:
            plan_response = await aclient.completions.create(
                model=MODEL_NAME,
                prompt=self.current_base_prompt,
                stop=["</Plan>"], 
                max_tokens=512,
                temperature=0.1,
                extra_body={"skip_special_tokens": False} 
            )
            generated_text = plan_response.choices[0].text
            self.current_plan_block = f"{generated_text}</Plan>"
        except Exception as e:
            return f"Planning Failed: {e}"

        # --- Phase 2: Parsing ---
        graph = self.parser.parse(self.current_plan_block)
        if len(graph.nodes) == 0:
            return f"{self.current_plan_block}\n<Error>Failed to parse plan</Error>"

        # --- Phase 3: Parallel Execution ---
        results_dict = await self.executor.execute(graph)
        
        sorted_steps = [results_dict[i] for i in sorted(results_dict.keys())]
        full_execution_block = "<Execution>\n" + "\n".join(sorted_steps) + "\n</Execution>"

        # --- Phase 4: Conclusion ---
        # Synthesize final answer based on execution history
        final_prompt = (
            f"{self.current_base_prompt}"
            f"{self.current_plan_block}\n"
            f"{full_execution_block}\n"
            f"<Conclusion>"
        )
        
        conclusion_response = await aclient.completions.create(
            model=MODEL_NAME,
            prompt=final_prompt,
            stop=["</Think>"], 
            max_tokens=512,
            temperature=0.7,
            extra_body={"skip_special_tokens": False}
        )
        
        final_output = (
            f"{self.current_base_prompt}"
            f"{self.current_plan_block}\n"
            f"{full_execution_block}\n"
            f"<Conclusion>{conclusion_response.choices[0].text}</Conclusion>\n"
            f"</Think>\n" 
        )
        return final_output

# ==========================================
# Serial Baseline Engine (vLLM)
# ==========================================
class SerialBaselineEngine:
    async def run(self, question):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"Question:\n{question}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"<Think>\n"
        )
        start_time = time.time()
        response = await aclient.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            stop=["<|eot_id|>"], 
            max_tokens=2048,
            temperature=0.7
        )
        return response.choices[0].text, time.time() - start_time

def load_dataset(path, limit=None):
    samples = []
    if not os.path.exists(path):
        print(f"Dataset not found at: {path}")
        return []
        
    print(f"Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                question_text = data.get('question', '')
                options = data.get('options') or data.get('choices', {})
                options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
                
                formatted_question = f"{question_text}\nAnswer Choices:\n{options_str}"
                
                samples.append({
                    'id': data.get('idx', i),
                    'question': formatted_question,
                    'answer_idx': data.get('answer_idx', '')
                })
            except json.JSONDecodeError:
                continue
    return samples

def check_correctness(model_output, correct_label):
    if not correct_label:
        return False
    conclusion_match = re.search(r"<Conclusion>(.*?)</Conclusion>", model_output, re.DOTALL | re.IGNORECASE)
    search_content = conclusion_match.group(1) if conclusion_match else model_output
    pattern = re.compile(rf"Answer:\s*(?:Option\s*)?({correct_label})\b", re.IGNORECASE)
    if pattern.search(search_content):
        return True
    if re.search(rf"\b{correct_label}\b", search_content[-200:]): 
        return True
        
    return False

async def benchmark():
    samples = load_dataset(DATASET_PATH, limit=NUM_SAMPLES_TO_TEST)
    if not samples:
        print("No samples loaded.")
        return

    print("Initializing Engines...")
    parallel_engine = HybridInferenceEngine()
    serial_engine = SerialBaselineEngine()
    
    try:
        hf_topo_engine = HFTopologicalEngine(MODEL_NAME)
        hf_serial_engine = HFSerialEngine(MODEL_NAME)
    except Exception as e:
        print(f"Failed to load HF Engines: {e}")
        hf_topo_engine = None
        hf_serial_engine = None

    stats = {
        'par':     {'time': 0, 'corr': 0},
        'ser':     {'time': 0, 'corr': 0},
        'hf_topo': {'time': 0, 'corr': 0},
        'hf_ser':  {'time': 0, 'corr': 0}
    }
    
    print(f"\n{'ID':<5} | {'Type':<12} | {'Time (s)':<10} | {'Speedup':<10} | {'Correct?':<8}")
    print("-" * 65)

    for sample in samples:
        q_id = sample['id']
        question = sample['question']
        ground_truth = sample['answer_idx']

        # --- Run Parallel (vLLM) ---
        p_start = time.time()
        try:
            p_out = await parallel_engine.run(question)
            p_time = time.time() - p_start
            p_corr = check_correctness(p_out, ground_truth)
        except Exception as e:
            p_time, p_corr = 0, False
            print(f"Parallel Err: {e}")
        
        if p_corr: stats['par']['corr'] += 1
        stats['par']['time'] += p_time
        
        # --- Run Serial (vLLM) ---
        s_out, s_time = await serial_engine.run(question)
        s_corr = check_correctness(s_out, ground_truth)
        
        if s_corr: stats['ser']['corr'] += 1
        stats['ser']['time'] += s_time

        # --- Run HF Topological (Manual Cache) ---
        hft_time, hft_corr = 0, False
        if hf_topo_engine:
            start = time.time()
            try:
                out = await asyncio.to_thread(hf_topo_engine.run, question)
                hft_time = time.time() - start
                hft_corr = check_correctness(out, ground_truth)
            except Exception as e:
                print(f"HF Topo Err: {e}")
            if hft_corr: stats['hf_topo']['corr'] += 1
            stats['hf_topo']['time'] += hft_time

        # --- Run HF Serial (Autoregressive Baseline) ---
        hfs_time, hfs_corr = 0, False
        if hf_serial_engine:
            start = time.time()
            try:
                # HFSerialEngine already handles to_thread internally
                out, _ = await hf_serial_engine.run(question) 
                hfs_time = time.time() - start
                hfs_corr = check_correctness(out, ground_truth)
            except Exception as e:
                print(f"HF Serial Err: {e}")
            if hfs_corr: stats['hf_ser']['corr'] += 1
            stats['hf_ser']['time'] += hfs_time

        vllm_speedup = s_time / p_time if p_time > 0 else 0
        hf_speedup = hfs_time / hft_time if hft_time > 0 else 0

        print(f"{q_id:<5} | vLLM Par     | {p_time:<10.2f} | {vllm_speedup:<10.2f}x | {'✅' if p_corr else '❌'}")
        print(f"{'':<5} | vLLM Ser     | {s_time:<10.2f} | {'(Base)':<10} | {'✅' if s_corr else '❌'}")
        
        if hf_topo_engine:
            print(f"{'':<5} | HF Topo      | {hft_time:<10.2f} | {hf_speedup:<10.2f}x | {'✅' if hft_corr else '❌'}")
        if hf_serial_engine:
            print(f"{'':<5} | HF Serial    | {hfs_time:<10.2f} | {'(Base)':<10} | {'✅' if hfs_corr else '❌'}")
        print("-" * 65)

    total = len(samples)
    print("\n================ FINAL SUMMARY ================")
    print(f"Total Samples: {total}")
    print(f"{'Type':<15} | {'Avg Time':<10} | {'Accuracy':<10}")
    print("-" * 50)
    
    print(f"{'vLLM Parallel':<15} | {stats['par']['time']/total:<10.2f} | {stats['par']['corr']}/{total}")
    print(f"{'vLLM Serial':<15} | {stats['ser']['time']/total:<10.2f} | {stats['ser']['corr']}/{total}")
    
    if hf_topo_engine:
        print(f"{'HF Topological':<15} | {stats['hf_topo']['time']/total:<10.2f} | {stats['hf_topo']['corr']}/{total}")
    if hf_serial_engine:
        print(f"{'HF Serial':<15} | {stats['hf_ser']['time']/total:<10.2f} | {stats['hf_ser']['corr']}/{total}")
        
    print("-" * 50)
    if stats['par']['time'] > 0:
        print(f"vLLM Speedup (Par vs Ser): {stats['ser']['time'] / stats['par']['time']:.2f}x")
    if stats['hf_topo']['time'] > 0 and stats['hf_ser']['time'] > 0:
        print(f"HF Speedup (Topo vs Ser) : {stats['hf_ser']['time'] / stats['hf_topo']['time']:.2f}x")

if __name__ == "__main__":
    asyncio.run(benchmark())