import torch
import time
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFSerialEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading HF Serial Engine: {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

    async def run(self, question):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"Question:\n{question}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"<Think>\n"
        )
        return await asyncio.to_thread(self._generate_sync, prompt)

    def _generate_sync(self, prompt):
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, time.time() - start_time