from transformers import AutoConfig
print(AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True))
