import json
from tqdm import tqdm
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
import os

os.environ["HF_HUB_CACHE"] = "/workdir/nlp/hf_cache"
os.environ["HF_TOKEN"] = "TO REPLACE"

dataset = json.load(open(f"./python_code_instructions_18k_alpaca.json", "r"))

start_idx = 6000 # TO replace
final_idx = min(start_idx + 2000, len(dataset))
dataset = dataset[start_idx:final_idx]
print(f"Starting from index {start_idx} to index {final_idx}")

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    dtype=torch.bfloat16, 
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    max_model_len=8128,
    enable_prefix_caching=True,
)
params = SamplingParams(
    max_tokens=8128,
    temperature=0.6, 
)

new_prompts = []

for data in dataset:
    prompt = f"Here is an instruction and Python code that solves that instruction. Translate the Python code to R code.\n\nInstruction: {data['instruction']}\n\nExample input: {data['input']}\n\Python code:\n\n```python\n{data['output']}\n```"
    messages = [
        {"role": "user", "content": prompt},
    ]
    new_prompts.append(messages)

batch_size = 16
all_responses = []
for i in tqdm(range(0, len(new_prompts), batch_size)):
    end_idx = min(i + batch_size, len(new_prompts))
    messages = new_prompts[i:end_idx]

    responses = llm.chat(messages, params, use_tqdm=False)
    responses_text = []
    for response in responses:
        responses_text.append(response.outputs[0].text)

    all_responses += responses_text


for i, data in enumerate(tqdm(dataset)):
    data['response'] = all_responses[i]

json.dump(dataset, open(f"./translated_datasets/python_code_instructions_18k_alpaca_translated_vllm_{start_idx}_{final_idx}.json", "w"), indent=4)
