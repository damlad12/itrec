import argparse
import json
import re
import jsonlines
from fractions import Fraction
from vllm import LLM, SamplingParams

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
import sys
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

MAX_INT = sys.maxsize

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def preprocess_data(example):
    return {
        "instruction": example.get("user", ""),  # Use the 'user' column as instruction
        "input": example.get("system", ""),      # Use the 'system' column as input
        "output": example.get("assistant", "")   # Use the 'assistant' column as output
    }
    
def flatten_example(example):
            flattened = {}
            if "messages" in example:
                for message in example["messages"]:
                    role = message.get("role", "unknown")
                    content = message.get("content", [])
                    # Combine all content into a single string (if content is a list of dicts)
                    content_text = " ".join(item.get("content", "") for item in content)
                    flattened[role] = content_text
            return flattened
            
            
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

root_path = '/content/drive/MyDrive/RecRanker/RecRanker/conventional recommender/dataset/ml-100k_data_generated_for_RecRanker_training_and_testing/inference_results'
data_path = f'/content/drive/MyDrive/RecRanker/conventional recommender/dataset/ml-100k_data_generated_for_RecRanker_training_and_testing/test_ml-100k_MF_pointwise.jsonl'
INVALID_ANS = "[invalid]"
res_ins = []
res_answers = []
problem_prompt = (
        "{instruction}"
    )
    
def main(model_path: str ,lora_weights: str = "tloen/alpaca-lora-7b", start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, share_gradio: bool = False,):
    start = 0
    end = MAX_INT  
    
    data_path = f'/content/drive/MyDrive/RecRanker/RecRanker/conventional recommender/dataset/ml-100k_data_generated_for_RecRanker_training_and_testing/test_ml-100k_MF_pointwise.jsonl'
    base_model = f'/content/drive/MyDrive/Final_project/llama-7b'
    #tokenizer = LlamaTokenizer.from_pretrained(model)
    lora_weights = '/content/drive/MyDrive/Final_project/ml-100k'
    
    load_8bit = False
    
    
    stop_tokens = []
    sampling_params = SamplingParams(temperature=0.1, top_k=10, top_p=0.1, max_tokens=300,
                                     stop=stop_tokens)  # stop=stop_tokens
                                     
    
    print('sampleing =====', sampling_params)
    #llm = LLM(model, tensor_parallel_size=tensor_parallel_size)
    llm = LLM(model=model_path, enable_lora=True)
    

    for kkk in ['MF']:  # , 'SASRec','BERT4Rec','CL4SRec''SGL','MF', 'LightGCN', 'SGL',
        INVALID_ANS = "[invalid]"
        res_ins = []
        res_answers = []
        problem_prompt = (
            "{instruction}"
        )
        with open(data_path, "r+", encoding="utf8") as f:
            reader = jsonlines.Reader(f)
            for idx, item in enumerate(reader):
                flattened = flatten_example(item)
                if not flattened:
                    # Assuming the original structure contains an 'inst' field
                    flattened = {
                        "user": item.get("inst", ""),
                        "system": "",        # No system input available
                        "assistant": ""      # No assistant output available
                    }
                
                # Step 2: Preprocess the flattened data
                preprocessed = preprocess_data(flattened)
                
                # Step 3: Generate the prompt using the preprocessed data
                problem_prompt = generate_prompt(preprocessed)
                temp_instr = problem_prompt.format(instruction=item["inst"])
                res_ins.append(temp_instr)
        print('res_ins', res_ins)
        res_ins = res_ins[start:end]
        res_answers = res_answers[start:end]
        print('lenght ====', len(res_ins))
        batch_res_ins = batch_data(res_ins, batch_size=batch_size)
        result = []
        res_completions = []
        idx = 0
        for prompt_batch in batch_res_ins:
            
            completions = llm.generate(
                    prompt_batch,
                    sampling_params,
                    lora_request=LoRARequest("weights_adapter", 1, lora_weights)
                )
            for output in completions:
                    local_idx = 'INDEX ' + str(idx) + ':'
                    generated_text = output.outputs[0].text
                    generated_text = generated_text.replace('\n', '').replace('    ', '')
                    generated_text = local_idx + generated_text
                    res_completions.append(generated_text)
                    idx += 1
        print('res_completions', res_completions[:])
        
        
        
        def write_list_to_file(string_list, output_file):
            with open(output_file, 'w') as file:
                for item in string_list:
                    file.write(item + '\n')
        import pandas as pd
        df = pd.DataFrame(res_completions)
        df.to_csv(f'/content/drive/MyDrive/llmres_MF_ml-100k_pointwise.txt', index=None, header=None)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(main)
