from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import importlib.util
import os
import argparse
import vllm.envs as envs
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    
    n_sampling = args.n_sampling
    factor = 8
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     top_p=args.top_p,
                                     )
    
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    # out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
    # out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_roll{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
    
    
    # if os.path.exists(out_file):
    #     print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
    #     return
    # os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)
    # os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)
    
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        # parse question and answer
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])

    llm = LLM(model=model_name_or_path, 
              tensor_parallel_size=len(available_gpus), 
              trust_remote_code=True, 
              dtype="float16", 
            #   swap_space=32,
              gpu_memory_utilization=0.85,
              )
    
    file_outputs = []
    for cur_generation_epoch in range(generation_epoch):
        print(f"generation round {cur_generation_epoch + 1}/{generation_epoch} starts, {datetime.now()}")
     
        completions = llm.generate(prompt_batch, sampling_params)
        
        # save_completions(completions, completions_save_file)
        for i in range(len(examples)):
            d = examples[i]
            question = parse_question(d, args.data_name)
            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
            if cur_generation_epoch == 0:
                file_outputs.append({
                    "question": question,
                    "generated_responses": generated_responses,
                })
                if "id" in d:
                    file_outputs[i]["id"] = d["id"]
                if "source" in d:
                    file_outputs[i]["source"] = d["source"]
            else:
                file_outputs[i]['generated_responses'] += generated_responses
    print("llm generate done, start to check correctness...")
    print(len(file_outputs))

    # Save file_outputs as a pickle file for easy reloading
    pickle_save_dir = f"results/{model_name}"
    os.makedirs(pickle_save_dir, exist_ok=True)
    with open(f"{pickle_save_dir}/{args.data_name}.pkl", "wb") as pf:
        pickle.dump(file_outputs, pf, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
