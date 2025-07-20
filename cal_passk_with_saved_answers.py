from math import comb
import pickle
import argparse
from utils.math_normalization import *
from utils.grader import *
from utils.parser import *
from utils.data_loader import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="math")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument("--roll", type=int, default=128)
parser.add_argument("--data_dir", default="./data", type=str)
args = parser.parse_args()

examples = load_data(args.data_name, "test", args.data_dir)

# We only use 1024 and 128 at present
if args.roll == 1024:
    k_values = [1, 4, 16, 64, 256, 1024]
elif args.roll == 128:
    k_values = [1, 4, 16, 32, 64, 128]
else:
    raise ValueError(f"Invalid roll value: {args.roll}")

correct_cnt = 0
test_sets_correct_cnt = []

with open(f"results/{args.model_name}/{args.data_name}.pkl", "rb") as f:
    generated_outputs = pickle.load(f)

# print(type(generated_outputs), len(generated_outputs), generated_outputs[0])

pass_at_k_dict = {}
for k in k_values:
    pass_at_k_dict[k] = []

for i in range(len(examples)):
    d = examples[i]
    gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
    
    generated_responses = generated_outputs[i]['generated_responses']
    generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
    is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
    print(f"problem {i} gets correct for {sum(is_correct_list)}/{len(is_correct_list)}")
    is_correct = any(is_correct_list)
    if is_correct:
        correct_cnt += 1
    test_sets_correct_cnt.append(is_correct)
        
    for k in k_values:
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_dict[k].append(pass_at_k)
            else:
                pass_at_k_dict[k].append(0)
# print(pass_at_k_dict)
print(f"Accuracy: {correct_cnt / len(examples)}")
for k in k_values:
    average_pass_at_k = sum(pass_at_k_dict[k]) / len(pass_at_k_dict[k])
    print(f"Pass@{k}: {sum(pass_at_k_dict[k])}/{len(pass_at_k_dict[k])} = {average_pass_at_k:.4f}")

