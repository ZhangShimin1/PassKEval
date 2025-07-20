from datasets import load_dataset
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="yentinglin/aime_2025")
parser.add_argument("--data_name", type=str, default="aime2025")
args = parser.parse_args()

ds = load_dataset(args.data_path, "default")
data = ds["train"] 
n_samples = len(data)

output_dir = f"{args.data_name}/test"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/test.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for item in data.select(range(n_samples)):
        entry = {
            "id": item["id"],
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": item["answer"],
            "url": item["url"],
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved {n_samples} entries to {output_path}")
