# model_name="xwc216/Qwen2.5-7B-32step"
model_name="Qwen/Qwen2.5-7B"
data_name="aime2025"  # aime, aime2025, amc

PT=$1
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES='0,1,2,3' \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python eval.py \
--model_name_or_path $model_name  \
--data_name $data_name \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1024 \
--split "test" \
--max_tokens 8192 \
--seed 0 \
--top_p 0.95 \
--surround_with_messages \

# "/home/smzhang/work25/verl/checkpoints/koop/Qwen2.5-3B-math-baseline/global_step_200/hf_merged"
# "/home/smzhang/work25/verl/checkpoints/koop/Qwen2.5-3B-math-0.5/global_step_200/hf_merged"
# "Qwen/Qwen2.5-3B"

python cal_passk_with_saved_answers.py \
--model_name $model_name \
--data_name $data_name \
--roll 1024 \
--data_dir ./data