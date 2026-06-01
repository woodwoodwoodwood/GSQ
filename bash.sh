# vllm 启动 GPTQ-Int4 模型
CUDA_VISIBLE_DEVICES=6 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 --trust-remote-code --quantization gptq --dtype float16 --port 8900 --max-model-len 4096

# 测试数据集
export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,leaderboard_gpqa_main \
  --model_args model=/data1/models/Qwen3-30B-A3B-GPTQ-Int4,base_url=http://127.0.0.1:8900/v1/completions,num_concurrent=8,tokenizer=/data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3-30B-A3B-GPTQ-Int4/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code

export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python eval_model.py \
  --model-path /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
  --base-url http://127.0.0.1:9001/v1/completions \
  --tasks hellaswag,humaneval,leaderboard_gpqa_main \
  --num-concurrent 8 \
  --no-wandb

# vllm 启动 2bit-humming 模型
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming --trust-remote-code --quantization humming --tensor-parallel-size 1 --host 127.0.0.1 --port 8901 --max-model-len 4096 --gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 32
# CUDA_VISIBLE_DEVICES=6 FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONPATH=/usr/local/app/vllm-0.21.0-fresh /usr/local/app/GSQ/.venv/bin/python -m vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming --trust-remote-code --quantization humming --tensor-parallel-size 1 --host 127.0.0.1 --port 8900 --max-model-len 4096 --gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 32

# 测试数据集
export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,leaderboard_gpqa_main \
  --model_args model=/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming,base_url=http://127.0.0.1:8901/v1/completions,num_concurrent=8,tokenizer=/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code


# vllm 启动 fp16 模型
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507 --trust-remote-code --dtype float16 --tensor-parallel-size 1 --host 127.0.0.1 --port 8902 --max-model-len 4096 --gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 32

# 吞吐测试
MODEL_ID=$(curl -sS http://127.0.0.1:8902/v1/models | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "MODEL_ID=${MODEL_ID}"

vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8902 \
  --model "${MODEL_ID}" \
  --dataset-name random \
  --num-prompts 500 \
  --input-len 1024 \
  --output-len 128 \
  --request-rate inf \
  --max-concurrency 32 \
  --temperature 0 \
  --tokenizer /data1/models/Qwen3-30B-A3B-Instruct-2507 \
  --tokenizer-mode hf \
  --save-result \
  --result-dir /data1/models/Qwen3-30B-A3B-Instruct-2507/evals

# 自定义压测（固定并发，稳定版）
MODEL_ID=$(curl -sS http://127.0.0.1:8902/v1/models | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "MODEL_ID=${MODEL_ID}"

python /usr/local/app/GSQ/scripts/benchmark_throughput.py \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8902 \
  --dataset_path /usr/local/app/GSQ/benchmark/benchmark_input.csv \
  --col_idx 0 \
  --model "${MODEL_ID}" \
  --mode async \
  --stream \
  --concurrency 32 \
  --request_rate inf \
  --prompt_num 96 \
  --max_new_tokens 128 \
  --temperature 0 \
  --warmup_num_iters 1 \
  --repeat_num_iters 2 \
  --percentiles 50 90 95 99 \
  --perf_csv /tmp/fp16_perf_c32.csv \
  --output_csv /tmp/fp16_output_c32.csv

# 自定义压测（并发扫点）
for C in 8 16 32 64; do
  python /usr/local/app/GSQ/scripts/benchmark_throughput.py \
    --backend openai-chat \
    --host 127.0.0.1 \
    --port 8902 \
    --dataset_path /usr/local/app/GSQ/benchmark/benchmark_input.csv \
    --col_idx 0 \
    --model "${MODEL_ID}" \
    --mode async \
    --stream \
    --concurrency ${C} \
    --request_rate inf \
    --prompt_num 96 \
    --max_new_tokens 128 \
    --temperature 0 \
    --warmup_num_iters 1 \
    --repeat_num_iters 1 \
    --percentiles 50 90 95 99 \
    --perf_csv /tmp/fp16_perf_c${C}.csv \
    --output_csv /tmp/fp16_output_c${C}.csv
done

# 长序列（Agent场景）服务：注意 max-model-len 需覆盖 128k
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507 --trust-remote-code --dtype float16 --tensor-parallel-size 1 --host 127.0.0.1 --port 8903 --max-model-len 131072 --gpu-memory-utilization 0.92 --tokenizer-mode hf --max-num-seqs 4

# 长序列压测（1k/4k/16k/32k/64k/128k）
# 说明：长上下文显存压力很高，建议低并发（1/2/4）
python /usr/local/app/GSQ/scripts/benchmark_speed.py \
  --base-url http://127.0.0.1:8903 \
  --model /data1/models/Qwen3-30B-A3B-Instruct-2507 \
  --input-len 1024 4096 16384 32768 65536 131072 \
  --output-len 256 \
  --concurrency 1 2 4 \
  --num-prompts 8 \
  --warmup 1 \
  --output-json /tmp/fp16_longctx_agent.json

python /usr/local/app/GSQ/scripts/benchmark_throughput.py \
  --host 127.0.0.1 \
  --port 8902 \
  --backend openai-chat \
  --dataset_name customize \
  --dataset_path /usr/local/app/GSQ/benchmark/benchmark_input.csv \
  --model_type empty \
  --stream \
  --concurrency 8 \
  --request_rate inf \
  --max_new_tokens 128 \
  --prompt_num 96 \
  --perf_csv /usr/local/app/GSQ/benchmark/perf_throughput.csv
