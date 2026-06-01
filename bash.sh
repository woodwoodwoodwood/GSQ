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

python /usr/local/app/GSQ/scripts/benchmark_throughput.py \
  --host 127.0.0.1 \
  --port 8902 \
  --backend openai-chat \
  --dataset_name customize \
  --dataset_path /usr/local/app/GSQ/benchmark/benchmark_input.csv \
  --model /data1/models/Qwen3-30B-A3B-Instruct-2507 \
  --stream \
  --concurrency 8 \
  --request_rate inf \
  --max_new_tokens 128 \
  --prompt_num 96 \
  --perf_csv /usr/local/app/GSQ/benchmark/perf_throughput.csv


mkdir -p /usr/local/app/GSQ/benchmark/vllm_bench

for c in 8 16 24 32 48 64; do
  for r in 8 16 24 32 48 64 96 128 inf; do
    vllm bench serve \
      --backend openai-chat \
      --base-url http://127.0.0.1:8902 \
      --endpoint /v1/chat/completions \
      --dataset-name random \
      --input-len 1024 \
      --output-len 128 \
      --num-prompts 512 \
      --max-concurrency $c \
      --request-rate $r \
      --num-warmups 16 \
      --save-result \
      --result-dir /usr/local/app/GSQ/benchmark/vllm_bench \
      --result-filename c${c}_r${r}.json \
      --disable-tqdm
  done
done
