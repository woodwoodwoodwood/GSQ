# vllm 启动 GPTQ-Int4 模型
CUDA_VISIBLE_DEVICES=6 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
    --trust-remote-code \
    --quantization gptq \
    --dtype float16 \
    --host 127.0.0.1 \
    --port 8900 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --generation-config vllm

CUDA_VISIBLE_DEVICES=6 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
  --trust-remote-code \
  --quantization gptq_marlin \
  --dtype float16 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --generation-config vllm

CUDA_VISIBLE_DEVICES=6 vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
    --trust-remote-code \
    --quantization gptq_marlin \
    --dtype float16 \
    --host 127.0.0.1 \
    --port 8900 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --generation-config vllm \
    --disable-log-requests \
    --disable-uvicorn-access-log

# 压测 benchmark vllm 0.9.0
cd /usr/local/app/GSQ && http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY= NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost BASE_URL="http://127.0.0.1:8900" ENDPOINT="/v1/chat/completions" DATASET_NAME="random" RANDOM_INPUT_LEN=1024 RANDOM_OUTPUT_LEN=128 NUM_PROMPTS=128 NUM_WARMUPS=8 CONCURRENCY_LIST="1 8 16 24 32 40 48 56 64" BENCH_MODEL="/data1/models/Qwen3-30B-A3B-GPTQ-Int4" TOKENIZER="/data1/models/Qwen3-30B-A3B-Instruct-2507" bash /usr/local/app/GSQ/scripts/run_concurrency_sweep_vllm090.sh --model /data1/models/Qwen3-30B-A3B-GPTQ-Int4



# 压测 benchmark vllm 0.21.0
cd /usr/local/app/GSQ

http_proxy= \
https_proxy= \
HTTP_PROXY= \
HTTPS_PROXY= \
all_proxy= \
ALL_PROXY= \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
VENV_PATH="/usr/local/app/GSQ/.venv" \
BASE_URL="http://127.0.0.1:8900" \
ENDPOINT="/v1/chat/completions" \
BACKEND="openai-chat" \
DATASET_NAME="random" \
INPUT_LEN=1024 \
OUTPUT_LEN=128 \
NUM_PROMPTS=128 \
NUM_WARMUPS=8 \
TEMPERATURE=0 \
CONCURRENCY_LIST="1 8 16 24 32 40 48 56 64" \
REQUEST_RATE_LIST="inf" \
bash /usr/local/app/GSQ/scripts/run_concurrency_sweep_simple.sh \
  --model /data1/models/Qwen3-30B-A3B-GPTQ-Int4



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

# 1) 启动服务（示例：humming）
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
  --trust-remote-code \
  --quantization humming \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode hf \
  --max-num-seqs 32 \
  --generation-config vllm


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

CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507 \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --tokenizer-mode hf \
  --max-num-seqs 32 \
  --generation-config vllm
