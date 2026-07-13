# vllm 启动 GPTQ-Int4 模型
CUDA_VISIBLE_DEVICES=7 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
    --trust-remote-code \
    --quantization gptq \
    --dtype float16 \
    --host 127.0.0.1 \
    --port 8900 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --generation-config vllm \
    --no-enable-prefix-caching

CUDA_VISIBLE_DEVICES=7 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
  --trust-remote-code \
  --quantization gptq_marlin \
  --dtype float16 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
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
NUM_PROMPTS=64 \
NUM_WARMUPS=8 \
TEMPERATURE=0 \
CONCURRENCY_LIST="1 8 16 24 32 40 48 56 64" \
REQUEST_RATE_LIST="inf" \
bash /usr/local/app/GSQ/scripts/run_concurrency_sweep_simple.sh \
  --model /data1/models/Qwen3-30B-A3B-GPTQ-Int4

for L in 8192 16384 25600 30720 35840 46080 76800 102400; do
  echo "===== context = $L ====="
  python3 benchmark_throughput.py --host localhost --port 8080 --backend openai-chat \
    --tokenizer_path /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
    --model_type qwen --mode sync \
    --repeat_to_tokens $L \
    --max_new_tokens 128 --ignore_eos \
    --prompt_num 1 \
    --request_rate inf --stream --quiet \
    --perf_csv longctx_${L}.csv
done



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
  --gpu-memory-utilization 0.95 \
  --tokenizer-mode hf \
  --max-num-seqs 128 \
  --generation-config vllm

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  curl -s --noproxy '*' http://127.0.0.1:8900/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data1/models/qwen36-35b-a3b-2bit-humming",
    "prompt": "你好，请介绍一下你自己",
    "max_tokens": 50,
    "temperature": 0
  }'

env -u http_proxy -u https_proxy curl -s --noproxy '*' http://127.0.0.1:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05
  }'

env -u http_proxy -u https_proxy curl -s http://127.0.0.1:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"/data1/models/qwen36-35b-a3b-2bit-humming/",
    "messages":[{"role":"user","content":"你好，介绍一下量子物理"}],
    "max_tokens":512,
    "chat_template_kwargs":{"enable_thinking":false}
  }' \
  | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"



# 测试数据集
export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,piqa \
  --model_args model=/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming,base_url=http://127.0.0.1:8900/v1/completions,num_concurrent=8,tokenizer=/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code

export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,leaderboard_gpqa_main,arc_challenge,arc_easy,gsm8k,piqa,winogrande,mmlu \
  --model_args model=/data1/models/DeepSeek-V4-Flash,base_url=http://127.0.0.1:8900/v1/completions,num_concurrent=8,tokenizer=/data1/models/DeepSeek-V4-Flash \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/DeepSeek-V4-Flash/evals \
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
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --tokenizer-mode hf \
  --max-num-seqs 32 \
  --generation-config vllm


# nsys profile
mkdir -p /usr/local/app/GSQ/outputs/nsys

# profile humming 2bit
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --show-output=true \
  --output=/usr/local/app/GSQ/outputs/nsys/humming_2bit \
  /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
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

# profile gptq_marlin 4bit
mkdir -p /usr/local/app/GSQ/outputs/nsys

CUDA_VISIBLE_DEVICES=7 nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --force-overwrite=true \
  --show-output=true \
  --output=/usr/local/app/GSQ/outputs/nsys/marlin_4bit \
  vllm serve /data1/models/Qwen3-30B-A3B-GPTQ-Int4 \
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


# gptq + gsq quantization training
# single GPU
cd /usr/local/app/GSQ

CUDA_VISIBLE_DEVICES=7 \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
/usr/local/app/GSQ/.venv/bin/python /usr/local/app/GSQ/main.py \
  --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml \
  2>&1 | tee /usr/local/app/GSQ/outputs/train_single_$(date +%Y%m%d_%H%M%S).log

# multi GPU
GSQ_ROUTE_DEBUG=1 GSQ_ROUTE_DEBUG_INTERVAL=20 CUDA_VISIBLE_DEVICES=6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GSQ_PROFILE_DEQUANT=1 PYTHONUNBUFFERED=1 /usr/local/app/GSQ/.venv/bin/torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29517 /usr/local/app/GSQ/main.py --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

torchrun --standalone --nproc_per_node=4 /usr/local/app/GSQ/main.py \
  --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml \
  2>&1 | tee /usr/local/app/GSQ/outputs/train_4gpu_$(date +%Y%m%d_%H%M%S).log

GSQ_ROUTE_DEBUG=1 \
GSQ_ROUTE_DEBUG_INTERVAL=20 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GSQ_PROFILE_DEQUANT=1 \
PYTHONUNBUFFERED=1 \
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_BLOCKING_WAIT=1 \
/usr/local/app/GSQ/.venv/bin/torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29517 \
  /usr/local/app/GSQ/main.py \
  --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml

WANDB_MODE=offline \
GSQ_ROUTE_DEBUG=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GSQ_PROFILE_DEQUANT=0 \
PYTHONUNBUFFERED=1 \
NCCL_DEBUG=WARN \
TORCH_DISTRIBUTED_DEBUG=OFF \
NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_BLOCKING_WAIT=1 \
/usr/local/app/GSQ/.venv/bin/torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29517 \
  /usr/local/app/GSQ/main.py \
  --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml

WANDB_MODE=offline GSQ_ROUTE_DEBUG=0 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  /usr/local/app/GSQ/.venv/bin/torchrun \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29517 \
    /usr/local/app/GSQ/main.py \
    --config /usr/local/app/GSQ/configs/deepseek_v4/deepseek_v4_flash_2bit.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
NCCL_ASYNC_ERROR_HANDLING=1 \
/usr/local/app/GSQ/.venv/bin/torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29517 \
  /usr/local/app/GSQ/main.py \
  --config /usr/local/app/GSQ/configs/qwen36/qwen36_35B_A3B.yaml


# 1. main.py -> GPTQ + GSQ quantization training
# 2. save_quantized_model.py -> save quantized model
/usr/local/app/GSQ/.venv/bin/python save_model.py \
    --config configs/deepseek_v4/deepseek_v4_flash_2bit.yaml \
    --run-id 20260609-181157_fcc6b7
# 3. convert_to_humming.py -> convert to humming
/usr/local/app/GSQ/.venv/bin/python convert_to_humming.py \
    --in-dir /usr/local/app/GSQ/checkpoints/deepseek-v4-flash-2bit/20260609-181157_fcc6b7/assembled \
    --out-dir /data1/models/deepseek-v4-flash-2bit-humming \
    --symmetric \
    --target-dtype bfloat16

# fix vllm0.21.0 for humming quant
# cp /usr/local/app/vllm-0.21.0-fresh/vllm/model_executor/layers/fused_moe/layer.py \
#    /usr/local/app/GSQ/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/layer.py

# cp /usr/local/app/vllm-0.21.0-fresh/vllm/model_executor/layers/quantization/humming.py \
#    /usr/local/app/GSQ/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/humming.py

# cp /usr/local/app/vllm-0.21.0-fresh/vllm/model_executor/layers/quantization/utils/humming_utils.py \
#    /usr/local/app/GSQ/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/humming_utils.py

# 
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  --output /usr/local/app/GSQ/vllm_decode_%p \
  --force-overwrite true \
  --trace cuda,nvtx,osrt,cublas \
  --sample cpu \
  --python-sampling true \
  --delay 140 \
  --duration 40 \
  /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
    --trust-remote-code --quantization humming --tensor-parallel-size 1 \
    --host 127.0.0.1 --port 8900 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 32 \
    --generation-config vllm

CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  --output /usr/local/app/GSQ/vllm_decode_bs128_%p \
  --force-overwrite true \
  --trace cuda,nvtx,osrt,cublas \
  --cuda-graph-trace node \
  --sample cpu \
  --python-sampling true \
  --delay 140 \
  --duration 40 \
  /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming \
    --trust-remote-code --quantization humming --tensor-parallel-size 1 \
    --host 127.0.0.1 --port 8900 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 128 \
    --generation-config vllm

# GPTQ + GSQ 量化训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  CONFIG_FILE=configs/qwen36/qwen36_35B_A3B.yaml \
  RESUME=latest bash scripts/run.sh

# thinking 模型 eval
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,leaderboard_gpqa_main,arc_challenge,arc_easy,gsm8k,piqa,winogrande,mmlu \
  --model_args model=/data1/models/qwen36-35b-a3b-2bit-humming,base_url=http://127.0.0.1:8900/v1/completions,num_concurrent=8,tokenizer=/data1/models/qwen36-35b-a3b-2bit-humming,max_length=8192 \
  --gen_kwargs 'temperature=0,seed=42,max_gen_toks=4096,until=<|im_end|>' \
  --output_path /data1/models/qwen36-35b-a3b-2bit-humming/evals \
  --log_samples --trust_remote_code --confirm_run_unsafe_code

# qwen3.6 35B A3B 2bit humming 模型部署 + tool call
CUDA_VISIBLE_DEVICES=5 VLLM_LOGGING_LEVEL=INFO FLASHINFER_DISABLE_VERSION_CHECK=1 /usr/local/app/GSQ/.venv/bin/vllm serve /data1/models/qwen36-35b-a3b-2bit-humming \
  --trust-remote-code \
  --quantization humming \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8900 \
  --max-model-len 256000 \
  --gpu-memory-utilization 0.95 \
  --tokenizer-mode hf \
  --max-num-seqs 128 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3

sudo socat TCP-LISTEN:8080,fork,reuseaddr TCP:127.0.0.1:8900 & 

claude --dangerously-skip-permissions

# llamacpp 启动推理服务
CUDA_VISIBLE_DEVICES=4 /usr/local/app/llama.cpp/build/bin/llama-cli   \
  -m /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/qwen36-35b-a3b-IQ2_M.gguf   \
  -c 8192 \
  -n 4096 \
  -t 8 \
  -ngl 99 \
  -p "请介绍一下你自己"

CUDA_VISIBLE_DEVICES=4 /usr/local/app/llama.cpp/build/bin/llama-server \
  -m /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/qwen36-35b-a3b-IQ2_M.gguf \
  -c 81920 \
  -t 8 \
  -ngl 99 \
  --port 8901 \
  --host 0.0.0.0 \
  -np 32 \
  -cb

CUDA_VISIBLE_DEVICES=4 python -m llama_cpp.server \
  --model /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/qwen36-35b-a3b-IQ2_M.gguf \
  --n_gpu_layers 99 \
  --n_ctx 8192 \
  --port 8901 \
  --host 0.0.0.0

# 评测 llamacpp 部署的模型
export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks hellaswag,leaderboard_gpqa_main \
  --model_args model=/data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF,base_url=http://127.0.0.1:8901/v1/completions,num_concurrent=8,tokenizer=/data1/models/Qwen3.6-35B-A3B \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code


unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export HF_ALLOW_CODE_EVAL=1
/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model gguf \
  --tasks hellaswag,leaderboard_gpqa_main,arc_challenge,arc_easy,gsm8k,piqa,winogrande,mmlu \
  --model_args base_url=http://127.0.0.1:8901,max_length=4096 \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code

/usr/local/app/GSQ/.venv/bin/python -m lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args model=/data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF,base_url=http://127.0.0.1:8901/v1/completions,num_concurrent=8,tokenizer=/data1/models/Qwen3.6-35B-A3B \
  --gen_kwargs temperature=0,seed=42 \
  --output_path /data1/models/Qwen3.6-35B-A3B-IQ2_M-GGUF/evals \
  --log_samples \
  --trust_remote_code \
  --confirm_run_unsafe_code

# llama 框架性能测试
CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-bench -m /data1/models/qwen36-35b-a3b-gsq2.gguf -ngl 99 -p 512 -n 128
# llama 对话
CUDA_VISIBLE_DEVICES=6 /usr/local/app/llama.cpp/build/bin/llama-cli \
  -m /data1/models/qwen36-35b-a3b-gsq2.gguf \
  -c 8192 \
  -n 4096 \
  -t 8 \
  -ngl 99 \
  -cnv
# llama-bench 单测性能
CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-bench \
  -m /data1/models/qwen36-35b-a3b-gsq2.gguf \
  -ngl 99 \
  -p 8192,16384,25000,30000,35000,45000,75000,100000 \
  -n 4096 \
  -t 8 \
  -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -fa

CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-server \
  -m /data1/models/qwen36-35b-a3b-gsq2.gguf \
  -c 110000 -ngl 99 -t 8 \
  --port 8901 --host 0.0.0.0 \
  -ctk q8_0 -ctv q8_0 -fa on

