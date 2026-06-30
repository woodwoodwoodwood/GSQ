# vllm 启动 deepseek-v4-flash
# 4 * H20
CUDA_VISIBLE_DEVICES=4,5,6,7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
/usr/local/app/GSQ/.venv/bin/vllm serve \
  /data1/models/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enforce-eager