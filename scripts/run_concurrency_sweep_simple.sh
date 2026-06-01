#!/usr/bin/env bash
set -euo pipefail

# 极简吞吐扫参脚本：只改 max-concurrency + request-rate
# 适用场景：服务参数已固定，只想快速找到当前配置下的最佳吞吐点

BASE_URL="${BASE_URL:-http://127.0.0.1:8900}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
BACKEND="${BACKEND:-openai-chat}"

DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
TEMPERATURE="${TEMPERATURE:-0}"

# 只扫这两个变量
CONCURRENCY_LIST="${CONCURRENCY_LIST:-8 16 24 32 40 48 56 64}"
REQUEST_RATE_LIST="${REQUEST_RATE_LIST:-8 16 24 32 48 64 96 128 inf}"

RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench_concurrency_only}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULT_DIR}/${RUN_TAG}"

echo "[INFO] base_url=${BASE_URL}"
echo "[INFO] concurrency_list=${CONCURRENCY_LIST}"
echo "[INFO] request_rate_list=${REQUEST_RATE_LIST}"
echo "[INFO] result_dir=${RESULT_DIR}/${RUN_TAG}"

for c in ${CONCURRENCY_LIST}; do
  for r in ${REQUEST_RATE_LIST}; do
    echo
    echo "[INFO] ===== concurrency=${c}, request_rate=${r} ====="

    vllm bench serve \
      --backend "${BACKEND}" \
      --base-url "${BASE_URL}" \
      --endpoint "${ENDPOINT}" \
      --dataset-name "${DATASET_NAME}" \
      --input-len "${INPUT_LEN}" \
      --output-len "${OUTPUT_LEN}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${c}" \
      --request-rate "${r}" \
      --num-warmups "${NUM_WARMUPS}" \
      --temperature "${TEMPERATURE}" \
      --save-result \
      --result-dir "${RESULT_DIR}/${RUN_TAG}" \
      --result-filename "c${c}_r${r}.json" \
      --disable-tqdm
  done
done

echo
echo "[DONE] Sweep finished. Results: ${RESULT_DIR}/${RUN_TAG}"
