#!/usr/bin/env bash
set -euo pipefail

# vLLM 在线吞吐压测扫参脚本（openai-chat）
# 默认参数与你提供的命令一致，可按需通过环境变量覆盖。

BASE_URL="${BASE_URL:-http://127.0.0.1:8902}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-512}"
NUM_WARMUPS="${NUM_WARMUPS:-16}"
RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench}"

# 扫描网格
CONCURRENCY_LIST=(${CONCURRENCY_LIST:-8 16 24 32 48 64})
REQUEST_RATE_LIST=(${REQUEST_RATE_LIST:-8 16 24 32 48 64 96 128 inf})

mkdir -p "${RESULT_DIR}"

echo "[INFO] Result dir: ${RESULT_DIR}"
echo "[INFO] Base URL: ${BASE_URL}"

# 兼容两种调用方式
if command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
else
  BENCH_CMD=(python -m vllm.entrypoints.cli.main bench serve)
fi

echo "[INFO] Bench command: ${BENCH_CMD[*]}"

for c in "${CONCURRENCY_LIST[@]}"; do
  for r in "${REQUEST_RATE_LIST[@]}"; do
    OUT_JSON="${RESULT_DIR}/c${c}_r${r}.json"
    echo "[RUN] concurrency=${c}, request_rate=${r}, output=${OUT_JSON}"

    "${BENCH_CMD[@]}" \
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
      --save-result \
      --result-dir "${RESULT_DIR}" \
      --result-filename "c${c}_r${r}.json" \
      --disable-tqdm
  done
done

echo "[DONE] Benchmark sweep completed."
