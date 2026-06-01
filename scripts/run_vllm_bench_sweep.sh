#!/usr/bin/env bash
set -euo pipefail

# 固定参数吞吐测试脚本（仅跑一组）
# 目标参数（已确认）：
# - backend=openai-chat
# - base-url=http://127.0.0.1:8902
# - endpoint=/v1/chat/completions
# - dataset-name=random
# - input-len=1024
# - output-len=128
# - num-prompts=128
# - max-concurrency=32
# - request-rate=inf
# - num-warmups=8

BASE_URL="${BASE_URL:-http://127.0.0.1:8902}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
TEMPERATURE="${TEMPERATURE:-0}"
RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench_fixed}"
RESULT_FILENAME="${RESULT_FILENAME:-fixed_c32_rinf.json}"
VENV_PATH="${VENV_PATH:-/usr/local/app/GSQ/.venv}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] venv not found: ${VENV_PATH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
mkdir -p "${RESULT_DIR}"

if command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
else
  BENCH_CMD=(python -m vllm.entrypoints.cli.main bench serve)
fi

echo "[INFO] Python: $(command -v python)"
echo "[INFO] Running fixed config: c=${MAX_CONCURRENCY}, r=${REQUEST_RATE}, prompts=${NUM_PROMPTS}, temp=${TEMPERATURE}"

'time' "${BENCH_CMD[@]}" \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --endpoint "${ENDPOINT}" \
  --dataset-name "${DATASET_NAME}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --request-rate "${REQUEST_RATE}" \
  --num-warmups "${NUM_WARMUPS}" \
  --temperature "${TEMPERATURE}" \
  --save-result \
  --result-dir "${RESULT_DIR}" \
  --result-filename "${RESULT_FILENAME}" \
  --disable-tqdm

echo "[DONE] Result saved: ${RESULT_DIR}/${RESULT_FILENAME}"
