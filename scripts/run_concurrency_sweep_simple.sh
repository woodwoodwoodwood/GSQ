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

# 默认只扫这两个变量：每个并发仅测 request_rate=concurrency 和 inf
# 如需自定义，可手动设置 REQUEST_RATE_LIST（例如："16 32 inf"）
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 8 16 24 32 40 48 56 64}"
REQUEST_RATE_LIST="${REQUEST_RATE_LIST:-}"

RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench_concurrency_only}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VENV_PATH="${VENV_PATH:-/usr/local/app/GSQ/.venv}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] venv not found: ${VENV_PATH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

if command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
else
  BENCH_CMD=(python -m vllm.entrypoints.cli.main bench serve)
fi

mkdir -p "${RESULT_DIR}/${RUN_TAG}"

echo "[INFO] python=$(command -v python)"
echo "[INFO] base_url=${BASE_URL}"
echo "[INFO] concurrency_list=${CONCURRENCY_LIST}"
if [[ -n "${REQUEST_RATE_LIST}" ]]; then
  echo "[INFO] request_rate_list=${REQUEST_RATE_LIST} (manual override)"
else
  echo "[INFO] request_rate_list=<auto: concurrency,inf>"
fi
echo "[INFO] result_dir=${RESULT_DIR}/${RUN_TAG}"

for c in ${CONCURRENCY_LIST}; do
  if [[ -n "${REQUEST_RATE_LIST}" ]]; then
    RATE_LIST="${REQUEST_RATE_LIST}"
  else
    RATE_LIST="${c} inf"
  fi

  for r in ${RATE_LIST}; do
    # 约束：仅测试 request_rate >= concurrency（inf 视为满足）
    if [[ "${r}" != "inf" ]] && awk -v rr="${r}" -v cc="${c}" 'BEGIN { exit !(rr < cc) }'; then
      echo "[SKIP] request_rate(${r}) < concurrency(${c})"
      continue
    fi

    echo
    echo "[INFO] ===== concurrency=${c}, request_rate=${r} ====="

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
      --temperature "${TEMPERATURE}" \
      --save-result \
      --result-dir "${RESULT_DIR}/${RUN_TAG}" \
      --result-filename "c${c}_r${r}.json" \
      --disable-tqdm
  done
done

echo
echo "[DONE] Sweep finished. Results: ${RESULT_DIR}/${RUN_TAG}"
