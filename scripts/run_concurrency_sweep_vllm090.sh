#!/usr/bin/env bash
set -euo pipefail

# vLLM==0.9.0 专用并发压测脚本（不依赖新版本 input-len/output-len 别名）
# 关键点：使用 0.9.0 更稳妥的 random 数据集参数
#   --random-input-len / --random-output-len
#
# 默认参数采用你已确认的固定组：
# - backend=openai-chat
# - base-url=http://127.0.0.1:8902
# - endpoint=/v1/chat/completions
# - dataset-name=random
# - random-input-len=1024
# - random-output-len=128
# - num-prompts=128
# - num-warmups=8
# - 并发扫描：1 8 16 24 32 40 48 56 64
# - 每个并发默认测 request-rate={concurrency, inf}

BASE_URL="${BASE_URL:-http://127.0.0.1:8900}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
MODEL="${MODEL:-}"

CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 8 16 24 32 40 48 56 64}"
REQUEST_RATE_LIST="${REQUEST_RATE_LIST:-}"

RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench_concurrency_v090}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VENV_PATH="${VENV_PATH:-/usr/local/app/GSQ/.venv}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_concurrency_sweep_vllm090.sh --model <served-model-name-or-id>

Optional env overrides:
  BASE_URL, ENDPOINT, BACKEND, DATASET_NAME,
  RANDOM_INPUT_LEN, RANDOM_OUTPUT_LEN,
  NUM_PROMPTS, NUM_WARMUPS, TEMPERATURE, SEED,
  CONCURRENCY_LIST, REQUEST_RATE_LIST,
  RESULT_DIR, RUN_TAG, VENV_PATH

Examples:
  MODEL="/path/or/model_id" bash scripts/run_concurrency_sweep_vllm090.sh
  bash scripts/run_concurrency_sweep_vllm090.sh --model Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "[ERROR] --model is required (or set MODEL env)"
  usage
  exit 2
fi

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
echo "[INFO] model=${MODEL}"
echo "[INFO] base_url=${BASE_URL}"
echo "[INFO] endpoint=${ENDPOINT}"
echo "[INFO] backend=${BACKEND}"
echo "[INFO] dataset=${DATASET_NAME}"
echo "[INFO] random_input_len=${RANDOM_INPUT_LEN}, random_output_len=${RANDOM_OUTPUT_LEN}"
echo "[INFO] num_prompts=${NUM_PROMPTS}, num_warmups=${NUM_WARMUPS}, temperature=${TEMPERATURE}, seed=${SEED}"
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
      --model "${MODEL}" \
      --backend "${BACKEND}" \
      --base-url "${BASE_URL}" \
      --endpoint "${ENDPOINT}" \
      --dataset-name "${DATASET_NAME}" \
      --random-input-len "${RANDOM_INPUT_LEN}" \
      --random-output-len "${RANDOM_OUTPUT_LEN}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${c}" \
      --request-rate "${r}" \
      --num-warmups "${NUM_WARMUPS}" \
      --temperature "${TEMPERATURE}" \
      --seed "${SEED}" \
      --save-result \
      --result-dir "${RESULT_DIR}/${RUN_TAG}" \
      --result-filename "c${c}_r${r}.json" \
      --disable-tqdm
  done
done

echo
echo "[DONE] Sweep finished. Results: ${RESULT_DIR}/${RUN_TAG}"
