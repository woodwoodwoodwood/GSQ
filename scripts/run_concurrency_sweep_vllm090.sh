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
BENCH_MODEL="${BENCH_MODEL:-}"
TOKENIZER="${TOKENIZER:-}"
API_KEY="${API_KEY:-}"

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
  BENCH_MODEL, TOKENIZER, API_KEY,
  CONCURRENCY_LIST, REQUEST_RATE_LIST,
  RESULT_DIR, RUN_TAG, VENV_PATH

Examples:
  # 仅传服务端模型名
  bash scripts/run_concurrency_sweep_vllm090.sh --model Qwen3-30B-A3B-GPTQ-Int4

  # 推荐：显式指定本地路径，避免bench去HF拉tokenizer
  BENCH_MODEL="/data1/models/Qwen3-30B-A3B-GPTQ-Int4" \
  TOKENIZER="/data1/models/Qwen3-30B-A3B-Instruct-2507" \
  bash scripts/run_concurrency_sweep_vllm090.sh --model Qwen3-30B-A3B-GPTQ-Int4

  # 若服务端启用 --api-key，传入API_KEY自动追加Authorization头
  API_KEY="<your_api_key>" bash scripts/run_concurrency_sweep_vllm090.sh --model Qwen3-30B-A3B-GPTQ-Int4
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

# 如果未显式提供，bench模型默认跟请求模型一致
if [[ -z "${BENCH_MODEL}" ]]; then
  BENCH_MODEL="${MODEL}"
fi

if command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
else
  BENCH_CMD=(python -m vllm.entrypoints.cli.main bench serve)
fi

HELP_TEXT="$(${BENCH_CMD[@]} --help 2>&1 || true)"

has_flag() {
  local flag="$1"
  grep -q -- "${flag}" <<<"${HELP_TEXT}"
}

# base-url 兼容：若不支持 --base-url，则回退 --host/--port
BASE_URL_ARGS=()
if has_flag "--base-url"; then
  BASE_URL_ARGS=(--base-url "${BASE_URL}")
else
  URL_NO_SCHEME="${BASE_URL#http://}"
  URL_NO_SCHEME="${URL_NO_SCHEME#https://}"
  URL_HOSTPORT="${URL_NO_SCHEME%%/*}"
  URL_HOST="${URL_HOSTPORT%%:*}"
  URL_PORT="${URL_HOSTPORT##*:}"
  if [[ "${URL_PORT}" == "${URL_HOSTPORT}" ]]; then
    URL_PORT="8000"
  fi
  if has_flag "--host"; then
    BASE_URL_ARGS+=(--host "${URL_HOST}")
  fi
  if has_flag "--port"; then
    BASE_URL_ARGS+=(--port "${URL_PORT}")
  fi
fi

mkdir -p "${RESULT_DIR}/${RUN_TAG}"

echo "[INFO] python=$(command -v python)"
echo "[INFO] model=${MODEL}"
echo "[INFO] bench_model=${BENCH_MODEL}"
echo "[INFO] tokenizer=${TOKENIZER:-<auto>}"
echo "[INFO] api_key=$([[ -n "${API_KEY}" ]] && echo "<set>" || echo "<empty>")"
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

    CMD=("${BENCH_CMD[@]}")

    # v0.9 不同小版本参数差异较大：只传当前 help 中存在的参数
    # 关键兼容：--model 用 BENCH_MODEL（本地路径/HF id），请求体模型名用 --served-model-name 覆盖
    if has_flag "--model"; then CMD+=(--model "${BENCH_MODEL}"); fi
    if has_flag "--served-model-name"; then CMD+=(--served-model-name "${MODEL}"); fi
    if has_flag "--tokenizer" && [[ -n "${TOKENIZER}" ]]; then
      CMD+=(--tokenizer "${TOKENIZER}")
    elif has_flag "--skip-tokenizer-init"; then
      CMD+=(--skip-tokenizer-init)
    fi

    # 兼容不同版本字段命名：backend / endpoint-type
    if has_flag "--backend"; then
      CMD+=(--backend "${BACKEND}")
    elif has_flag "--endpoint-type"; then
      if [[ "${ENDPOINT}" == *"/chat/completions"* ]]; then
        CMD+=(--endpoint-type "openai-chat")
      else
        CMD+=(--endpoint-type "openai-comp")
      fi
    fi

    # 若服务端启用API Key鉴权，自动注入Authorization头
    if [[ -n "${API_KEY}" ]] && has_flag "--header"; then
      CMD+=(--header "Authorization=Bearer ${API_KEY}")
    fi
    CMD+=("${BASE_URL_ARGS[@]}")
    if has_flag "--endpoint"; then CMD+=(--endpoint "${ENDPOINT}"); fi
    if has_flag "--dataset-name"; then CMD+=(--dataset-name "${DATASET_NAME}"); fi

    if has_flag "--random-input-len"; then CMD+=(--random-input-len "${RANDOM_INPUT_LEN}");
    elif has_flag "--input-len"; then CMD+=(--input-len "${RANDOM_INPUT_LEN}"); fi

    if has_flag "--random-output-len"; then CMD+=(--random-output-len "${RANDOM_OUTPUT_LEN}");
    elif has_flag "--output-len"; then CMD+=(--output-len "${RANDOM_OUTPUT_LEN}"); fi

    if has_flag "--num-prompts"; then CMD+=(--num-prompts "${NUM_PROMPTS}"); fi
    if has_flag "--max-concurrency"; then CMD+=(--max-concurrency "${c}"); fi
    if has_flag "--request-rate"; then CMD+=(--request-rate "${r}"); fi
    if has_flag "--num-warmups"; then CMD+=(--num-warmups "${NUM_WARMUPS}"); fi
    if has_flag "--temperature"; then CMD+=(--temperature "${TEMPERATURE}"); fi
    if has_flag "--seed"; then CMD+=(--seed "${SEED}"); fi
    if has_flag "--save-result"; then CMD+=(--save-result); fi
    if has_flag "--result-dir"; then CMD+=(--result-dir "${RESULT_DIR}/${RUN_TAG}"); fi
    if has_flag "--result-filename"; then CMD+=(--result-filename "c${c}_r${r}.json"); fi
    if has_flag "--disable-tqdm"; then CMD+=(--disable-tqdm); fi

    "${CMD[@]}"
  done
done

echo
echo "[DONE] Sweep finished. Results: ${RESULT_DIR}/${RUN_TAG}"
