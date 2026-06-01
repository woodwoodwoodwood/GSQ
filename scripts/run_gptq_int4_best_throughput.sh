#!/usr/bin/env bash
set -euo pipefail

# GPTQ-INT4 最佳吞吐点扫参脚本（vLLM bench serve）
#
# 功能：
# 1) 自动按参数网格重启 vLLM(gptq_marlin) 服务
# 2) 每组参数进行 warmup + 正式压测（可重复多次）
# 3) 保存每次压测的 result json + 控制台日志
# 4) 输出汇总 CSV，便于快速挑选最佳吞吐点
#
# 使用示例：
#   bash /usr/local/app/GSQ/scripts/run_gptq_int4_best_throughput.sh
#
# 可通过环境变量覆盖默认值，例如：
#   CUDA_DEVICE=6 PORT=8900 RUNS_PER_SETTING=3 \
#   SEQ_CANDIDATES="32 48 64" BTOK_CANDIDATES="8192 12288 16384" \
#   bash /usr/local/app/GSQ/scripts/run_gptq_int4_best_throughput.sh

MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen3-30B-A3B-GPTQ-Int4}"
CUDA_DEVICE="${CUDA_DEVICE:-6}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8900}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"

VENV_PATH="${VENV_PATH:-/usr/local/app/GSQ/.venv}"
RESULT_ROOT="${RESULT_ROOT:-/usr/local/app/GSQ/benchmark/vllm_bench_gptq_int4_best}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RESULT_ROOT}/${RUN_TAG}"

# 固定压测口径（默认与当前讨论一致）
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"
TEMPERATURE="${TEMPERATURE:-0}"

# 服务参数搜索空间
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_CANDIDATES="${GPU_MEM_CANDIDATES:-0.90}"
SEQ_CANDIDATES="${SEQ_CANDIDATES:-32 48 64}"
BTOK_CANDIDATES="${BTOK_CANDIDATES:-8192 12288 16384}"

# 每组参数重复次数（建议 2~3，取中位数更稳）
RUNS_PER_SETTING="${RUNS_PER_SETTING:-2}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] venv not found: ${VENV_PATH}"
  exit 1
fi

mkdir -p "${RUN_DIR}" "${RUN_DIR}/json" "${RUN_DIR}/logs"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

if command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm)
else
  VLLM_CMD=(python -m vllm.entrypoints.cli.main)
fi

echo "setting_id,gpu_mem,max_num_seqs,max_num_batched_tokens,run_id,total_tps,output_tps,ttft_p99_ms,tpot_p99_ms,itl_p99_ms,result_json,bench_log,serve_log" > "${SUMMARY_CSV}"

wait_for_health() {
  local url="$1"
  local timeout_s="${2:-180}"
  local waited=0
  while true; do
    if curl -sSf "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
    if [[ "${waited}" -ge "${timeout_s}" ]]; then
      return 1
    fi
  done
}

stop_old_server() {
  # 尽量精确匹配，避免误杀其他任务
  pkill -f "vllm serve ${MODEL_PATH}.*--port ${PORT}" >/dev/null 2>&1 || true
  pkill -f "vllm.entrypoints.cli.main serve ${MODEL_PATH}.*--port ${PORT}" >/dev/null 2>&1 || true
  sleep 2
}

start_server() {
  local gpu_mem="$1"
  local max_num_seqs="$2"
  local max_num_batched_tokens="$3"
  local serve_log="$4"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${VLLM_CMD[@]}" serve "${MODEL_PATH}" \
    --trust-remote-code \
    --quantization gptq_marlin \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${gpu_mem}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --generation-config vllm \
    > "${serve_log}" 2>&1 &

  local serve_pid=$!
  echo "${serve_pid}"
}

run_bench_once() {
  local setting_id="$1"
  local run_id="$2"
  local bench_log="$3"
  local result_json_name="$4"

  set +e
  "${VLLM_CMD[@]}" bench serve \
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
    --result-dir "${RUN_DIR}/json" \
    --result-filename "${result_json_name}" \
    --disable-tqdm \
    > "${bench_log}" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[WARN] setting=${setting_id} run=${run_id} bench failed, rc=${rc}. See ${bench_log}"
    echo "NA,NA,NA,NA,NA"
    return 0
  fi

  # 从控制台日志抽取核心指标（避免依赖 result json 内部 schema）
  local total_tps output_tps ttft_p99 tpot_p99 itl_p99
  total_tps=$(grep -F "Total token throughput (tok/s):" "${bench_log}" | tail -n1 | awk '{print $NF}')
  output_tps=$(grep -F "Output token throughput (tok/s):" "${bench_log}" | tail -n1 | awk '{print $NF}')
  ttft_p99=$(grep -F "P99 TTFT (ms):" "${bench_log}" | tail -n1 | awk '{print $NF}')
  tpot_p99=$(grep -F "P99 TPOT (ms):" "${bench_log}" | tail -n1 | awk '{print $NF}')
  itl_p99=$(grep -F "P99 ITL (ms):" "${bench_log}" | tail -n1 | awk '{print $NF}')

  total_tps="${total_tps:-NA}"
  output_tps="${output_tps:-NA}"
  ttft_p99="${ttft_p99:-NA}"
  tpot_p99="${tpot_p99:-NA}"
  itl_p99="${itl_p99:-NA}"

  echo "${total_tps},${output_tps},${ttft_p99},${tpot_p99},${itl_p99}"
}

setting_idx=0
best_tps="-1"
best_desc=""

echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Base URL: ${BASE_URL}"
echo "[INFO] Search space:"
echo "       gpu_mem=[${GPU_MEM_CANDIDATES}]"
echo "       max_num_seqs=[${SEQ_CANDIDATES}]"
echo "       max_num_batched_tokens=[${BTOK_CANDIDATES}]"
echo "       runs_per_setting=${RUNS_PER_SETTING}"

for gpu_mem in ${GPU_MEM_CANDIDATES}; do
  for seqs in ${SEQ_CANDIDATES}; do
    for btok in ${BTOK_CANDIDATES}; do
      setting_idx=$((setting_idx + 1))
      setting_id=$(printf "s%02d" "${setting_idx}")

      serve_log="${RUN_DIR}/logs/${setting_id}_serve.log"
      echo
      echo "[INFO] ===== ${setting_id}: gpu_mem=${gpu_mem}, max_num_seqs=${seqs}, max_num_batched_tokens=${btok} ====="

      stop_old_server
      serve_pid=$(start_server "${gpu_mem}" "${seqs}" "${btok}" "${serve_log}")

      if ! wait_for_health "${BASE_URL}" 240; then
        echo "[WARN] server not ready for ${setting_id}. See ${serve_log}"
        kill "${serve_pid}" >/dev/null 2>&1 || true
        continue
      fi

      # 检查是否走到 gptq_marlin（仅提示，不强制失败）
      if grep -qi "gptq_marlin" "${serve_log}"; then
        echo "[INFO] ${setting_id}: detected gptq_marlin in serve log"
      else
        echo "[WARN] ${setting_id}: gptq_marlin not found in serve log yet (may appear later)."
      fi

      # 每组参数重复压测多次，便于观察波动
      for run_id in $(seq 1 "${RUNS_PER_SETTING}"); do
        bench_log="${RUN_DIR}/logs/${setting_id}_run${run_id}.log"
        result_json_name="${setting_id}_run${run_id}.json"

        echo "[INFO] ${setting_id} run${run_id}/${RUNS_PER_SETTING} benchmarking..."
        metrics=$(run_bench_once "${setting_id}" "${run_id}" "${bench_log}" "${result_json_name}")

        IFS=',' read -r total_tps output_tps ttft_p99 tpot_p99 itl_p99 <<< "${metrics}"

        echo "${setting_id},${gpu_mem},${seqs},${btok},${run_id},${total_tps},${output_tps},${ttft_p99},${tpot_p99},${itl_p99},${RUN_DIR}/json/${result_json_name},${bench_log},${serve_log}" >> "${SUMMARY_CSV}"

        echo "[INFO] ${setting_id} run${run_id} => total_tps=${total_tps}, output_tps=${output_tps}, ttft_p99=${ttft_p99}, itl_p99=${itl_p99}"

        # 记录全局最佳（按 total_tps）
        if [[ "${total_tps}" != "NA" ]]; then
          awk -v a="${total_tps}" -v b="${best_tps}" 'BEGIN{exit !(a>b)}' && {
            best_tps="${total_tps}"
            best_desc="${setting_id} (gpu_mem=${gpu_mem}, seqs=${seqs}, btok=${btok}, run=${run_id})"
          }
        fi
      done

      # 切参数前结束当前服务
      kill "${serve_pid}" >/dev/null 2>&1 || true
      sleep 1
    done
  done
done

echo
echo "[DONE] Sweep finished."
echo "[DONE] Summary CSV: ${SUMMARY_CSV}"
if [[ "${best_tps}" != "-1" ]]; then
  echo "[DONE] Best total throughput: ${best_tps} tok/s @ ${best_desc}"
else
  echo "[DONE] No valid throughput parsed. Please inspect logs under ${RUN_DIR}/logs"
fi
