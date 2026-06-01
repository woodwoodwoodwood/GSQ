#!/usr/bin/env bash
set -euo pipefail

# vLLM 在线吞吐两阶段自动搜索脚本
# - stage1: 粗扫快速定位高吞吐区域
# - stage2: 围绕 stage1 最优点细扫
# 目标：在 failed==0 的前提下最大化 total_token_throughput（次选 output_throughput）

BASE_URL="${BASE_URL:-http://127.0.0.1:8902}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
BACKEND="${BACKEND:-openai-chat}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
RESULT_DIR="${RESULT_DIR:-/usr/local/app/GSQ/benchmark/vllm_bench}"
VENV_PATH="${VENV_PATH:-/usr/local/app/GSQ/.venv}"

# 阶段1（粗扫）：样本小、速度快
STAGE1_NUM_PROMPTS="${STAGE1_NUM_PROMPTS:-128}"
STAGE1_NUM_WARMUPS="${STAGE1_NUM_WARMUPS:-8}"
STAGE1_CONCURRENCY_LIST=(${STAGE1_CONCURRENCY_LIST:-16 32 48})
STAGE1_REQUEST_RATE_LIST=(${STAGE1_REQUEST_RATE_LIST:-16 32 64 96 inf})

# 阶段2（细扫）：样本大、用于最终确认
STAGE2_NUM_PROMPTS="${STAGE2_NUM_PROMPTS:-512}"
STAGE2_NUM_WARMUPS="${STAGE2_NUM_WARMUPS:-16}"

# 细扫邻域（围绕 stage1 最优点）
CONCURRENCY_DELTAS=(${CONCURRENCY_DELTAS:--8 0 8})
RATE_FACTORS=(${RATE_FACTORS:-0.75 1.0 1.25})
MIN_CONCURRENCY="${MIN_CONCURRENCY:-1}"
MIN_RATE="${MIN_RATE:-1}"
MAX_RATE="${MAX_RATE:-512}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] venv not found: ${VENV_PATH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

mkdir -p "${RESULT_DIR}"

echo "[INFO] Result dir: ${RESULT_DIR}"
echo "[INFO] Base URL: ${BASE_URL}"
echo "[INFO] Venv path: ${VENV_PATH}"
echo "[INFO] Python: $(command -v python)"

if command -v vllm >/dev/null 2>&1; then
  BENCH_CMD=(vllm bench serve)
else
  BENCH_CMD=(python -m vllm.entrypoints.cli.main bench serve)
fi

echo "[INFO] Bench command: ${BENCH_CMD[*]}"

run_one() {
  local stage="$1"
  local c="$2"
  local r="$3"
  local num_prompts="$4"
  local num_warmups="$5"

  local out_json="${RESULT_DIR}/${stage}_c${c}_r${r}.json"
  echo "[RUN] stage=${stage}, concurrency=${c}, request_rate=${r}, output=${out_json}"

  "${BENCH_CMD[@]}" \
    --backend "${BACKEND}" \
    --base-url "${BASE_URL}" \
    --endpoint "${ENDPOINT}" \
    --dataset-name "${DATASET_NAME}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${c}" \
    --request-rate "${r}" \
    --num-warmups "${num_warmups}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${stage}_c${c}_r${r}.json" \
    --disable-tqdm
}

find_best() {
  local pattern="$1"
  python - "$RESULT_DIR" "$pattern" << 'PY'
import glob
import json
import os
import re
import sys

result_dir = sys.argv[1]
pattern = sys.argv[2]
files = glob.glob(os.path.join(result_dir, pattern))

best = None
# 排序规则：
# 1) failed 越小越好（优先 0）
# 2) total_token_throughput 越大越好
# 3) output_throughput 越大越好
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        continue

    failed = int(d.get("failed", 999999))
    ttt = float(d.get("total_token_throughput", 0.0) or 0.0)
    ott = float(d.get("output_throughput", 0.0) or 0.0)

    base = os.path.basename(f)
    m = re.search(r"_c(\d+)_r([^\.]+)\.json$", base)
    if not m:
        continue
    c = m.group(1)
    r = m.group(2)

    key = (-failed, ttt, ott)
    if best is None or key > best[0]:
        best = (key, f, c, r, failed, ttt, ott)

if best is None:
    print("", end="")
    sys.exit(1)

_, f, c, r, failed, ttt, ott = best
print(f"{c}\t{r}\t{failed}\t{ttt}\t{ott}\t{f}")
PY
}

echo "[INFO] ===== Stage 1: 粗扫开始 ====="
for c in "${STAGE1_CONCURRENCY_LIST[@]}"; do
  for r in "${STAGE1_REQUEST_RATE_LIST[@]}"; do
    run_one "stage1" "${c}" "${r}" "${STAGE1_NUM_PROMPTS}" "${STAGE1_NUM_WARMUPS}"
  done
done

echo "[INFO] ===== Stage 1: 粗扫完成，开始选点 ====="
STAGE1_BEST="$(find_best 'stage1_c*_r*.json')"
if [[ -z "${STAGE1_BEST}" ]]; then
  echo "[ERROR] Stage1 没有可解析结果，无法进入 Stage2"
  exit 1
fi

BEST_C="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $1}')"
BEST_R="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $2}')"
BEST_FAILED="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $3}')"
BEST_TTT="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $4}')"
BEST_OTT="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $5}')"
BEST_FILE="$(echo "${STAGE1_BEST}" | awk -F'\t' '{print $6}')"

echo "[INFO] Stage1 best: c=${BEST_C}, r=${BEST_R}, failed=${BEST_FAILED}, total_tps=${BEST_TTT}, output_tps=${BEST_OTT}"
echo "[INFO] Stage1 best file: ${BEST_FILE}"

echo "[INFO] ===== Stage 2: 细扫开始 ====="

declare -A SEEN
STAGE2_RUNS=()

add_stage2_run() {
  local c="$1"
  local r="$2"
  local key="${c}_${r}"
  if [[ -n "${SEEN[$key]:-}" ]]; then
    return
  fi
  SEEN[$key]=1
  STAGE2_RUNS+=("${c}|${r}")
}

for dc in "${CONCURRENCY_DELTAS[@]}"; do
  c2=$((BEST_C + dc))
  if (( c2 < MIN_CONCURRENCY )); then
    continue
  fi

  if [[ "${BEST_R}" == "inf" ]]; then
    # 如果粗扫最优是 inf，则细扫用固定高 RPS 邻域
    for r2 in 64 96 128 inf; do
      add_stage2_run "${c2}" "${r2}"
    done
  else
    for f in "${RATE_FACTORS[@]}"; do
      r2="$(python - << PY
best_r = float(${BEST_R})
f = float(${f})
v = int(round(best_r * f))
print(v)
PY
)"
      if (( r2 < MIN_RATE )); then
        r2=${MIN_RATE}
      fi
      if (( r2 > MAX_RATE )); then
        r2=${MAX_RATE}
      fi
      add_stage2_run "${c2}" "${r2}"
    done

    # 额外补一个 inf，观察是否还能提升
    add_stage2_run "${c2}" "inf"
  fi
done

for item in "${STAGE2_RUNS[@]}"; do
  c="${item%%|*}"
  r="${item##*|}"
  run_one "stage2" "${c}" "${r}" "${STAGE2_NUM_PROMPTS}" "${STAGE2_NUM_WARMUPS}"
done

echo "[INFO] ===== Stage 2: 细扫完成，汇总最终最优 ====="
FINAL_BEST="$(find_best 'stage2_c*_r*.json')"
if [[ -z "${FINAL_BEST}" ]]; then
  echo "[WARN] Stage2 没有可解析结果，回退使用 Stage1 最优"
  FINAL_BEST="${STAGE1_BEST}"
fi

FINAL_C="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $1}')"
FINAL_R="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $2}')"
FINAL_FAILED="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $3}')"
FINAL_TTT="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $4}')"
FINAL_OTT="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $5}')"
FINAL_FILE="$(echo "${FINAL_BEST}" | awk -F'\t' '{print $6}')"

echo "[DONE] ===== 两阶段搜索完成 ====="
echo "[DONE] BEST concurrency=${FINAL_C}, request_rate=${FINAL_R}"
echo "[DONE] BEST failed=${FINAL_FAILED}, total_token_throughput=${FINAL_TTT}, output_throughput=${FINAL_OTT}"
echo "[DONE] BEST result file=${FINAL_FILE}"
