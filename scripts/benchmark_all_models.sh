#!/usr/bin/env bash
# ============================================================================
# GSQ — Speed & memory benchmark for FP16 / GPTQ-Int4 / GSQ-2bit models
# ============================================================================
# Tests each model sequentially: start vLLM → benchmark → stop → next model.
# Benchmarks across multiple input lengths (1k → 128k) and concurrency levels.
#
# Usage:
#   bash scripts/benchmark_all_models.sh
#   MODELS="fp16 gptq" bash scripts/benchmark_all_models.sh   # subset only
# ============================================================================

set -euo pipefail

# Skip .venv activation — use conda env
export VENV_PATH="/nonexistent_venv_skip"
source "$(dirname "$0")/_common.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PORT="8900"
HOST="127.0.0.1"

# ── Benchmark parameters ──────────────────────────────────────────────────
# Long sequence scenarios: 1k, 4k, 16k, 32k, 64k, 128k
INPUT_LENGTHS="1024 4096 16384 32768 65536 131072"
OUTPUT_LEN="128"
CONCURRENCY_LEVELS="1 4 8 16"
NUM_PROMPTS="16"
WARMUP="2"

# ── Model definitions ─────────────────────────────────────────────────────
declare -A MODEL_PATHS=(
    ["fp16"]="/usr/local/app/models/Qwen3-30B-A3B-Instruct-2507"
    ["gptq"]="/usr/local/app/models/Qwen3-30B-A3B-GPTQ-Int4"
    ["gsq"]="/usr/local/app/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit"
)

# GPU utilization varies: FP16 needs ~92GB on H20 (96GB), quantized models ~80-85%
declare -A GPU_UTIL=(
    ["fp16"]="0.92"
    ["gptq"]="0.85"
    ["gsq"]="0.85"
)

# max-model-len: must be >= longest input_len + output_len
# FP16 on H20 (96GB): max-model-len likely limited to ~32k-64k
# Quantized models: can support 128k
declare -A MAX_MODEL_LEN_MAP=(
    ["fp16"]="65536"
    ["gptq"]="131072"
    ["gsq"]="131072"
)

# Which models to test (default: all)
MODELS="${MODELS:-fp16 gptq gsq}"

RESULTS_DIR="${REPO_ROOT}/benchmark_results"
mkdir -p "${RESULTS_DIR}"

cd "${REPO_ROOT}"

# ── Helper functions ───────────────────────────────────────────────────────

kill_vllm() {
    echo "  Stopping vLLM..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "serve_vllm.py" 2>/dev/null || true
    sleep 5
    # Verify GPU memory freed
    for i in $(seq 1 6); do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 2>/dev/null | head -1 || echo "0")
        if [[ "${USED}" -lt 2048 ]]; then
            echo "  GPU memory freed (used=${USED}MiB)"
            return 0
        fi
        echo "  Waiting for GPU to free... used=${USED}MiB"
        sleep 5
    done
    echo "  WARNING: GPU may still have residual memory"
}

wait_for_server() {
    local url="http://${HOST}:${PORT}/health"
    echo "  Waiting for server at ${url}..."
    for i in $(seq 1 90); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "  Server healthy after ~$((i*2))s"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: server did not become healthy within 180s" >&2
    return 1
}

# ── Main loop ──────────────────────────────────────────────────────────────

echo "=========================================="
echo "Speed & Memory Benchmark — Qwen3-30B-A3B"
echo "Models: ${MODELS}"
echo "Input lengths: ${INPUT_LENGTHS}"
echo "Output length: ${OUTPUT_LEN}"
echo "Concurrency levels: ${CONCURRENCY_LEVELS}"
echo "=========================================="

for MODEL_NAME in ${MODELS}; do
    MODEL_PATH="${MODEL_PATHS[${MODEL_NAME}]}"
    GPU_MEM="${GPU_UTIL[${MODEL_NAME}]:-0.85}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN_MAP[${MODEL_NAME}]:-131072}"
    JSON_OUT="${RESULTS_DIR}/benchmark_${MODEL_NAME}.json"

    echo ""
    echo "================================================================"
    echo "Benchmarking: ${MODEL_NAME} (${MODEL_PATH})"
    echo "  max-model-len: ${MAX_MODEL_LEN}"
    echo "  gpu-memory-utilization: ${GPU_MEM}"
    echo "================================================================"

    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "  SKIP: model path does not exist: ${MODEL_PATH}"
        continue
    fi

    # Kill any existing vLLM process
    kill_vllm

    # Start vLLM server
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    VLLM_CMD=(
        python "${REPO_ROOT}/serve_vllm.py"
        --num-nodes 1
        --port "${PORT}"
        "${MODEL_PATH}"
        --tensor-parallel-size 1
        --trust-remote-code
        --host "${HOST}"
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEM}"
        --tokenizer-mode hf
        --max-num-seqs 32
    )

    echo "  Starting vLLM..."
    "${VLLM_CMD[@]}" &>"${RESULTS_DIR}/vllm_${MODEL_NAME}.log" &
    SERVER_PID=$!

    # Wait for server
    if ! wait_for_server; then
        echo "  ERROR: server failed to start. Check ${RESULTS_DIR}/vllm_${MODEL_NAME}.log"
        kill ${SERVER_PID} 2>/dev/null || true
        continue
    fi

    # Filter input lengths that fit within max-model-len
    USABLE_LENGTHS=""
    for len in ${INPUT_LENGTHS}; do
        if [[ $((len + OUTPUT_LEN)) -le ${MAX_MODEL_LEN} ]]; then
            USABLE_LENGTHS="${USABLE_LENGTHS} ${len}"
        else
            echo "  NOTE: skipping input_len=${len} (exceeds max-model-len=${MAX_MODEL_LEN})"
        fi
    done

    if [[ -z "${USABLE_LENGTHS}" ]]; then
        echo "  ERROR: no usable input lengths for max-model-len=${MAX_MODEL_LEN}"
        kill ${SERVER_PID} 2>/dev/null || true
        continue
    fi

    # Run benchmark
    echo "  Running benchmark with input lengths: ${USABLE_LENGTHS}"
    python "${REPO_ROOT}/scripts/benchmark_speed.py" \
        --base-url "http://${HOST}:${PORT}" \
        --model "${MODEL_PATH}" \
        --input-len ${USABLE_LENGTHS} \
        --output-len "${OUTPUT_LEN}" \
        --concurrency ${CONCURRENCY_LEVELS} \
        --num-prompts "${NUM_PROMPTS}" \
        --warmup "${WARMUP}" \
        --output-json "${JSON_OUT}" \
        || echo "  WARNING: benchmark had errors"

    echo "  Results saved to ${JSON_OUT}"

    # Stop server
    kill ${SERVER_PID} 2>/dev/null || true
    kill_vllm

    echo "  Done with ${MODEL_NAME}"
done

# ── Print final comparison ─────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "FINAL COMPARISON"
echo "================================================================"

python - <<'PY'
import json, os, sys

results_dir = os.environ.get("RESULTS_DIR", "benchmark_results")
models = ["fp16", "gptq", "gsq"]

all_data = {}
for m in models:
    path = os.path.join(results_dir, f"benchmark_{m}.json")
    if os.path.exists(path):
        with open(path) as f:
            all_data[m] = json.load(f)

if not all_data:
    print("No benchmark results found!")
    sys.exit(0)

# Group by input_len
input_lens = sorted(set(r["input_len"] for results in all_data.values() for r in results))

for il in input_lens:
    print(f"\n--- Input Length: {il} tokens ---")
    print(f"{'Model':<8} | {'Conc':>4} | {'Throughput':>10} | {'TTFT':>8} | {'TPOT':>8} | {'Latency':>8} | {'GPUMem':>8}")
    print(f"{'':8} | {'':4} | {'tok/s':>10} | {'s':>8} | {'ms':>8} | {'s':>8} | {'GiB':>8}")
    print("-" * 75)
    for m in models:
        if m not in all_data:
            continue
        for r in all_data[m]:
            if r["input_len"] != il:
                continue
            ttft = f"{r['ttft_mean_s']:.3f}" if r.get('ttft_mean_s') else "N/A"
            tpot = f"{r['tpot_mean_ms']:.2f}" if r.get('tpot_mean_ms') else "N/A"
            lat = f"{r['latency_mean_s']:.3f}" if r.get('latency_mean_s') else "N/A"
            gmem = f"{r['gpu_mem_gib']:.2f}" if r.get('gpu_mem_gib') else "N/A"
            print(f"{m:<8} | {r['concurrency']:>4} | {r['throughput_tok_per_s']:>10.2f} | "
                  f"{ttft:>8} | {tpot:>8} | {lat:>8} | {gmem:>8}")
PY
