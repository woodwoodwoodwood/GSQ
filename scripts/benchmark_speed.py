#!/usr/bin/env python3
"""
vLLM server benchmark: measures TTFT, TPOT, throughput and peak GPU memory
at various concurrency levels and input lengths.

Usage:
    python benchmark_speed.py --base-url http://127.0.0.1:8900 --model /path/to/model
    python benchmark_speed.py --base-url http://127.0.0.1:8900 --model /path/to/model \
        --input-len 1024 4096 16384 --concurrency 1 4 8 16
"""
import os
import time
import json
import argparse
import threading
import subprocess
import statistics
from concurrent.futures import ThreadPoolExecutor

os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import requests


def get_gpu_memory_mib(gpu_id=0):
    """Get current GPU memory usage in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            text=True
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return None


class GPUMemoryMonitor:
    """Background thread that polls GPU memory and tracks peak."""

    def __init__(self, gpu_id=0, interval=0.5):
        self.gpu_id = gpu_id
        self.interval = interval
        self.peak_mib = 0
        self.baseline_mib = 0
        self._stop = threading.Event()
        self._thread = None
        self._samples = []

    def start(self):
        self.baseline_mib = get_gpu_memory_mib(self.gpu_id) or 0
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            mem = get_gpu_memory_mib(self.gpu_id)
            if mem is not None:
                self._samples.append(mem)
                if mem > self.peak_mib:
                    self.peak_mib = mem
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_results(self):
        return {
            "baseline_mib": self.baseline_mib,
            "peak_mib": self.peak_mib,
            "delta_mib": self.peak_mib - self.baseline_mib,
            "delta_gib": (self.peak_mib - self.baseline_mib) / 1024,
            "num_samples": len(self._samples),
        }


def send_request_stream(args, prompt, idx):
    """Send a single streaming request, measure TTFT and per-token timing."""
    url = f"{args.base_url}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_len,
        "temperature": 0.0,
        "stream": True,
    }

    result = {"idx": idx, "success": False, "ttft": None, "tokens": 0,
              "latency": None, "tpot": None, "error": None}

    t0 = time.perf_counter()
    first_token_time = None
    prev_token_time = None

    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=1200) as resp:
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        else:
                            if prev_token_time is not None:
                                pass  # we just count tokens for TPOT
                            prev_token_time = now
                        result["tokens"] += 1

        t1 = time.perf_counter()
        result["success"] = True
        result["latency"] = t1 - t0

        if first_token_time is not None:
            result["ttft"] = first_token_time - t0
            decode_time = t1 - first_token_time
            decode_tokens = result["tokens"] - 1
            if decode_tokens > 0 and decode_time > 0:
                result["tpot"] = decode_time / decode_tokens

    except Exception as e:
        result["error"] = str(e)

    return result


def build_prompt(input_len):
    """Build a prompt of approximately input_len tokens."""
    base_text = ("The history of artificial intelligence spans several decades, "
                 "transforming from theoretical concepts to practical applications "
                 "that affect nearly every aspect of modern life. ")
    # ~5 chars/token for English text
    prompt = base_text * (input_len // 18 + 2)
    prompt = prompt[:input_len * 5]
    return prompt


def run_benchmark(args, input_len, concurrency, num_prompts):
    """Run benchmark at a given input length and concurrency level."""
    prompt = build_prompt(input_len)

    print(f"\n{'='*70}")
    print(f"Input: {input_len} tokens | Concurrency: {concurrency} | "
          f"Prompts: {num_prompts} | Output: {args.output_len} tokens")
    print(f"{'='*70}")

    # Start GPU memory monitor
    gpu_mon = GPUMemoryMonitor(gpu_id=args.gpu_id, interval=0.3)
    gpu_mon.start()

    # Send requests
    results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request_stream, args, prompt, i)
                   for i in range(num_prompts)]
        for f in futures:
            results.append(f.result())

    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Stop GPU monitor
    gpu_mon.stop()
    gpu_results = gpu_mon.get_results()

    # Analyze
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print(f"  No successful requests! Errors: {[r['error'] for r in failed[:3]]}")
        return None

    total_tokens = sum(r["tokens"] for r in successful)
    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    tpots = [r["tpot"] for r in successful if r["tpot"] is not None]
    latencies = [r["latency"] for r in successful if r["latency"] is not None]

    output = {
        "input_len": input_len,
        "output_len": args.output_len,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "successful": len(successful),
        "failed": len(failed),
        "total_output_tokens": total_tokens,
        "wall_time_s": round(wall_time, 3),
        # Throughput
        "throughput_tok_per_s": round(total_tokens / wall_time, 2) if wall_time > 0 else 0,
        "requests_per_s": round(len(successful) / wall_time, 3) if wall_time > 0 else 0,
        # TTFT
        "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
        "ttft_p99_s": round(sorted(ttfts)[int(len(ttfts)*0.99)], 4) if len(ttfts) > 1 else (round(ttfts[0], 4) if ttfts else None),
        # TPOT
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 3) if tpots else None,
        "tpot_median_ms": round(statistics.median(tpots) * 1000, 3) if tpots else None,
        # Latency
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else None,
        "latency_median_s": round(statistics.median(latencies), 4) if latencies else None,
        # GPU Memory
        "gpu_baseline_mib": gpu_results["baseline_mib"],
        "gpu_peak_mib": gpu_results["peak_mib"],
        "gpu_delta_mib": gpu_results["delta_mib"],
        "gpu_delta_gib": round(gpu_results["delta_gib"], 2),
    }

    print(f"  Success: {len(successful)}/{num_prompts}  |  Wall: {wall_time:.2f}s")
    print(f"  Throughput : {output['throughput_tok_per_s']:.2f} tok/s  |  {output['requests_per_s']:.3f} req/s")
    if ttfts:
        print(f"  TTFT       : mean={output['ttft_mean_s']:.3f}s  median={output['ttft_median_s']:.3f}s  p99={output['ttft_p99_s']:.3f}s")
    if tpots:
        print(f"  TPOT       : mean={output['tpot_mean_ms']:.2f}ms  median={output['tpot_median_ms']:.2f}ms")
    if latencies:
        print(f"  Latency    : mean={output['latency_mean_s']:.3f}s  median={output['latency_median_s']:.3f}s")
    print(f"  GPU Memory : baseline={gpu_results['baseline_mib']}MiB  peak={gpu_results['peak_mib']}MiB  "
          f"delta={gpu_results['delta_mib']}MiB ({gpu_results['delta_gib']:.2f}GiB)")

    return output


def main():
    parser = argparse.ArgumentParser(description="vLLM server speed & memory benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8900")
    parser.add_argument("--model", required=True, help="Model path (must match vLLM server)")
    parser.add_argument("--input-len", type=int, nargs="+", default=[1024],
                        help="Input lengths to test (tokens)")
    parser.add_argument("--output-len", type=int, default=256,
                        help="Output length (tokens)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Concurrency levels to test")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Total requests per test (default: max(16, 2*concurrency))")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID for memory monitoring")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup requests")
    parser.add_argument("--output-json", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Server: {args.base_url}")
    print(f"Input lengths: {args.input_len}")
    print(f"Concurrency levels: {args.concurrency}")
    print(f"Output length: {args.output_len}")

    # Warmup
    print(f"\nWarming up with {args.warmup} requests...")
    for i in range(args.warmup):
        r = send_request_stream(args, build_prompt(256), i)
        status = "OK" if r["success"] else f"FAIL: {r['error']}"
        print(f"  Warmup {i+1}: {status}")

    # Run benchmarks
    all_results = []
    for input_len in args.input_len:
        for c in args.concurrency:
            num_prompts = args.num_prompts or max(16, c * 2)
            result = run_benchmark(args, input_len, c, num_prompts)
            if result:
                all_results.append(result)
            else:
                print(f"  SKIP: input_len={input_len} concurrency={c} failed, skipping remaining concurrency for this input_len")
                break
            # Cool down
            time.sleep(3)

    # Print summary table
    if all_results:
        print(f"\n{'='*100}")
        print(f"SUMMARY")
        print(f"{'='*100}")
        print(f"{'Input':>7} | {'Conc':>4} | {'Throughput':>10} | {'TTFT':>8} | {'TPOT':>8} | {'Latency':>8} | {'PeakMem':>8} | {'ΔMem':>7}")
        print(f"{'tokens':>7} | {'':4} | {'tok/s':>10} | {'s':>8} | {'ms':>8} | {'s':>8} | {'MiB':>8} | {'GiB':>7}")
        print("-" * 100)
        for r in all_results:
            ttft = f"{r['ttft_mean_s']:.3f}" if r.get('ttft_mean_s') else "N/A"
            tpot = f"{r['tpot_mean_ms']:.2f}" if r.get('tpot_mean_ms') else "N/A"
            lat = f"{r['latency_mean_s']:.3f}" if r.get('latency_mean_s') else "N/A"
            print(f"{r['input_len']:>7} | {r['concurrency']:>4} | {r['throughput_tok_per_s']:>10.2f} | "
                  f"{ttft:>8} | {tpot:>8} | {lat:>8} | {r['gpu_peak_mib']:>8} | {r['gpu_delta_gib']:>7.2f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
