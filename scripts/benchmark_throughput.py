#!/usr/bin/env python3
"""
Standalone throughput benchmark with ksanal-aligned metric logic.

- Compatible with OpenAI Chat Completions endpoint
- CSV input benchmark (customize/sharegpt500)
- Stream metrics: TTFT / ITL / TPOT
- Throughput metric matches ksanal logic
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp


# (prompt_len_chars, output_len_chars, input_token_num, output_token_num,
#  request_latency_s, first_token_latency_s, inter_token_latencies_s)
REQUEST_LATENCY: List[Tuple[int, int, int, int, float, float, List[float]]] = []


@dataclass
class BenchmarkMetrics:
    request_rate: float = 0.0
    concurrency: int = 1
    total_latency: float = 0.0
    request_throughput: float = 0.0
    avg_latency: float = 0.0
    avg_input_chars: float = 0.0
    avg_output_chars: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_tokens_per_sec: float = 0.0
    percentile_latency: List[Tuple[int, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Request rate: {self.request_rate:.3f} requests/s",
            f"Concurrency requests: {self.concurrency}",
            f"Total latency: {self.total_latency:.3f} s",
            f"Request throughput: {self.request_throughput:.3f} requests/s",
            f"Average latency: {self.avg_latency:.3f} s",
            f"Average input len: {self.avg_input_chars:.3f} chars",
            f"Average output len: {self.avg_output_chars:.3f} chars",
            f"Average input len: {self.avg_input_tokens:.3f} tokens",
            f"Average output len: {self.avg_output_tokens:.3f} tokens",
            f"Token throughput: {self.avg_tokens_per_sec:.3f} tokens/s",
        ]
        lines += [f"P{p} latency: {v:.3f} s" for p, v in self.percentile_latency]
        return "\n".join(lines)


@dataclass
class BenchmarkStreamMetrics:
    avg_first_token_latency: float = 0.0
    median_first_token_latency: float = 0.0
    percentiles_first_token_latency: List[Tuple[int, float]] = field(default_factory=list)

    avg_inter_token_latency: float = 0.0
    median_inter_token_latency: float = 0.0
    percentiles_inter_token_latency: List[Tuple[int, float]] = field(default_factory=list)

    avg_latency_per_out_token: float = 0.0
    median_latency_per_out_token: float = 0.0
    percentiles_latency_per_out_token: List[Tuple[int, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Average TTFT: {self.avg_first_token_latency:.3f} s",
            f"Median TTFT: {self.median_first_token_latency:.3f} s",
        ]
        lines += [f"P{p} TTFT: {v:.3f} s" for p, v in self.percentiles_first_token_latency]
        lines += [
            f"Average ITL: {self.avg_inter_token_latency:.5f} s",
            f"Median ITL: {self.median_inter_token_latency:.5f} s",
        ]
        lines += [f"P{p} ITL: {v:.5f} s" for p, v in self.percentiles_inter_token_latency]
        lines += [
            f"Average TPOT: {self.avg_latency_per_out_token:.5f} s",
            f"Median TPOT: {self.median_latency_per_out_token:.5f} s",
        ]
        lines += [f"P{p} TPOT: {v:.5f} s" for p, v in self.percentiles_latency_per_out_token]
        return "\n".join(lines)


def percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def args_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Throughput benchmark (ksanal-aligned metrics)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=str, default="8902", help="server port")

    parser.add_argument("--dataset_name", type=str, default="customize", choices=["customize", "sharegpt500"])
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/cakejiang/github/GSQ/benchmark/benchmark_input.csv",
        help="input csv path",
    )
    parser.add_argument("--col_idx", type=int, default=0, help="which column to read from csv")
    parser.add_argument("--output_csv", type=str, default=None, help="generated text output csv")
    parser.add_argument("--perf_csv", type=str, default=None, help="performance output csv")

    parser.add_argument("--request_rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random", action="store_true", help="randomized inter-arrival")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--percentiles", nargs="+", type=int, default=[99])

    parser.add_argument("--request_rate_step", type=float, default=1.0)
    parser.add_argument("--request_rate_num_iters", type=int, default=1)
    parser.add_argument("--max_avg_latency", type=float, default=float("inf"))
    parser.add_argument("--max_first_token_latency", type=float, default=float("inf"))

    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup_num_iters", type=int, default=0)
    parser.add_argument("--repeat_num_iters", type=int, default=1)
    parser.add_argument("--mode", type=str, default="async", choices=["async", "sync"])

    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--backend", type=str, default="openai-chat", choices=["openai-chat"])

    parser.add_argument("--prompt_num", type=int, default=0)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--model_type", type=str, default="empty")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--client_timeout", type=int, default=30 * 60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--show_decode_token_throughput",
        action="store_true",
        help="if set, token throughput only counts decode tokens (output-1)",
    )

    args = parser.parse_args()
    args.host = args.host.split(",") if "," in args.host else [args.host]
    args.port = args.port.split(",") if "," in args.port else [args.port]
    return args


def read_from_csv(csv_file: str, col_idx: int = 0, remove_head: bool = True) -> List[str]:
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        if remove_head:
            next(reader, None)
        return [row[col_idx] for row in reader if len(row) > col_idx]


def adjust_list_length(inputs: List[str], args: argparse.Namespace) -> List[str]:
    if args.prompt_num == 0:
        args.prompt_num = len(inputs)
        return inputs

    if args.prompt_num > len(inputs):
        repeat_times = args.prompt_num // len(inputs)
        if len(inputs) * repeat_times != args.prompt_num:
            raise ValueError(f"prompt_num={args.prompt_num} 无法整倍数扩展当前长度={len(inputs)}")
        return inputs * repeat_times

    return inputs[:args.prompt_num]


def construct_request_data(prompt: str, args: argparse.Namespace) -> bytes:
    data = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.topp,
        "max_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "stream": args.stream,
    }
    if args.stream:
        data["stream_options"] = {"include_usage": True}
    if args.topk > 0:
        data["top_k"] = args.topk
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


async def generate_req_data_async(
    input_requests: List[Tuple[str, bytes]],
    request_rate: float,
    concurrency: int,
    randomized: bool,
) -> AsyncGenerator[Tuple[int, Tuple[str, bytes]], None]:
    input_requests = enumerate(input_requests)
    request_num = 0
    total_requests = 0
    start_time = time.time()

    for req_id, request in input_requests:
        yield req_id, request

        request_num += 1
        total_requests += 1
        if request_rate == float("inf"):
            continue
        if request_num < concurrency:
            continue

        request_num = 0
        expected_time = total_requests / request_rate
        actual_time = time.time() - start_time
        interval_adjustment = expected_time - actual_time

        if randomized:
            interval = random.expovariate(request_rate / max(concurrency, 1)) + interval_adjustment
        else:
            interval = 1.0 / (request_rate / max(concurrency, 1)) + interval_adjustment

        await asyncio.sleep(max(interval, 0.0))


def _parse_sse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    if line.startswith("data: "):
        line = line[6:]
    if line == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


async def send_request_async(
    args: argparse.Namespace,
    prompt: str,
    req_data: bytes,
    api_url: str,
    req_id: int,
    result_list: List[str],
) -> None:
    headers = {
        "User-Agent": "Benchmark Client",
        "Content-Type": "application/json",
        "req_id": str(req_id),
    }

    api_url = api_url.replace("##host##", args.host[req_id % len(args.host)])
    api_url = api_url.replace("##port##", args.port[req_id % len(args.port)])

    timeout = aiohttp.ClientTimeout(total=args.client_timeout)

    retries = 0
    while True:
        request_start_time = time.perf_counter()
        first_token_latency = 0.0
        inter_token_latencies: List[float] = []
        most_recent_timestamp = request_start_time

        output: Optional[dict] = None
        prompt_tokens = 0
        completion_tokens = 0
        server_text = ""

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(api_url, headers=headers, data=req_data) as response:
                    if response.status != 200:
                        body = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {body[:500]}")

                    if args.stream:
                        chunk_acc = ""
                        async for chunk_bytes, _ in response.content.iter_chunks():
                            timestamp = time.perf_counter()
                            chunk_bytes = chunk_bytes.strip(b"\x00")
                            if not chunk_bytes:
                                continue
                            try:
                                chunk = chunk_bytes.decode("utf-8")
                            except UnicodeDecodeError:
                                continue

                            chunk_acc += chunk
                            lines = chunk_acc.split("\n")
                            chunk_acc = lines[-1]

                            for line in lines[:-1]:
                                obj = _parse_sse_line(line)
                                if obj is None or obj.get("__done__"):
                                    continue

                                output = obj
                                choice = (obj.get("choices") or [{}])[0]
                                delta = choice.get("delta") or {}
                                delta_text = delta.get("reasoning_content", "") + (delta.get("content", "") or "")

                                if delta_text:
                                    server_text += delta_text
                                    if first_token_latency == 0.0:
                                        first_token_latency = timestamp - request_start_time
                                    else:
                                        inter_token_latencies.append(timestamp - most_recent_timestamp)
                                    most_recent_timestamp = timestamp

                                usage = obj.get("usage") or {}
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)

                        if chunk_acc.strip():
                            obj = _parse_sse_line(chunk_acc)
                            if obj and not obj.get("__done__"):
                                output = obj
                                usage = obj.get("usage") or {}
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                    else:
                        output = await response.json(content_type=None)
                        msg = (output.get("choices") or [{}])[0].get("message") or {}
                        server_text = (msg.get("reasoning_content", "") or "") + (msg.get("content", "") or "")
                        usage = output.get("usage") or {}
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        first_token_latency = time.perf_counter() - request_start_time
                        most_recent_timestamp = time.perf_counter()

            if output is not None and "error" not in output:
                request_end_time = most_recent_timestamp if args.stream else time.perf_counter()

                if completion_tokens <= 0:
                    completion_tokens = max(len(server_text) // 4, 0)
                if prompt_tokens <= 0:
                    prompt_tokens = max(len(prompt) // 4, 0)

                request_latency = request_end_time - request_start_time
                result_list[req_id] = server_text
                REQUEST_LATENCY.append(
                    (
                        len(prompt),
                        len(server_text) if len(server_text) > 0 else 1,
                        prompt_tokens,
                        completion_tokens,
                        request_latency,
                        first_token_latency,
                        inter_token_latencies,
                    )
                )
                return

        except Exception as e:
            err = str(e)
            if len(err) > 400:
                err = err[:400] + "..."
            print(f"[WARN] req_id={req_id} attempt={retries + 1}/{args.max_retries + 1} failed: {err}")

        retries += 1
        if retries > args.max_retries:
            raise RuntimeError(f"The request(req_id={req_id}) failed after retries")


async def benchmark_async(args: argparse.Namespace, api_url: str,
                          inputs: List[Tuple[str, bytes]]) -> List[str]:
    tasks: List[asyncio.Task] = []
    result_list = [""] * len(inputs)

    # 保留 ksanal 的请求到达逻辑，但用 semaphore 真正限制并发，避免 inf+async 洪峰打挂。
    sem = asyncio.Semaphore(max(args.concurrency, 1))

    async def _wrapped_send(req_id: int, prompt: str, req_data: bytes) -> None:
        async with sem:
            await send_request_async(args, prompt, req_data, api_url, req_id, result_list)

    async for req_id, (prompt, req_data) in generate_req_data_async(
        inputs,
        args.request_rate,
        args.concurrency,
        args.random,
    ):
        task = asyncio.create_task(_wrapped_send(req_id, prompt, req_data))
        tasks.append(task)

    await asyncio.gather(*tasks)
    return result_list


async def benchmark_sync(args: argparse.Namespace, api_url: str,
                         inputs: List[Tuple[str, bytes]]) -> List[str]:
    result_list = [""] * len(inputs)
    async for req_id, (prompt, req_data) in generate_req_data_async(
        inputs,
        args.request_rate,
        args.concurrency,
        args.random,
    ):
        await send_request_async(args, prompt, req_data, api_url, req_id, result_list)
    return result_list


def run_benchmark(args: argparse.Namespace, api_url: str,
                  inputs: List[Tuple[str, bytes]]) -> List[str]:
    if args.mode == "async":
        return asyncio.run(benchmark_async(args, api_url, inputs))
    return asyncio.run(benchmark_sync(args, api_url, inputs))


def search_request_rate(args: argparse.Namespace,
                        request_rate_list: List[Tuple[float, float, float]]) -> float:
    def round_to_tenth(number: float) -> float:
        return max(round(number * 10) / 10, 0.1)

    step = len(request_rate_list)
    request_rate = -1.0

    if step < args.request_rate_num_iters:
        request_rate = args.request_rate + (args.request_rate_step if step > 0 else 0)
    elif args.max_avg_latency != float("inf") or args.max_first_token_latency != float("inf"):
        request_rate_list.sort(key=lambda x: x[0])

        if request_rate_list[-1][1] <= args.max_avg_latency and request_rate_list[-1][2] <= args.max_first_token_latency:
            request_rate = min(request_rate_list[-1][0] * 2, float(args.prompt_num))
        elif request_rate_list[0][1] > args.max_avg_latency or request_rate_list[0][2] > args.max_first_token_latency:
            request_rate = round_to_tenth(request_rate_list[0][0] / 2)
        else:
            rate_left = max(
                filter(lambda x: x[1] <= args.max_avg_latency and x[2] <= args.max_first_token_latency, request_rate_list),
                key=lambda x: x[0],
            )[0]
            rate_right = min(
                filter(lambda x: x[1] > args.max_avg_latency or x[2] > args.max_first_token_latency, request_rate_list),
                key=lambda x: x[0],
            )[0]
            request_rate = round_to_tenth((rate_left + rate_right) / 2)

        if any(item[0] == request_rate for item in request_rate_list):
            print(f"Duplicate request rate detected: {request_rate}. Terminating the search.")
            request_rate = -1.0

    return request_rate


def main(args: argparse.Namespace) -> None:
    global REQUEST_LATENCY

    random.seed(args.seed)

    api_url = "http://##host##:##port##/v1/chat/completions"

    if args.dataset_name == "sharegpt500":
        sharegpt_path = Path(args.dataset_path).parent / "share_gpt_500.csv"
        dataset_path = sharegpt_path
    else:
        dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path not found: {dataset_path}")

    inputs = read_from_csv(str(dataset_path), args.col_idx)
    if args.shuffle:
        random.shuffle(inputs)
    inputs = adjust_list_length(inputs, args)
    req_inputs = [(prompt, construct_request_data(prompt, args)) for prompt in inputs]

    perf_result_list: List[Tuple[BenchmarkMetrics, BenchmarkStreamMetrics]] = []
    request_rate_list: List[Tuple[float, float, float]] = []
    all_result_list: List[List[str]] = []

    while True:
        metrics = BenchmarkMetrics()
        metrics.request_rate = search_request_rate(args, request_rate_list)
        args.request_rate = metrics.request_rate
        if metrics.request_rate == -1:
            break

        metrics.concurrency = args.concurrency

        for i in range(args.warmup_num_iters):
            print(f"Start warmup iteration {i} with request rate {metrics.request_rate:.3f}")
            run_benchmark(args, api_url, req_inputs)

        REQUEST_LATENCY.clear()

        bench_start = time.perf_counter()
        for i in range(args.repeat_num_iters):
            print(f"Start profile iteration {i} with request rate {metrics.request_rate:.3f}")
            result_list = run_benchmark(args, api_url, req_inputs)
            all_result_list.append(result_list)
        bench_end = time.perf_counter()

        metrics.total_latency = (bench_end - bench_start) / args.repeat_num_iters
        metrics.request_throughput = len(req_inputs) / metrics.total_latency if metrics.total_latency > 0 else 0.0

        latencies = [x[4] for x in REQUEST_LATENCY]
        input_chars = [x[0] for x in REQUEST_LATENCY]
        output_chars = [x[1] for x in REQUEST_LATENCY]
        input_tokens = [x[2] for x in REQUEST_LATENCY]
        output_tokens = [x[3] for x in REQUEST_LATENCY]

        metrics.avg_latency = statistics.mean(latencies) if latencies else 0.0
        metrics.percentile_latency = [(p, percentile(latencies, p)) for p in args.percentiles]
        metrics.avg_input_chars = statistics.mean(input_chars) if input_chars else 0.0
        metrics.avg_output_chars = statistics.mean(output_chars) if output_chars else 0.0
        metrics.avg_input_tokens = statistics.mean(input_tokens) if input_tokens else 0.0
        metrics.avg_output_tokens = statistics.mean(output_tokens) if output_tokens else 0.0

        # ksanal 同口径吞吐逻辑
        summary_token = metrics.avg_input_tokens + metrics.avg_output_tokens
        if args.show_decode_token_throughput:
            summary_token = max(metrics.avg_output_tokens - 1, 0)

        if metrics.total_latency > 0 and REQUEST_LATENCY:
            metrics.avg_tokens_per_sec = summary_token * len(REQUEST_LATENCY) / metrics.total_latency / args.repeat_num_iters
        else:
            metrics.avg_tokens_per_sec = 0.0

        print(metrics)

        stream_metrics = BenchmarkStreamMetrics()
        if args.stream and REQUEST_LATENCY:
            first_token_latencies = [x[5] for x in REQUEST_LATENCY if x[5] > 0]
            if first_token_latencies:
                stream_metrics.avg_first_token_latency = statistics.mean(first_token_latencies)
                stream_metrics.median_first_token_latency = statistics.median(first_token_latencies)
                stream_metrics.percentiles_first_token_latency = [
                    (p, percentile(first_token_latencies, p)) for p in args.percentiles
                ]

            inter_token_latencies = [itl for x in REQUEST_LATENCY for itl in x[6]]
            if inter_token_latencies:
                stream_metrics.avg_inter_token_latency = statistics.mean(inter_token_latencies)
                stream_metrics.median_inter_token_latency = statistics.median(inter_token_latencies)
                stream_metrics.percentiles_inter_token_latency = [
                    (p, percentile(inter_token_latencies, p)) for p in args.percentiles
                ]

            latencies_per_out_token = [
                (lat - ttft) / (out_tok - 1)
                for _, _, _, out_tok, lat, ttft, _ in REQUEST_LATENCY
                if out_tok > 1 and lat > ttft > 0
            ]
            if latencies_per_out_token:
                stream_metrics.avg_latency_per_out_token = statistics.mean(latencies_per_out_token)
                stream_metrics.median_latency_per_out_token = statistics.median(latencies_per_out_token)
                stream_metrics.percentiles_latency_per_out_token = [
                    (p, percentile(latencies_per_out_token, p)) for p in args.percentiles
                ]

            print(stream_metrics)

        perf_result_list.append((metrics, stream_metrics))
        request_rate_list.append((metrics.request_rate, metrics.avg_latency, stream_metrics.avg_first_token_latency))
        REQUEST_LATENCY.clear()

    if args.output_csv and all_result_list:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for text in all_result_list[-1]:
                writer.writerow([text.replace("</s>", "")])

    if args.perf_csv and perf_result_list:
        with open(args.perf_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "Request rate", "Concurrency", "Total latency", "Request throughput", "Avg latency",
                "Avg input chars", "Avg output chars", "Avg input tokens", "Avg output tokens", "Token throughput",
            ]
            header.extend([f"P{p} latency" for p in args.percentiles])
            if args.stream:
                header.extend(
                    ["Avg TTFT", "Median TTFT"]
                    + [f"P{p} TTFT" for p in args.percentiles]
                    + ["Avg ITL", "Median ITL"]
                    + [f"P{p} ITL" for p in args.percentiles]
                    + ["Avg TPOT", "Median TPOT"]
                    + [f"P{p} TPOT" for p in args.percentiles]
                )
            writer.writerow(header)

            for metrics, stream_metrics in perf_result_list:
                row: List[str] = []

                def append_values(metric_obj):
                    for value in metric_obj.__dict__.values():
                        if isinstance(value, list):
                            row.extend([f"{pair[1]:.5f}" for pair in value])
                        else:
                            row.append(f"{value:.5f}")

                append_values(metrics)
                if args.stream:
                    append_values(stream_metrics)
                writer.writerow(row)


if __name__ == "__main__":
    try:
        import uvloop  # type: ignore

        uvloop.install()
    except Exception:
        pass

    _args = args_config()
    main(_args)
