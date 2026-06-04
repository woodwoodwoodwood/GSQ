#!/usr/bin/env python3
"""One-click comparison for two vLLM benchmark runs + Nsight Systems kernel CSV.

Inputs:
- Two benchmark JSON files or directories containing benchmark JSON files.
- Two nsys CSV files exported from:
  nsys stats --report cuda_api_sum,gpu_kern_sum,gpu_mem_time_sum,gpu_mem_size_sum --format csv

Outputs:
- Markdown comparison table for throughput/latency changes.
- Top bottleneck kernels from gpu_kern_sum for both runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class KernelRow:
    name: str
    time_pct: float | None
    total_time_ns: float | None
    instances: float | None


def to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)

    s = str(x).strip().strip('"').replace(",", "")
    if not s or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pct_delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None or rhs == 0:
        return None
    return (lhs - rhs) / rhs * 100.0


def fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {path}")
    data["__file__"] = str(path)
    return data


def collect_benchmark_json(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        return [load_json(input_path)]

    files = sorted(input_path.rglob("*.json"))
    out: list[dict[str, Any]] = []
    for p in files:
        try:
            out.append(load_json(p))
        except Exception:
            # Skip non-benchmark JSON files.
            continue
    return out


def pick_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No valid benchmark JSON found.")

    def score(r: dict[str, Any]) -> tuple[float, float]:
        out_tp = to_float(r.get("output_throughput")) or -1.0
        total_tp = to_float(r.get("total_token_throughput")) or -1.0
        return (out_tp, total_tp)

    return max(records, key=score)


def normalize_header(h: str) -> str:
    return h.strip().lower()


def detect_ns_to_unit_factor(total_time_col: str) -> float:
    c = total_time_col.lower()
    if "(ns)" in c:
        return 1.0
    if "(us)" in c:
        return 1e3
    if "(ms)" in c:
        return 1e6
    if "(s)" in c:
        return 1e9
    # Fallback: assume ns.
    return 1.0


def parse_nsys_gpu_kernels(csv_path: Path) -> list[KernelRow]:
    rows: list[KernelRow] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        content = list(reader)

    current_section = ""
    i = 0
    while i < len(content):
        row = content[i]
        row0 = row[0].strip() if row else ""
        row0_l = row0.lower()

        if len(row) == 1 and row0:
            current_section = row0_l

        # We only parse tables under gpu kernel summary section.
        if "gpu_kern_sum" in current_section and row:
            headers = [normalize_header(x) for x in row]
            if "name" in headers and any("time (%)" == h for h in headers):
                idx_name = headers.index("name")
                idx_pct = headers.index("time (%)")

                total_time_idx = None
                total_time_factor = 1.0
                for idx, h in enumerate(headers):
                    if h.startswith("total time"):
                        total_time_idx = idx
                        total_time_factor = detect_ns_to_unit_factor(h)
                        break

                instances_idx = headers.index("instances") if "instances" in headers else None

                j = i + 1
                while j < len(content):
                    r = content[j]
                    if not r or all(not c.strip() for c in r):
                        break

                    # New section marker.
                    if len(r) == 1 and r[0].strip() and ":" in r[0]:
                        break

                    name = r[idx_name].strip() if idx_name < len(r) else ""
                    if not name or set(name) <= {"-", "="}:
                        j += 1
                        continue

                    pct = to_float(r[idx_pct] if idx_pct < len(r) else None)
                    total = (
                        to_float(r[total_time_idx]) * total_time_factor
                        if total_time_idx is not None and total_time_idx < len(r)
                        else None
                    )
                    inst = (
                        to_float(r[instances_idx])
                        if instances_idx is not None and instances_idx < len(r)
                        else None
                    )

                    rows.append(KernelRow(name=name, time_pct=pct, total_time_ns=total, instances=inst))
                    j += 1

                i = j
                continue

        i += 1

    rows.sort(key=lambda x: (x.total_time_ns or -1.0, x.time_pct or -1.0), reverse=True)
    return rows


def get_metric(rec: dict[str, Any], key: str) -> float | None:
    return to_float(rec.get(key))


def build_metric_rows(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[tuple[str, float | None, float | None, float | None]]:
    keys = [
        # Throughput
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        # Latency
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "p50_e2el_ms",
        "p95_e2el_ms",
        "p99_e2el_ms",
    ]

    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for k in keys:
        l = get_metric(lhs, k)
        r = get_metric(rhs, k)
        rows.append((k, l, r, pct_delta(l, r)))
    return rows


def render_report(
    lhs_label: str,
    rhs_label: str,
    lhs_bench: dict[str, Any],
    rhs_bench: dict[str, Any],
    lhs_kernels: list[KernelRow],
    rhs_kernels: list[KernelRow],
    top_n: int,
) -> str:
    lines: list[str] = []

    lines.append("# Inference Comparison Summary")
    lines.append("")
    lines.append(f"- Left run: **{lhs_label}**")
    lines.append(f"  - benchmark source: `{lhs_bench.get('__file__', '-')}`")
    lines.append(f"- Right run: **{rhs_label}**")
    lines.append(f"  - benchmark source: `{rhs_bench.get('__file__', '-')}`")
    lines.append("")
    lines.append("> Delta% formula: `(left - right) / right * 100%`.")
    lines.append("> Throughput: positive means left is higher; Latency: negative means left is lower/faster.")
    lines.append("")

    lines.append("## Throughput & Latency Table")
    lines.append("")
    lines.append(f"| Metric | {lhs_label} | {rhs_label} | Delta% (left vs right) |")
    lines.append("|---|---:|---:|---:|")

    for metric, l, r, d in build_metric_rows(lhs_bench, rhs_bench):
        lines.append(f"| {metric} | {fmt_num(l)} | {fmt_num(r)} | {fmt_pct(d)} |")

    lines.append("")
    lines.append("## Top Bottleneck Kernels (from nsys gpu_kern_sum)")
    lines.append("")

    top_l = lhs_kernels[:top_n]
    top_r = rhs_kernels[:top_n]

    lines.append(f"### {lhs_label} top-{top_n}")
    lines.append("| Rank | Kernel | Time % | Total Time (ms) | Instances |")
    lines.append("|---:|---|---:|---:|---:|")
    for idx, k in enumerate(top_l, 1):
        total_ms = k.total_time_ns / 1e6 if k.total_time_ns is not None else None
        lines.append(
            f"| {idx} | `{k.name}` | {fmt_num(k.time_pct)} | {fmt_num(total_ms)} | {fmt_num(k.instances, 0)} |"
        )

    lines.append("")
    lines.append(f"### {rhs_label} top-{top_n}")
    lines.append("| Rank | Kernel | Time % | Total Time (ms) | Instances |")
    lines.append("|---:|---|---:|---:|---:|")
    for idx, k in enumerate(top_r, 1):
        total_ms = k.total_time_ns / 1e6 if k.total_time_ns is not None else None
        lines.append(
            f"| {idx} | `{k.name}` | {fmt_num(k.time_pct)} | {fmt_num(total_ms)} | {fmt_num(k.instances, 0)} |"
        )

    lines.append("")
    lines.append("## Quick Bottleneck Readout")

    l0 = top_l[0] if top_l else None
    r0 = top_r[0] if top_r else None

    if l0 is None or r0 is None:
        lines.append("- Unable to parse top kernel rows from one or both nsys CSV files.")
    else:
        lines.append(
            f"- {lhs_label} #1: `{l0.name}` (Time%={fmt_num(l0.time_pct)}, Total={fmt_num((l0.total_time_ns or 0)/1e6)} ms)"
        )
        lines.append(
            f"- {rhs_label} #1: `{r0.name}` (Time%={fmt_num(r0.time_pct)}, Total={fmt_num((r0.total_time_ns or 0)/1e6)} ms)"
        )
        same = l0.name == r0.name
        lines.append(f"- Same top kernel: **{'Yes' if same else 'No'}**")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two benchmark runs and nsys kernel summaries.")

    p.add_argument("--lhs-bench", required=True, help="Left benchmark JSON file or directory.")
    p.add_argument("--rhs-bench", required=True, help="Right benchmark JSON file or directory.")
    p.add_argument("--lhs-nsys", required=True, help="Left nsys CSV path.")
    p.add_argument("--rhs-nsys", required=True, help="Right nsys CSV path.")

    p.add_argument("--lhs-label", default="humming-2bit", help="Label for left run.")
    p.add_argument("--rhs-label", default="marlin-int4", help="Label for right run.")
    p.add_argument("--top-n", type=int, default=8, help="Top-N kernels to display.")
    p.add_argument("--output", default="", help="Optional output markdown file path.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    lhs_bench_path = Path(args.lhs_bench).expanduser().resolve()
    rhs_bench_path = Path(args.rhs_bench).expanduser().resolve()
    lhs_nsys_path = Path(args.lhs_nsys).expanduser().resolve()
    rhs_nsys_path = Path(args.rhs_nsys).expanduser().resolve()

    lhs_records = collect_benchmark_json(lhs_bench_path)
    rhs_records = collect_benchmark_json(rhs_bench_path)

    lhs_bench = pick_best_record(lhs_records)
    rhs_bench = pick_best_record(rhs_records)

    lhs_kernels = parse_nsys_gpu_kernels(lhs_nsys_path)
    rhs_kernels = parse_nsys_gpu_kernels(rhs_nsys_path)

    report = render_report(
        lhs_label=args.lhs_label,
        rhs_label=args.rhs_label,
        lhs_bench=lhs_bench,
        rhs_bench=rhs_bench,
        lhs_kernels=lhs_kernels,
        rhs_kernels=rhs_kernels,
        top_n=max(1, args.top_n),
    )

    print(report)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n[OK] Report written to: {out_path}")


if __name__ == "__main__":
    main()
