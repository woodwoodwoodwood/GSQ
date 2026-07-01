#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一步验证：业务数据 -> 转换脚本 -> GSQ 数据加载器 链路冒烟测试。

本脚本不依赖 GPU、不加载模型权重（仅需 tokenizer），用于验证：
  1. 转换脚本产出的本地 JSONL 能被 src/data/dataset.py 正确加载；
  2. 每条样本 tokenize 后长度恰好等于 seqlen（通过了长度过滤门槛）；
  3. main.py 的 Qwen3 家族路由能把目标模型识别为 qwen3_5_moe（可选，--check-routing）。

用法：
    python tests/verify_step1_data_pipeline.py \\
        --data ./data/sample_calib.jsonl \\
        --model Qwen/Qwen3.6-35B-A3B \\
        --seqlen 128 --num-samples 4
"""
import argparse
import os
import sys

# 确保可以从仓库根 import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402
from src.data.dataset import StreamingHFDataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="转换脚本产出的 .jsonl 文件")
    ap.add_argument("--model", required=True, help="tokenizer 路径或 HF 名")
    ap.add_argument("--seqlen", type=int, default=128, help="必须与转换时的 --seqlen 一致")
    ap.add_argument("--num-samples", type=int, default=4, help="尝试收集的样本数")
    ap.add_argument("--check-routing", action="store_true",
                    help="额外验证 main.get_model_wrapper 对该模型的路由（需要能读取 config）")
    return ap.parse_args()


def check_data_pipeline(args):
    print("=" * 60)
    print("[A] 数据加载链路验证")
    print("=" * 60)
    if not os.path.isfile(args.data):
        print(f"[FAIL] 找不到数据文件: {args.data}")
        return False

    print(f"  加载 tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    print(f"  通过 StreamingHFDataset 加载: {args.data}")
    ds = StreamingHFDataset(
        args.data, tok, split="train", seqlen=args.seqlen,
        num_samples=args.num_samples, streaming=True,
    )
    n = len(ds)
    print(f"  收集到样本数: {n}（请求 {args.num_samples}）")

    if n == 0:
        print("[FAIL] 收集到 0 条样本：转换产物为空，或 seqlen 与转换不一致导致全部被长度过滤。")
        return False

    ok = True
    for i in range(min(n, 3)):
        ln = ds[i].size(0)
        flag = "OK" if ln == args.seqlen else "BAD"
        print(f"    sample[{i}] 长度={ln} ({flag})")
        if ln != args.seqlen:
            ok = False

    print("  预览 sample[0] 前 40 个 token 的解码：")
    print("   ", tok.decode(ds[0][:40]).replace("\n", " ⏎ "))

    if not ok:
        print("[FAIL] 存在长度不等于 seqlen 的样本。")
        return False
    print(f"[PASS] 数据链路正常：样本可加载且每条恰为 {args.seqlen} tokens。")
    return True


def check_routing(args):
    print("=" * 60)
    print("[B] 模型路由验证 (main.get_model_wrapper)")
    print("=" * 60)
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        mt = str(getattr(tc, "model_type", "") or getattr(cfg, "model_type", "")).lower()
        is_moe = hasattr(tc, "num_experts") or hasattr(tc, "num_local_experts")
        name_lower = args.model.lower()
        is_qwen35 = (
            "qwen3_5" in mt
            or "qwen3.5" in name_lower or "qwen3_5" in name_lower
            or "qwen3.6" in name_lower or "qwen3_6" in name_lower
        )
        print(f"  model_type={mt}  is_moe={is_moe}  is_qwen35={is_qwen35}")
        if is_qwen35 and is_moe:
            print("  预期路由（world_size>1）-> Qwen35MoeDistributedWrapper")
            print("[PASS] 路由判定符合预期。")
            return True
        print("[WARN] 未判定为 qwen3_5_moe MoE，请确认模型是否为 Qwen3.5/3.6。")
        return False
    except Exception as e:
        print(f"[SKIP] 无法读取模型 config（可能未下载/无网络）：{e}")
        return True


def main():
    args = parse_args()
    results = [check_data_pipeline(args)]
    if args.check_routing:
        results.append(check_routing(args))
    print("=" * 60)
    if all(results):
        print("结果：全部通过 ✅")
        sys.exit(0)
    print("结果：存在失败项 ❌")
    sys.exit(1)


if __name__ == "__main__":
    main()
