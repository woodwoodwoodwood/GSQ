#!/usr/bin/env python3
"""Render the Marvis business jsonl into a plain-text calib.txt for llama-imatrix.

Each agent trace is rendered with the model's own chat_template (incl. tools),
so the imatrix reflects the real agent activation distribution. Documents are
separated by a blank line so imatrix chunking never mixes two traces.

Usage:
  python build_calib_txt.py \
      --model /data1/models/Qwen3.6-35B-A3B-FP16 \
      --jsonl /usr/local/app/GSQ/qwen3.6_trace_v2_merged_train_masked.jsonl \
      --out calib.txt \
      --max-samples 0            # 0 = use all
"""
import argparse
import json

from transformers import AutoTokenizer

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "function_call": "assistant",
    "observation": "tool",
    "tool": "tool",
}


def render_example(example, tokenizer):
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": str(example["system"])})
    for msg in example.get("conversations", []):
        role = ROLE_MAP.get(msg.get("from", ""), "user")
        messages.append({"role": role, "content": str(msg.get("value", ""))})

    tools = None
    raw_tools = example.get("tools")
    if raw_tools:
        try:
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        except Exception:
            tools = None

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
    # Fallback: plain concat.
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (for chat_template)")
    ap.add_argument("--jsonl", required=True, help="Marvis business jsonl path")
    ap.add_argument("--out", default="calib.txt")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all lines")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_ok, n_tok = 0, 0
    with open(args.jsonl, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.max_samples and n_ok >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            text = render_example(ex, tok)
            if not text or not text.strip():
                continue
            fout.write(text.rstrip() + "\n\n")  # blank line separates traces
            n_ok += 1
            n_tok += len(tok(text).input_ids)
            if n_ok % 200 == 0:
                print(f"  rendered {n_ok} traces, ~{n_tok} tokens", flush=True)

    print(f"done: {n_ok} traces -> {args.out}, ~{n_tok} tokens total", flush=True)


if __name__ == "__main__":
    main()
