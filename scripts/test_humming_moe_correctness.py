# Numerical correctness reproducer for the KsanaLLM humming int2 MoE path.
#
# Goal: isolate WHERE KsanaLLM's humming 2bit MoE goes wrong (it currently emits
# finite-but-garbage tokens) by checking, with REAL layer-0/expert-0 weights:
#   1) Does the humming w13/w2 GEMM (dequant + scale + zero_point) match a
#      bf16 dequantized reference?  -> validates the kernel + weight feeding.
#   2) Which silu convention is correct for the [up, gate] merged w13?
#        A = silu(first_half) * second_half   (== KsanaLLM InvokeSiluAndMul today)
#        B = silu(second_half) * first_half
#      -> tells us whether the humming layer silu's the wrong half.
#
# Reference dequant uses humming's own humming.utils.weight.dequantize_weight,
# so the formula is guaranteed to match the kernel's quant scheme.
#
# Run:  /usr/local/app/GSQ/.venv/bin/python scripts/test_humming_moe_correctness.py

import json
import torch
import torch.nn.functional as F
from safetensors import safe_open

from humming import dtypes
from humming.layer import HummingLayer
from humming.schema.humming import HummingWeightSchema
from humming.utils.weight import dequantize_weight

MODEL_DIR = "/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming"
LAYER, EXPERT = 0, 0
HIDDEN = 2048          # K for w13, N for w2
INTER = 768            # per-proj N for gate/up, K for w2
GROUP_SIZE = 128
B_DTYPE = dtypes.DataType.from_str("uint2")
DTYPE = torch.bfloat16
DEVICE = "cuda"

print(f"device: {torch.cuda.get_device_name(0)}")
idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]


def load(prefix):
    out = {}
    for suf in ["weight", "weight_scale", "zero_point"]:
        key = f"{prefix}.{suf}"
        with safe_open(f"{MODEL_DIR}/{idx[key]}", framework="pt", device="cpu") as f:
            out[suf] = f.get_tensor(key).to(DEVICE)
    return out


pre = f"model.layers.{LAYER}.mlp.experts.{EXPERT}"
up = load(f"{pre}.up_proj")       # [768, 128] int32, scale/zp [768, 16]
gate = load(f"{pre}.gate_proj")
down = load(f"{pre}.down_proj")   # [2048, 48], scale/zp [2048, 6]

# --- Merge w13 as KsanaLLM does: [up, gate] (up first) ---
w13 = {k: torch.cat([up[k], gate[k]], dim=0).contiguous() for k in up}  # [1536, ...]
print("w13 merged:", {k: tuple(v.shape) for k, v in w13.items()})
print("w2(down):  ", {k: tuple(v.shape) for k, v in down.items()})

schema = HummingWeightSchema(
    b_dtype=B_DTYPE, weight_scale_group_size=GROUP_SIZE,
    has_zero_point=True, is_fp_zero_point=True,
)


def build_layer(shape_n, shape_k, tensors):
    layer = HummingLayer(shape_n=shape_n, shape_k=shape_k,
                         weight_config=schema, torch_dtype=DTYPE).to(DEVICE)
    layer.load_from_tensors({k: v.clone() for k, v in tensors.items()})
    layer.transform()
    return layer


def deq(tensors):
    return dequantize_weight(tensors["weight"], tensors["weight_scale"],
                             tensors["zero_point"], None, B_DTYPE, packed=True).to(DTYPE)


# --- bf16 dequantized reference weights ---
w13_deq = deq(w13)     # [1536, 2048]
w2_deq = deq(down)     # [2048, 768]
print("w13_deq:", tuple(w13_deq.shape), "w2_deq:", tuple(w2_deq.shape))

# --- humming layers (same kernel the KsanaLLM bridge uses) ---
w13_layer = build_layer(INTER * 2, HIDDEN, w13)
w2_layer = build_layer(HIDDEN, INTER, down)

torch.manual_seed(0)
T = 4
x = (torch.randn(T, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.05)


def rel(a, b):
    a, b = a.float(), b.float()
    md = (a - b).abs().max().item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return md, cos


# ---- 1) validate w13 GEMM (dequant/scale/zp/kernel) ----
gate_up_h = w13_layer(x)                 # [T, 1536] humming
gate_up_ref = x.float() @ w13_deq.float().T   # [T, 1536] reference, order [up | gate]
md, cos = rel(gate_up_h, gate_up_ref)
print(f"\n[w13 GEMM]   max|diff|={md:.4e}  cos={cos:.6f}   "
      f"({'OK' if cos > 0.99 else 'MISMATCH'})")

# ---- 2) silu convention ----
up_h, gate_h = gate_up_h[:, :INTER], gate_up_h[:, INTER:]   # [up | gate] per KsanaLLM merge
act_A = F.silu(up_h.float()) * gate_h.float()    # KsanaLLM InvokeSiluAndMul: silu(first)*second
act_B = F.silu(gate_h.float()) * up_h.float()    # silu(second)*first  (HF-correct if up=first)

# reference activation from dequant ref (HF: silu(gate)*up)
up_ref, gate_ref = gate_up_ref[:, :INTER], gate_up_ref[:, INTER:]
act_ref = F.silu(gate_ref) * up_ref

mdA, cosA = rel(act_A, act_ref)
mdB, cosB = rel(act_B, act_ref)
print(f"[silu A=silu(up)*gate ]  cos={cosA:.6f}  ({'MATCH' if cosA > 0.99 else 'no'})")
print(f"[silu B=silu(gate)*up ]  cos={cosB:.6f}  ({'MATCH' if cosB > 0.99 else 'no'})")

# ---- 3) full single-expert output ----
out_A = w2_layer(act_A.to(DTYPE))
out_B = w2_layer(act_B.to(DTYPE))
out_ref = act_ref.to(DTYPE).float() @ w2_deq.float().T
mdA2, cosA2 = rel(out_A, out_ref)
mdB2, cosB2 = rel(out_B, out_ref)
print(f"\n[expert out A] cos={cosA2:.6f}  ({'MATCH' if cosA2 > 0.99 else 'no'})")
print(f"[expert out B] cos={cosB2:.6f}  ({'MATCH' if cosB2 > 0.99 else 'no'})")

print("\n==== VERDICT ====")
if cos <= 0.99:
    print("w13 GEMM itself is WRONG -> bug in dequant/scale/zero_point/weight feeding "
          "(not silu). Check how KsanaLLM packs/binds scale & zero_point for the bridge.")
elif cosB2 > 0.99 and cosA2 <= 0.99:
    print("GEMM OK, but correct silu is B = silu(gate=second_half)*up(first_half). "
          "KsanaLLM's InvokeSiluAndMul does A=silu(first)*second -> SILU HALF IS SWAPPED.")
elif cosA2 > 0.99:
    print("GEMM OK and silu A (KsanaLLM's current) is correct -> bug is elsewhere "
          "(routing/sorted_ids/reduce/topk_weights), not GEMM or silu.")
else:
    print("Neither silu matches -> deeper issue; inspect gate_up values.")
