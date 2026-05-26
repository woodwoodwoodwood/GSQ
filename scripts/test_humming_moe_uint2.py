# Minimal reproducer for humming MoE uint2 ILLEGAL_ADDRESS on H20.
#
# Background:
#   - Dense uint2 HummingLayer.forward() works fine on H20.
#   - vLLM Qwen3-MoE + humming with --quantization humming dies with
#     CUDA_ERROR_ILLEGAL_ADDRESS during profile_run, in ops.humming_gemm
#     with both gemm_type="grouped" and gemm_type="indexed".
#
# This script bypasses vLLM entirely. It builds a synthetic 128-expert MoE
# down_proj layer (matching Qwen3-30B-A3B's experts.down_proj shape) with
# random uint2 weights, generates random MoE routing state via humming's own
# generate_random_moe_tensors, and calls HummingMethod.forward_layer with
# GROUPED_CONTIGUOUS / INDEXED gemm_type.
#
# Expected outcome:
#   - On H100: both succeed.
#   - On H20:  both raise CUDA_ERROR_ILLEGAL_ADDRESS at cuLaunchKernelEx.

import json
import os
import sys
import torch

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
from humming.schema.humming import HummingWeightSchema
from humming.utils.test import (
    generate_random_moe_tensors,
    generate_random_weight,
)

# Qwen3-30B-A3B down_proj shape: N=2048, K=768, group_size=128, b_dtype=uint2
# num_experts=128, num_experts_per_tok=8
N, K = 2048, 768
NUM_EXPERTS = 128
TOP_K = 8
GROUP_SIZE = 128
B_DTYPE = dtypes.DataType.from_str("uint2")
A_TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda"

print(f"device: {torch.cuda.get_device_name(0)}")
print(f"reproducing humming MoE uint2 on N={N} K={K} experts={NUM_EXPERTS} top_k={TOP_K}")

# 1) Build a HummingLayer with num_experts so it provisions per-expert packed weights.
schema = HummingWeightSchema(
    b_dtype=B_DTYPE,
    weight_scale_group_size=GROUP_SIZE,
    has_zero_point=True,
    is_fp_zero_point=True,
)
layer = HummingLayer(
    shape_n=N, shape_k=K,
    weight_config=schema,
    torch_dtype=A_TORCH_DTYPE,
    num_experts=NUM_EXPERTS,
).to(DEVICE)
print("layer constructed; param shapes:")
for name, p in layer.named_parameters():
    print(f"  {name}: {tuple(p.shape)} {p.dtype}")
for name, b in layer.named_buffers():
    print(f"  (buf) {name}: {tuple(b.shape)} {b.dtype}")

# 2) Generate random uint2 weights for all experts.
torch.manual_seed(0)
_, _, w_packed, w_scale, *_ = generate_random_weight(
    n=N, k=K,
    group_size=GROUP_SIZE,
    dtype=B_DTYPE,
    scale_dtype=dtypes.bfloat16,
    num_experts=NUM_EXPERTS,
    has_zero_point=True,
    is_fp_zero_point=True,
)
print(f"\nrandom weights: weight={tuple(w_packed.shape)} {w_packed.dtype} "
      f"scale={tuple(w_scale.shape)} {w_scale.dtype}")

# Manually fill the layer's params (the same fields humming would populate
# via load_from_tensors). zero_point will be created if has_zero_point=True.
zp = torch.zeros_like(w_scale)  # random fp zero point
tensors = {
    "weight": w_packed,
    "weight_scale": w_scale,
    "zero_point": zp,
}
layer.load_from_tensors({k: v.to(DEVICE) for k, v in tensors.items()})
layer.transform()
print("layer transform OK")


def try_gemm_type(gemm_type: GemmType, shape_m: int):
    """Try one MoE forward with the given gemm_type and shape_m tokens."""
    print(f"\n{'=' * 60}")
    print(f"TRY gemm_type={gemm_type.value} shape_m={shape_m}")
    print(f"{'=' * 60}")

    # Build random topk_ids + the MoE routing tensors humming kernels need.
    if gemm_type == GemmType.INDEXED:
        # INDEXED needs a block_size param; humming examples use 32 or 64.
        topk_ids, _, sorted_ids, expert_ids, num_tokens_padded = (
            generate_random_moe_tensors(
                shape_m=shape_m,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                gemm_type=gemm_type,
                block_size_config=32,
            )
        )
        expert_layout = None
    else:  # GROUPED_CONTIGUOUS
        topk_ids, expert_layout, sorted_ids, expert_ids, num_tokens_padded = (
            generate_random_moe_tensors(
                shape_m=shape_m,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                gemm_type=gemm_type,
            )
        )

    print(f"  topk_ids: {tuple(topk_ids.shape)} {topk_ids.dtype}")
    if expert_layout is not None:
        print(f"  expert_layout: {tuple(expert_layout.shape)} {expert_layout.dtype}")
    if sorted_ids is not None:
        print(f"  sorted_ids: {tuple(sorted_ids.shape)} {sorted_ids.dtype}")
        print(f"  expert_ids: {tuple(expert_ids.shape)} {expert_ids.dtype}")
        print(f"  num_tokens_padded: {num_tokens_padded.item()}")

    # Input activation matches what vLLM passes to humming MoE.
    # For GROUPED_CONTIGUOUS, vLLM permutes tokens so each expert's segment
    # is contiguous; inputs.shape[0] == sum(expert_layout diffs).
    if gemm_type == GemmType.GROUPED_CONTIGUOUS:
        total_tokens = shape_m * TOP_K  # after permute
    else:  # INDEXED
        total_tokens = num_tokens_padded.item()
    x = (torch.randn(total_tokens, K, dtype=A_TORCH_DTYPE, device=DEVICE) * 0.05)
    print(f"  input x: {tuple(x.shape)} {x.dtype}")

    compute_cfg = {
        "use_batch_invariant": False,
        "use_f16_accum": False,
        "gemm_type": gemm_type.value,
    }
    tuning_cfg = HummingMethod.get_default_tuning_configs(
        layer=layer,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=gemm_type,
    )
    print(f"  tuning_cfg entries: {len(tuning_cfg)}")

    try:
        out = HummingMethod.forward_layer(
            layer=layer,
            inputs=x,
            compute_config=compute_cfg,
            tuning_config=tuning_cfg,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            top_k=TOP_K if gemm_type == GemmType.INDEXED else 1,
            valid_shape_m=shape_m * TOP_K if gemm_type == GemmType.GROUPED_CONTIGUOUS else 0,
        )
        torch.cuda.synchronize()
        print(f"  OK: out {tuple(out.shape)} {out.dtype}")
        print(f"     first 5: {out.flatten()[:5].tolist()}")
        return True
    except RuntimeError as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


# 3) Run the two MoE paths. Use shape_m that matches vLLM profile_run
#    (typical batch=1 first, then a larger batch like 16).
results = {}
for shape_m in (1, 16):
    for gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.INDEXED):
        ok = try_gemm_type(gemm_type, shape_m)
        results[(gemm_type.value, shape_m)] = ok

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
for (gt, m), ok in results.items():
    print(f"  gemm_type={gt:24s} shape_m={m:4d}  {'OK' if ok else 'FAIL'}")
