# Minimal reproducer for humming MoE uint2 ILLEGAL_ADDRESS on H20.
# This script bypasses vLLM and uses REAL packed weights from your checkpoint.

import json
import torch
from safetensors import safe_open

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
from humming.schema.humming import HummingWeightSchema
from humming.utils.test import generate_random_moe_tensors

MODEL_DIR = "/usr/local/app/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming"
# down_proj shape:  N=2048, K=768, group_size=128, b_dtype=uint2
N, K = 2048, 768
NUM_EXPERTS = 128
TOP_K = 8
GROUP_SIZE = 128
B_DTYPE = dtypes.DataType.from_str("uint2")
A_TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda"

print(f"device: {torch.cuda.get_device_name(0)}")

# 1) Build HummingLayer with num_experts=128 (MoE).
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
print("expected MoE param shapes:")
for name, p in layer.named_parameters():
    print(f"  {name}: {tuple(p.shape)} {p.dtype}")

# 2) Load a real expert's packed weights from your checkpoint, then broadcast
#    to all 128 experts.
idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]
prefix = "model.layers.0.mlp.experts.0.down_proj"
single = {}
for suf in ["weight", "weight_scale", "zero_point"]:
    key = f"{prefix}.{suf}"
    with safe_open(f"{MODEL_DIR}/{idx[key]}", framework="pt", device="cpu") as f:
        single[suf] = f.get_tensor(key).to(DEVICE)
    print(f"  single expert {suf}: {tuple(single[suf].shape)} {single[suf].dtype}")

# Broadcast to 128 experts (same weights, same scale/zp -- doesn't matter
# for reproducing a launch crash).
tensors_moe = {
    "weight":       single["weight"].unsqueeze(0).expand(NUM_EXPERTS, -1, -1).contiguous(),
    "weight_scale": single["weight_scale"].unsqueeze(0).expand(NUM_EXPERTS, -1, -1).contiguous(),
    "zero_point":   single["zero_point"].unsqueeze(0).expand(NUM_EXPERTS, -1, -1).contiguous(),
}
for k, v in tensors_moe.items():
    print(f"  MoE {k}: {tuple(v.shape)} {v.dtype}")

layer.load_from_tensors(tensors_moe)
layer.transform()
print("layer transform OK\n")


def try_gemm_type(gemm_type: GemmType, shape_m: int):
    print(f"{'=' * 60}")
    print(f"TRY gemm_type={gemm_type.value} shape_m={shape_m}")
    print(f"{'=' * 60}")

    if gemm_type == GemmType.INDEXED:
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

    print(f"  topk_ids: {tuple(topk_ids.shape)}")
    if expert_layout is not None:
        print(f"  expert_layout: {tuple(expert_layout.shape)} {expert_layout.dtype}")
        print(f"  expert_layout sample: {expert_layout[:8].tolist()} ... {expert_layout[-3:].tolist()}")
    if sorted_ids is not None:
        print(f"  sorted_ids: {tuple(sorted_ids.shape)}  num_tokens_padded={num_tokens_padded.item()}")

    if gemm_type == GemmType.GROUPED_CONTIGUOUS:
        total_tokens = shape_m * TOP_K
    else:
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
        print(f"  OK: out {tuple(out.shape)}  first 5: {out.flatten()[:5].tolist()}")
        return "OK"
    except RuntimeError as e:
        msg = str(e)[:200]
        print(f"  FAILED: {type(e).__name__}: {msg}")
        return f"FAIL ({msg[:80]})"


results = {}
for shape_m in (1, 16):
    for gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.INDEXED):
        results[(gemm_type.value, shape_m)] = try_gemm_type(gemm_type, shape_m)
        print()

print(f"{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
for (gt, m), r in results.items():
    print(f"  gemm_type={gt:24s} shape_m={m:4d}  {r}")
