# Closer reproducer for vLLM humming MoE crash on H20.
# Test both w13 (gate_up) and w2 (down) shapes with large batches matching
# vLLM profile_run (~4096 tokens * top_k=8 = ~32K tokens after permute).

import json
import torch
from safetensors import safe_open

from humming import dtypes
from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
from humming.schema.humming import HummingWeightSchema
from humming.utils.test import generate_random_moe_tensors

MODEL_DIR = "/data1/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming"
NUM_EXPERTS = 128
TOP_K = 8
GROUP_SIZE = 128
B_DTYPE = dtypes.DataType.from_str("uint2")
A_TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda"

# Qwen3-30B-A3B shapes
# w13 (gate_proj + up_proj fused): N = 2 * moe_intermediate_size = 2 * 768 = 1536
#                                  K = hidden_size = 2048
# w2  (down_proj):                  N = hidden_size = 2048
#                                  K = moe_intermediate_size = 768
W13_N, W13_K = 1536, 2048
W2_N, W2_K = 2048, 768

print(f"device: {torch.cuda.get_device_name(0)}")


def build_layer(N: int, K: int, prefix_for_real_weight: str):
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

    # Try to load real weights; fall back to random packed int32 if shape mismatch.
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]
    single = {}
    try:
        for suf in ["weight", "weight_scale", "zero_point"]:
            key = f"{prefix_for_real_weight}.{suf}"
            with safe_open(f"{MODEL_DIR}/{idx[key]}", framework="pt", device="cpu") as f:
                t = f.get_tensor(key).to(DEVICE)
            single[suf] = t
        if single["weight"].shape != (N, K * 2 // 32):  # K * num_bits / 32
            raise RuntimeError(
                f"shape mismatch: got {tuple(single['weight'].shape)} "
                f"expected (N={N}, K*2/32={K*2//32}); falling back to random."
            )
    except Exception as e:
        print(f"  using random packed weights for N={N} K={K}: {e}")
        single = {
            "weight": torch.randint(
                -(2**31), 2**31 - 1, (N, K * 2 // 32), dtype=torch.int32, device=DEVICE
            ),
            "weight_scale": torch.randn(
                (N, K // GROUP_SIZE), dtype=A_TORCH_DTYPE, device=DEVICE
            ) * 0.01,
            "zero_point": torch.zeros(
                (N, K // GROUP_SIZE), dtype=A_TORCH_DTYPE, device=DEVICE
            ),
        }

    tensors = {
        k: v.unsqueeze(0).expand(NUM_EXPERTS, *v.shape).contiguous()
        for k, v in single.items()
    }
    layer.load_from_tensors(tensors)
    layer.transform()
    return layer


def try_gemm_type(layer: HummingLayer, name: str, gemm_type: GemmType,
                  shape_m: int, K: int, top_k_for_indexed: int = TOP_K):
    tag = f"{name} {gemm_type.value} shape_m={shape_m}"
    print(f"\n--- TRY {tag} ---")

    if gemm_type == GemmType.INDEXED:
        topk_ids, _, sorted_ids, expert_ids, num_tokens_padded = (
            generate_random_moe_tensors(
                shape_m=shape_m, num_experts=NUM_EXPERTS, top_k=TOP_K,
                gemm_type=gemm_type, block_size_config=32,
            )
        )
        expert_layout = None
        total_tokens = num_tokens_padded.item()
        used_top_k = top_k_for_indexed
        used_valid_shape_m = 0
    else:
        topk_ids, expert_layout, sorted_ids, expert_ids, num_tokens_padded = (
            generate_random_moe_tensors(
                shape_m=shape_m, num_experts=NUM_EXPERTS, top_k=TOP_K,
                gemm_type=gemm_type,
            )
        )
        total_tokens = shape_m * TOP_K  # post-permute
        used_top_k = 1
        used_valid_shape_m = shape_m * TOP_K

    x = (torch.randn(total_tokens, K, dtype=A_TORCH_DTYPE, device=DEVICE) * 0.05)
    print(f"  input x: {tuple(x.shape)}  total_tokens={total_tokens}")

    compute_cfg = {
        "use_batch_invariant": False, "use_f16_accum": False,
        "gemm_type": gemm_type.value,
    }
    tuning_cfg = HummingMethod.get_default_tuning_configs(
        layer=layer, use_f16_accum=False, use_batch_invariant=False,
        gemm_type=gemm_type,
    )

    try:
        out = HummingMethod.forward_layer(
            layer=layer, inputs=x,
            compute_config=compute_cfg, tuning_config=tuning_cfg,
            sorted_ids=sorted_ids, expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded, expert_layout=expert_layout,
            top_k=used_top_k, valid_shape_m=used_valid_shape_m,
        )
        torch.cuda.synchronize()
        print(f"  OK: out {tuple(out.shape)}")
        return "OK"
    except RuntimeError as e:
        msg = str(e)[:300]
        print(f"  FAILED: {msg}")
        return f"FAIL: {msg[:100]}"


# Build both layers.
print("\n=== Building w13 layer (1536 x 2048) ===")
layer_w13 = build_layer(W13_N, W13_K, "model.layers.0.mlp.experts.0.up_proj")
print("\n=== Building w2 layer (2048 x 768) ===")
layer_w2 = build_layer(W2_N, W2_K, "model.layers.0.mlp.experts.0.down_proj")

# vLLM profile_run typically uses max_model_len tokens. Try escalating batches.
shape_ms = [1, 16, 256, 1024, 4096]

results = {}
for shape_m in shape_ms:
    for gt in (GemmType.GROUPED_CONTIGUOUS, GemmType.INDEXED):
        results[("w13", gt.value, shape_m)] = try_gemm_type(
            layer_w13, "w13", gt, shape_m, W13_K
        )
        results[("w2", gt.value, shape_m)] = try_gemm_type(
            layer_w2, "w2", gt, shape_m, W2_K
        )

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
for (sub, gt, m), r in results.items():
    print(f"  {sub:4s} {gt:24s} shape_m={m:5d}  {r}")
