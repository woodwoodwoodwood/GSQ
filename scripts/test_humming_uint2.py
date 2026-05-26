# Minimal reproducer: load one real uint2 layer from your humming checkpoint
# and run its forward, bypassing vLLM entirely.
# This isolates whether the cuLaunchKernelEx ILLEGAL_ADDRESS is a humming bug
# (kernel-side) or a vLLM-integration bug.
import json
import torch
from safetensors import safe_open
from humming import dtypes
from humming.layer import HummingLayer
from humming.schema.humming import HummingWeightSchema

MODEL_DIR = "/usr/local/app/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming"
# layer 0, expert 0, down_proj: N=2048, K=768, group_size=128, b_dtype=uint2
PREFIX = "model.layers.0.mlp.experts.0.down_proj"
N, K = 2048, 768
GROUP_SIZE = 128
B_DTYPE_STR = "uint2"
DTYPE = torch.bfloat16
DEVICE = "cuda"

print(f"device: {torch.cuda.get_device_name(0)}")
print(f"humming layer: {PREFIX}  shape_n={N} shape_k={K} b_dtype={B_DTYPE_STR}")

# 1) Read packed tensors from the safetensors shard.
idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]
tensors = {}
for suf in ["weight", "weight_scale", "zero_point"]:
    key = f"{PREFIX}.{suf}"
    shard = f"{MODEL_DIR}/{idx[key]}"
    with safe_open(shard, framework="pt", device="cpu") as f:
        tensors[suf] = f.get_tensor(key).to(DEVICE)
    print(f"  {suf}: dtype={tensors[suf].dtype} shape={tuple(tensors[suf].shape)}")

# 2) Build HummingLayer the same way convert_to_humming.py verify_one does.
schema = HummingWeightSchema(
    b_dtype=dtypes.DataType.from_str(B_DTYPE_STR),
    weight_scale_group_size=GROUP_SIZE,
    has_zero_point=True,
    is_fp_zero_point=True,
)
print(f"\nschema: {schema}")

layer = HummingLayer(
    shape_n=N, shape_k=K,
    weight_config=schema, torch_dtype=DTYPE,
).to(DEVICE)
layer.load_from_tensors(tensors)
layer.transform()
print("layer transform OK")

# 3) Forward (this is where vLLM was crashing).
torch.manual_seed(0)
x = (torch.randn(16, K, dtype=DTYPE, device=DEVICE) * 0.05)
print(f"\ninput: {tuple(x.shape)} {x.dtype}")

y = layer(x)
torch.cuda.synchronize()
print(f"output: {tuple(y.shape)} {y.dtype}")
print(f"  first 5: {y[0, :5].tolist()}")
print("\nOK — humming uint2 forward succeeded on this device.")
