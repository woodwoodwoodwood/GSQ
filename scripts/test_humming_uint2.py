# 不走 vLLM，直接调 humming 跑你模型里的一个真实 layer
import torch
import json
from safetensors import safe_open
from humming.layer import HummingLayer

md = "/usr/local/app/models/Qwen3-30B-A3B-Instruct-2507-gsq-2bit-humming"
# 取一个 expert 的 down_proj
# down_proj: N=2048, K=768, group_size=128, b_dtype=uint2

layer = HummingLayer(
    shape_n=2048,
    shape_k=768,
    weight_config={"dtype": "uint2", "group_size": 128, "has_zero_point": True, "is_fp_zero_point": True},
    torch_dtype=torch.bfloat16,
).cuda()

# 直接加载 packed weight
idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
prefix = "model.layers.0.mlp.experts.0.down_proj"
sd = {}
for suf in ["weight", "weight_scale", "zero_point"]:
    k = f"{prefix}.{suf}"
    with safe_open(f"{md}/{idx[k]}", framework="pt", device="cuda:0") as f:
        sd[suf] = f.get_tensor(k)

# 把权重塞进 layer
layer.weight.data = sd["weight"].cuda()
layer.weight_scale.data = sd["weight_scale"].cuda()
layer.zero_point.data = sd["zero_point"].cuda()

# 跑前向
x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda")
y = layer(x)
print("OK:", y.shape, y.dtype)
print(y[0, :5])
