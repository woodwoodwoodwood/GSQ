# vLLM 0.21.0 编译与部署指南

> 适用于 **Driver 550 / H20 GPU (sm_90)** 的 Docker 容器。
> 已验证：cu129 容器编译 + DeepSeek-V4-Flash 推理成功。

---

## 核心问题

| 问题 | 根因 | 表现 |
|---|---|---|
| PTX ISA 不兼容 | nvcc 12.8+ 生成 PTX ISA 8.5，Driver 550 只支持 ≤8.4 | `cudaErrorUnsupportedPtxVersion` |
| 容器 CUDA > 驱动 CUDA | 容器内 cu128/cu129 运行时，驱动只提供 12.4 API | PTX JIT 编译失败，但 SASS 可正常运行 |
| setuptools license 校验 | setuptools ≥77.0.3 拒绝 `license = "Apache-2.0"` 格式 | 编译失败 |
| 编译 OOM | MAX_JOBS 过高导致并发 nvcc 占用内存过大 | segfault (exit 139) |
| torch 被替换为非 CUDA 版本 | 编译过程会重装 torch | `NVIDIA driver too old (found version 12040)` |

---

## 解决方案总览

**核心思路**：nvcc wrapper 将 `-gencode arch=compute_XX,code=sm_XX` 替换为 `-arch=sm_90`，直接生成 SASS 机器码（不含嵌入式 PTX），绕过驱动的 PTX JIT 版本检查。

```
编译前：nvcc → nvcc wrapper（添加 -arch=sm_90）→ nvcc.bak（真实 nvcc）
运行时：nvcc 恢复为真实 nvcc（DeepGEMM JIT 需要）
```

---

## 步骤一：环境准备

```bash
# 确认环境
nvcc --version          # 应为 CUDA 12.8 或 12.9
nvidia-smi | head -5    # Driver 550.144.03
python --version        # 3.12
```

---

## 步骤二：克隆 vLLM 源码（带 humming MoE fix）

```bash
cd /usr/local/app
git clone -b fix/humming-moe-v0.21.0 https://github.com/woodwoodwoodwood/vllm.git vllm-0.21.0-fresh
cd vllm-0.21.0-fresh
git log --oneline -1
```

---

## 步骤三：创建 GSQ venv

```bash
cd /usr/local/app/GSQ

# 安装 uv
pip install uv

# 创建 venv 并安装依赖
UV_INDEX_PYTORCH=https://download.pytorch.org/whl/cu129 uv sync

# 安装 setuptools_scm（编译依赖）
uv pip install setuptools_scm \
  --python /usr/local/app/GSQ/.venv/bin/python \
  -i https://mirrors.cloud.tencent.com/pypi/simple/
```

---

## 步骤四：修复 pyproject.toml + 安装 torch cu129

```bash
cd /usr/local/app/vllm-0.21.0-fresh

# 删除 license 行（兼容 setuptools 70.x）
sed -i '/^license/d' pyproject.toml

# 安装 torch cu129 到 venv
uv pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu129 --reinstall \
  --python /usr/local/app/GSQ/.venv/bin/python
```

---

## 步骤五：nvcc wrapper + 编译

```bash
# 1. 备份真实 nvcc
cp /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-12.9/bin/nvcc.bak

# 2. 创建 wrapper（强制 -arch=sm_90 生成纯 SASS）
cat > /usr/local/cuda-12.9/bin/nvcc << 'EOF'
#!/bin/bash
exec /usr/local/cuda-12.9/bin/nvcc.bak -arch=sm_90 "$@"
EOF
chmod +x /usr/local/cuda-12.9/bin/nvcc

# 3. 验证 wrapper
nvcc --version

# 4. 编译（MAX_JOBS=64，不要超过！）
CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=64 \
uv pip install /usr/local/app/vllm-0.21.0-fresh \
  --no-build-isolation --reinstall \
  --python /usr/local/app/GSQ/.venv/bin/python \
  --index-url https://mirrors.cloud.tencent.com/pypi/simple/

# 5. 恢复真实 nvcc（运行时 DeepGEMM JIT 需要）
cp /usr/local/cuda-12.9/bin/nvcc.bak /usr/local/cuda-12.9/bin/nvcc
```

> **注意**：如果容器 CUDA 版本不同（如 cu128 而非 cu129），把路径 `/usr/local/cuda-12.9` 换成 `/usr/local/cuda-12.8`。

---

## 步骤六：编译后处理

```bash
# 1. 重新安装 torch cu129（编译过程会替换）
uv pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu129 --reinstall \
  --python /usr/local/app/GSQ/.venv/bin/python

# 2. 清除 DeepGEMM 缓存
rm -rf /tmp/.deepgemm_cache /root/.cache/deepgemm ~/.deepgemm_cache 2>/dev/null

# 3. 验证组件
cd /tmp  # 不要在 vllm 源码目录里 import！
/usr/local/app/GSQ/.venv/bin/python -c "
import vllm; print('vllm:', vllm.__version__)
import torch; print('CUDA:', torch.version.cuda, '| OK:', torch.cuda.is_available())
import vllm._flashmla_C; print('FlashMLA OK')
from vllm.model_executor.layers.quantization.humming import HummingMoEMethod; print('Humming OK')
"
```

---

## 步骤七：启动 DeepSeek-V4-Flash 服务

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
/usr/local/app/GSQ/.venv/bin/vllm serve \
  /data1/models/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enforce-eager
```

# 启动 DeepSeek-V4-Flash 2bit humming（TP=4）
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
/usr/local/app/GSQ/.venv/bin/vllm serve \
  /data1/models/deepseek-v4-flash-2bit-humming \
  --trust-remote-code \
  --quantization humming \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8900 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enforce-eager
```

> **关键**：`--kv-cache-dtype fp8` 是必需的，DeepSeek V4 MLA 注意力只支持 fp8 KV cache。

---

## 步骤八：启动 Qwen3-MoE humming 服务

```bash
CUDA_VISIBLE_DEVICES=7 FLASHINFER_DISABLE_VERSION_CHECK=1 \
/usr/local/app/GSQ/.venv/bin/vllm serve \
  /data1/models/Qwen3-MoE-2bit-humming \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8900 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --enforce-eager
```

---

## 验证推理

```bash
env -u http_proxy -u https_proxy curl -s http://127.0.0.1:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data1/models/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 128
  }' | python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

---

## 快速参考

| 组件 | 状态 | 说明 |
|---|---|---|
| Driver | 550.144.03 | 仅支持 CUDA 12.4 API 和 PTX ISA ≤8.4 |
| 容器 CUDA | 12.9 (cu129) | 编译用 nvcc wrapper 解决 SASS/PTX |
| vLLM | 0.21.0 (源码编译) | fix/humming-moe-v0.21.0 分支 |
| torch | 2.11.0+cu129 | 每次编译后必须重装 |
| FlashMLA | OK | CUDA 12.9 编译 |
| DeepGEMM | OK | UE8M0 (FP4) 模式 |
| Humming MoE | OK | 3 文件 Python patch |
| DeepSeek-V4-Flash | ✅ 推理成功 | TP=4, 4×H20 |
| Qwen3-MoE-2bit-humming | ✅ 推理成功 | 单卡 H20 |

---

## 常见错误速查

| 错误 | 解决 |
|---|---|
| `cudaErrorUnsupportedPtxVersion` | nvcc wrapper 未生效，检查 wrapper 是否正确创建 |
| `AssertionError: DeepseekV4 only supports fp8 kv-cache` | 添加 `--kv-cache-dtype fp8` |
| `ModuleNotFoundError: vllm._flashmla_C` | 用 CUDA 12.8+ 重新编译 |
| `NVIDIA driver too old (12040)` | torch 被替换为非 CUDA 版本，重装 `torch==2.11.0+cu129` |
| 编译 segfault (exit 139) | `MAX_JOBS` 太高，改为 64 或 32 |
| `ModuleNotFoundError: setuptools_scm` | `uv pip install setuptools_scm` |
| setuptools license 校验失败 | `sed -i '/^license/d' pyproject.toml` |
| 127.0.0.1:8900 连接被拒或代理拦截 | `env -u http_proxy -u https_proxy curl ...` |

---

## 关键路径

| 项目 | 路径 |
|---|---|
| venv Python | `/usr/local/app/GSQ/.venv/bin/python` |
| vllm 源码 | `/usr/local/app/vllm-0.21.0-fresh` |
| GSQ 项目 | `/usr/local/app/GSQ` |
| 真实 nvcc | `/usr/local/cuda-12.9/bin/nvcc.bak` |
| DeepSeek 模型 | `/data1/models/DeepSeek-V4-Flash` |
| Qwen humming 模型 | `/data1/models/Qwen3-MoE-2bit-humming` |