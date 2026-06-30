# GSQ 支持 Qwen3.6-35B-A3B 量化训练与业务数据准备 — 需求说明书

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | GSQ 支持 Qwen3.6-35B-A3B（MoE）2-bit 量化训练与业务数据准备 需求说明书 |
| 所属项目 | GSQ（Gumbel-Softmax Quantization for LLMs） |
| 目标模型 | `Qwen/Qwen3.6-35B-A3B`（HuggingFace 公开模型） |
| 文档版本 | v2.2（含模型支持、业务数据准备、数据量修订、第一步验证方案） |
| 状态 | 代码已落盘；第一步（数据链路）验证待在容器内执行 |
| 涉及文件 | `main.py`、`configs/qwen36/qwen36_35B_A3B.yaml`、`src/data/dataset.py`、`scripts/convert_business_data_to_gsq.py`、`data/sample_biz_data.json`、`tests/verify_step1_data_pipeline.py` |

---

## 一、背景与目标

GSQ 是基于 Gumbel-Softmax 松弛的训练后量化（PTQ）框架，采用「GPTQ 初始化 + Gumbel-Softmax 精炼」两阶段、逐层量化，可在显存远小于模型规模时完成 2-bit 量化。

目标（两个衔接模块）：

1. **模型训练支持**：新增对 `Qwen3.6-35B-A3B`（MoE）的多卡 2-bit 量化训练支持，且不破坏既有模型族。
2. **业务数据准备**：支持使用业务自有数据（Alpaca 风格 instruction/input/output）作为校准语料。

约束：多卡（`world_size>1`）分布式专家并行；2-bit；运行环境为远程 Docker 容器。

---

## 二、关键调研结论（Qwen3.6 架构）

实测 `Qwen/Qwen3.6-35B-A3B` 的公开 `config.json`：

| 字段 | 取值 |
| --- | --- |
| `model_type`（顶层） | `qwen3_5_moe` |
| `architectures` | `Qwen3_5MoeForConditionalGeneration` |
| `num_experts` / `num_experts_per_tok` | 256 / 8（另含 1 共享专家） |
| `moe_intermediate_size` | 512 |
| `hidden_size` / `num_hidden_layers` | 2048 / 40 |
| 注意力 | 30 层 `linear_attention`（GatedDeltaNet）+ 10 层 `full_attention` |
| 专家权重 | fused 3D：`gate_up_proj[E,2I,H]` / `down_proj[E,H,I]` |
| 外壳 | 多模态：`text_config` + `vision_config` |

> 结论：Qwen3.6 与 Qwen3.5 **完全同构**（同为 `qwen3_5_moe`），GSQ 现有 `Qwen35MoeDistributedWrapper` 天然适配，**无需新增 wrapper**。

---

## 三、现状分析

- **统一入口**：`main.py` 的 `get_model_wrapper` 选择 wrapper。
- **wrapper 抽象**：继承 `src/models/base.py` 的 `BaseModelWrapper`。
- **完整实现**：`Qwen35MoeDistributedWrapper`（多卡）覆盖 fused 专家加载、专家并行、GPTQ 初始化、保存/恢复、分布式 PPL。
- **单卡缺陷**：`Qwen35MoeWrapper` 只识别 `nn.Linear`，而专家是 `nn.Parameter`，单卡无法量化 MoE 专家 → 必须多卡。
- **多模态外壳**：`base.py` 用 `text_config` 构建纯文本 CausalLM，检查点 `.language_model.*` 键经 `_ckpt_to_model_name` 桥接。
- **数据管线**：`src/data/dataset.py` 只读 `text` 字段，且丢弃 tokenize 后短于 `max_length` 的样本；改动前不支持本地 JSON/JSONL 与 Alpaca 格式。

---

## 四、解决方案

### 4.1 模型支持

- **新增** `configs/qwen36/qwen36_35B_A3B.yaml`：指向 `Qwen/Qwen3.6-35B-A3B`，2-bit、GPTQ、`groupsize=128`，超参与 `qwen35_35B_A3B.yaml` 对齐。
- **加固** `main.py` 路由：合并 Qwen3 家族分支，以 `config.model_type` 为权威依据（名称匹配兜底），避免本地改名误路由。

```python
elif 'qwen3' in name_lower:
    from transformers import AutoConfig as _AC
    _cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
    _tc = getattr(_cfg, 'text_config', _cfg)
    _mt = str(getattr(_tc, 'model_type', '') or getattr(_cfg, 'model_type', '')).lower()
    _is_moe = hasattr(_tc, 'num_experts') or hasattr(_tc, 'num_local_experts')
    _is_qwen35 = ('qwen3_5' in _mt
        or 'qwen3.5' in name_lower or 'qwen3_5' in name_lower
        or 'qwen3.6' in name_lower or 'qwen3_6' in name_lower)
    if _is_qwen35:
        if _is_moe and world_size > 1:
            from src.models.qwen35_moe_dist import Qwen35MoeDistributedWrapper
            return Qwen35MoeDistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        elif _is_moe:
            from src.models.qwen35_moe import Qwen35MoeWrapper
            return Qwen35MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        else:
            from src.models.qwen35 import Qwen35Wrapper
            return Qwen35Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    else:
        if _is_moe and world_size > 1:
            from src.models.qwen3_moe_dist import Qwen3MoeDistributedWrapper
            return Qwen3MoeDistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        elif _is_moe:
            from src.models.qwen3_moe import Qwen3MoeWrapper
            return Qwen3MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        else:
            from src.models.qwen3 import Qwen3Wrapper
            return Qwen3Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
```

### 4.2 业务数据准备

- **增强** `src/data/dataset.py`：新增本地 `.jsonl/.json` 加载分支（`load_dataset("json", data_files=..., split=split, streaming=True)`）。
- **新增** `scripts/convert_business_data_to_gsq.py`：Alpaca→GSQ 校准语料转换（chat template 渲染 + packing 到约 seqlen，输出 `{"text":...}` JSONL）。

---

## 五、变更清单

| 类型 | 文件 | 说明 |
| --- | --- | --- |
| 新增 | `configs/qwen36/qwen36_35B_A3B.yaml` | Qwen3.6 的 2-bit 多卡训练配置 |
| 修改 | `main.py` | 路由以 `model_type` 为权威依据 |
| 修改 | `src/data/dataset.py` | 新增本地 `.jsonl/.json` 加载分支 |
| 新增 | `scripts/convert_business_data_to_gsq.py` | Alpaca→GSQ 转换脚本 |
| 新增 | `data/sample_biz_data.json` | 第一步验证用样例数据（30 条） |
| 新增 | `tests/verify_step1_data_pipeline.py` | 第一步验证脚本（数据链路 + 可选路由） |

---

## 六、路由决策对照表

| 模型 / 路径示例 | `model_type` | `world_size` | 命中 wrapper |
| --- | --- | --- | --- |
| `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_moe` | > 1 | `Qwen35MoeDistributedWrapper` |
| `/data/models/q36`（改名） | `qwen3_5_moe` | > 1 | `Qwen35MoeDistributedWrapper`（model_type 兜底） |
| `Qwen/Qwen3-30B-A3B` | `qwen3_moe` | > 1 | `Qwen3MoeDistributedWrapper` |

---

## 七、数据准备指南

### 7.1 格式

- 输入（业务原始）：Alpaca JSON 数组/JSONL，字段 `instruction`/`input`/`output`。
- 目标（GSQ 消费）：JSONL，每行 `{"text": "<打包后 >= max_length 的文本>"}`。

### 7.2 数据量（重要）

> 校准数据量由 `num_samples` 与 `max_length` 决定，可调；并非固定要 1900 万 token。PTQ 校准本质需要的数据很少（经典 GPTQ 约 128 样本），关键是覆盖业务分布。

| 场景 | `max_length` | `num_samples` | `gptq.nsamples` | `val_samples` | 约需原始条数 |
| --- | --- | --- | --- | --- | --- |
| 最小可用 | 512 | 128 | 128 | 16 | 约 2–4 千条 |
| 推荐起步 | 1024 | 512 | 256 | 64 | 约 1.5–3 万条 |
| 较充分 | 2048 | 1024 | 256 | 64 | 约 4–8 万条 |
| 默认（一般无需） | 4096 | 4096 | 512 | 128 | 约 30 万条以上 |

（按业务短文本 50–80 token/条估算。）

---

## 八、第一步验证方案（数据链路冒烟测试）

### 8.1 目的与范围

在**不依赖 GPU、不加载 35B 权重**（仅需 tokenizer）的前提下，验证「Alpaca 数据 → 转换脚本 → GSQ 数据加载器」链路打通，这是整个流程最前置、最易独立验证的环节。

### 8.2 测试资产

- `data/sample_biz_data.json`：30 条餐饮客服中英样例（覆盖 `input` 空与非空）。
- `tests/verify_step1_data_pipeline.py`：用 `StreamingHFDataset` 加载转换产物，断言样本可加载且每条恰为 seqlen；可选 `--check-routing` 验证 Qwen3.6 路由判定。

### 8.3 执行步骤（容器内）

```bash
cd /usr/local/app/GSQ

# 1) 转换（冒烟用小 seqlen，确保少量数据也能产出样本）
python scripts/convert_business_data_to_gsq.py \
  --input ./data/sample_biz_data.json \
  --output ./data/sample_calib.jsonl \
  --model Qwen/Qwen3.6-35B-A3B \
  --seqlen 128 --margin 16 --format chat

# 2) 验证数据加载链路（+ 可选路由判定）
python tests/verify_step1_data_pipeline.py \
  --data ./data/sample_calib.jsonl \
  --model Qwen/Qwen3.6-35B-A3B \
  --seqlen 128 --num-samples 4 --check-routing
```

> `--model` 仅用于加载 tokenizer 与读取 config，不会下载 35B 权重。若容器无外网，请将 `--model` 换成本地 Qwen3.6 目录或任一可用的 Qwen tokenizer 路径。

### 8.4 判定标准

- 转换脚本结尾打印「输出样本数 > 0」；
- 验证脚本输出 `[PASS] 数据链路正常：样本可加载且每条恰为 128 tokens`；
- 启用 `--check-routing` 时输出 `model_type=qwen3_5_moe ... [PASS] 路由判定符合预期`；
- 末尾打印「结果：全部通过 ✅」，进程退出码 0。

### 8.5 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 转换输出 0 样本 | seqlen 过大 / 数据太少 | 调小 `--seqlen` 或增加样例数据 |
| 验证收集到 0 条 | seqlen 与转换不一致 | 保证两处 seqlen 一致 |
| tokenizer 加载失败 | 无外网/路径错误 | 改用本地 tokenizer 路径 |
| 路由判定非 qwen3_5_moe | transformers 版本过低 | 升级到 `>= 5.3` |

---

## 九、后续步骤（验证通过后）

1. 用真实业务数据按「七.2」选定档位执行转换；
2. 在 `configs/qwen36/qwen36_35B_A3B.yaml` 设 `data.dataset_name` 指向产物、`max_length` 与 seqlen 一致；
3. 多卡训练：`NPROC=4 CONFIG_FILE=configs/qwen36/qwen36_35B_A3B.yaml bash scripts/run.sh`；
4. 导出：`python save_model.py --config configs/qwen36/qwen36_35B_A3B.yaml`；
5. 部署评测：`CONFIG_FILE=configs/qwen36/qwen36_35B_A3B.yaml bash scripts/serve_model.sh`。

---

## 十、风险与注意事项

1. **transformers 版本** 必须 `>= 5.3`。
2. **务必多卡**：单卡无法量化 fused MoE 专家。
3. **vLLM TP 约束**：`moe_intermediate_size=512`、`groupsize=128` → 有效 TP 仅 {1,2,4}。
4. **数据长度门槛**：短于 `max_length` 的样本会被丢弃，需 packing 或调小 `max_length`。
5. **数据量勿盲目堆大**：按规模选档，校准重在分布覆盖。
6. **校准分布一致性**：建议 `--format chat`。
7. **多模态外壳**：仅量化文本部分。

---

## 十一、验收标准

- 第一步验证脚本输出全部通过；
- 用业务校准语料启动多卡训练，正确路由到 `Qwen35MoeDistributedWrapper` 并逐层产出分片；
- 可经 `save_model.py` 拼装并由 vLLM 加载评测；
- 既有模型族无回归。
