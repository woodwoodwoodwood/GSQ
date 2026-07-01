# Qwen3.6-35B-A3B GSQ 2bit 量化训练适配

## 背景结论

实测拉取 `Qwen/Qwen3.6-35B-A3B` 的 `config.json`,确认与 Qwen3.5 **完全同构**:
- `model_type = qwen3_5_moe`(与 Qwen3.5 相同)
- `architectures = Qwen3_5MoeForConditionalGeneration`
- 256 专家 / 8 激活,hidden_size=2048,40 层
- 30 层 linear_attention(GatedDeltaNet)+ 10 层 full_attention
- 多模态外壳:text_config + vision_config

因此 GSQ 现有 `Qwen35MoeDistributedWrapper` / `Qwen35MoeWrapper` 直接复用,**无需新增 wrapper、无需改 main.py 路由、无需改 save_model.py**(均按 model_type 动态判断,已覆盖 qwen3_5_moe)。

`main.py:487` 路由 `elif 'qwen3.5' in name_lower or 'qwen3.6' in name_lower` 已能命中标准模型名 `Qwen/Qwen3.6-35B-A3B`,无需加固。

## 改动范围

**仅新增 1 个文件,不改任何现有代码。**

### 新增 `configs/qwen36/qwen36_35B_A3B.yaml`

从 `configs/qwen35/qwen35_35B_A3B.yaml` 复制,改动:
- `model.name`: `Qwen/Qwen3.5-35B-A3B` → `Qwen/Qwen3.6-35B-A3B`
- `training.checkpoint_dir`: `checkpoints/qwen35-35b` → `checkpoints/qwen36-35b`
- `training.log_dir`: `logs/qwen35-35b` → `logs/qwen36-35b`
- 头部注释更新为 Qwen3.6 架构说明

其余超参完全对齐 qwen35(open_thoughts 数据集、2bit、groupsize=128、10 epochs、num_samples=4096、max_length=4096、nsamples=512),因为架构同构、规模相同(35B-A3B)。

## 为什么这样足够

1. **模型加载**:`main.py:get_model_wrapper` 命中 `qwen3.6` 分支 → 读 AutoConfig → `num_experts` 存在且 world_size>1 → `Qwen35MoeDistributedWrapper`,与 qwen35 走同一路径。
2. **数据**:用内置 `open_thoughts`,无需任何数据层改动。
3. **量化训练**:`Qwen35MoeDistributedWrapper` 对 `qwen3_5_moe` 的 fused 专家、layer_types、shared_expert 处理完全通用,不依赖模型名。
4. **导出**:`save_model.py:_build_ignore_list` 用 `model_type` 动态构建 ignore 列表,`"qwen3_5" in model_type` 命中 shared_expert,已覆盖 Qwen3.6。
5. **部署评测**:`save_model.py` 注入 `pack-quantized` quantization_config,vLLM/humming 可加载。

## 使用方式(适配完成后)

```bash
# 多卡训练(必须 world_size>1,单卡无法量化 fused MoE 专家)
NPROC=4 CONFIG_FILE=configs/qwen36/qwen36_35B_A3B.yaml bash scripts/run.sh

# 导出
python save_model.py --config configs/qwen36/qwen36_35B_A3B.yaml
```

## 验证

适配后可通过(不加载权重的)路由检查确认 Qwen3.6 被正确识别:
```bash
python tests/verify_step1_data_pipeline.py \
  --data <任意 jsonl> --model Qwen/Qwen3.6-35B-A3B \
  --seqlen 128 --num-samples 1 --check-routing
# 期望输出: model_type=qwen3_5_moe ... [PASS] 路由判定符合预期
```

## 风险提示

1. **transformers >= 5.3.0**(Qwen3.5/3.6 MoE 支持)。
2. **必须多卡**:单卡无法量化 fused MoE 专家(`Qwen35MoeWrapper` 单卡只识别 nn.Linear)。
3. **模型需先下载**:本地 `/data1/models/Qwen3.6-35B-A3B` 当前不存在,训练前需完成 HF 下载(70GB+)。config 里 `model.name` 用 HF 名 `Qwen/Qwen3.6-35B-A3B`,若有本地路径可改为本地目录。
4. **vLLM TP 约束**:`moe_intermediate_size=512`、`groupsize=128` → 有效 TP 仅 {1,2,4}(导出评测时注意)。
