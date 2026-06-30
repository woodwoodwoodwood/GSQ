"""Qwen3.6-MoE 分布式（专家并行）适配 wrapper。

Qwen3.6（model_type: ``qwen3_5_moe``，architectures: ``Qwen3_5MoeForConditionalGeneration``）
与 Qwen3.5 共享完全相同的架构：256 路由专家 + 1 共享专家、混合注意力
（linear_attention / GatedDeltaNet + full_attention）、fused 3D 专家张量
（``gate_up_proj[E, 2I, H]`` / ``down_proj[E, H, I]``）、多模态外壳（text_config 嵌套）。

因此 Qwen3.6 的适配通过**继承** Qwen3.5 的分布式实现完成：后者已完整覆盖
fused 专家分片加载、专家并行 all-to-all 派发、GPTQ 初始化、量化权重保存/恢复、
分布式困惑度评测等。此处保留为独立子类，作为 Qwen3.6 的专属入口；若未来
Qwen3.6 的配置或模块布局与 Qwen3.5 出现差异，可在此覆写相应方法。
"""

from src.models.qwen35_moe_dist import Qwen35MoeDistributedWrapper


class Qwen36MoeDistributedWrapper(Qwen35MoeDistributedWrapper):
    """Qwen3.6 MoE 分布式专家并行 wrapper。

    当前与 ``Qwen35MoeDistributedWrapper`` 行为一致（同为 qwen3_5_moe 架构），
    作为显式子类提供 Qwen3.6 专属入口与未来覆写点。这是多卡 2-bit 量化的
    推荐路径。
    """

    pass
