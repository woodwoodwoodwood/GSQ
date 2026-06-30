"""Qwen3.6-MoE 单卡适配 wrapper。

Qwen3.6 与 Qwen3.5 架构同构（model_type: ``qwen3_5_moe``），单卡适配通过继承
``Qwen35MoeWrapper`` 完成。

注意：单卡 ``Qwen35MoeWrapper`` 存在已知限制——它继承的基类逻辑只识别
``nn.Linear``，而 MoE 专家权重以 fused ``nn.Parameter``（``gate_up_proj`` /
``down_proj``）形式存在，因此**单卡模式无法量化 MoE 专家**。Qwen3.6 同样继承
此限制。多卡 2-bit 量化请使用 ``Qwen36MoeDistributedWrapper``。
"""

from src.models.qwen35_moe import Qwen35MoeWrapper


class Qwen36MoeWrapper(Qwen35MoeWrapper):
    """Qwen3.6 MoE 单卡 wrapper（与 Qwen35MoeWrapper 行为一致）。

    仅用于非专家量化场景的调试；正式 2-bit 量化请走分布式版本
    ``Qwen36MoeDistributedWrapper``。
    """

    pass
