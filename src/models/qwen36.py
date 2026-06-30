"""Qwen3.6 稠密（dense）适配 wrapper。

为完整性提供 Qwen3.6 的稠密变体入口，继承 Qwen3.5 的稠密实现。Qwen3.6 的旗舰
权重为 MoE（35B-A3B）；若出现稠密变体，此 wrapper 直接复用 Qwen3.5 的逻辑。
"""

from src.models.qwen35 import Qwen35Wrapper


class Qwen36Wrapper(Qwen35Wrapper):
    """Qwen3.6 稠密 wrapper（与 Qwen35Wrapper 行为一致）。"""

    pass
