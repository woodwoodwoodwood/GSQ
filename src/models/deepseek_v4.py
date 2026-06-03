"""DeepSeek-V4-Flash GSQ wrapper (single-process / non-distributed).

This wrapper performs **online dequantization** while loading the safetensors
checkpoint, so no offline reconstruction is required.

Two quantization formats coexist in the checkpoint:

* **Dense layers** (attention, shared_experts): FP8 E4M3 weights with UE8M0
  per-block scales.  Key convention::

      layers.{L}.self_attn.{q,k,v,o}_proj.weight  (float8_e4m3fn)
      layers.{L}.self_attn.{q,k,v,o}_proj.scale   (float8_e8m0fnu / uint8)
      layers.{L}.ffn.shared_experts.gate_proj.weight / .scale  (same)

* **MoE experts**: MXFP4 packed E2M1 with UE8M0 scales.  Key convention::

      layers.{L}.ffn.experts.{E}.w1.weight  (uint8, packed E2M1)
      layers.{L}.ffn.experts.{E}.w1.scale   (uint8, UE8M0 block scale)

Mapping: ``w1 -> gate_proj``, ``w3 -> up_proj``, ``w2 -> down_proj``.

Checkpoint keys use ``layers.X`` prefix; the HF model uses
``model.layers.X``.  ``_ckpt_to_model_name`` handles the mapping.

Pipeline:

1. Strip ``quantization_config`` + ``expert_dtype`` from the HF config so
   ``from_config`` builds a vanilla BF16 model skeleton.
2. Replace each MoE layer's ``ffn.experts`` submodule with a barebones
   BF16 fused container.
3. Set ``fp8_dense=True`` + ``mxfp4_experts=True`` so ``_set_tensors``
   dequantizes both dense FP8 and expert MXFP4 weights on-the-fly.
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn

from .base import BaseModelWrapper
from .qwen3_moe import Qwen3MoeWrapper


# Prefix mappings from checkpoint naming to HF model naming.
# Derived from vLLM's ``_make_deepseek_v4_weights_mapper``.
_CKPT_PREFIX_MAP = [
    ("layers.", "model.layers."),
    ("embed.", "model.embed."),
    ("norm.", "model.norm."),
    ("hc_head", "model.hc_head"),
    ("mtp.", "model.mtp."),
]

_CKPT_SUFFIX_MAP = [
    ("head.weight", "lm_head.weight"),
    ("embed.weight", "embed_tokens.weight"),
    (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
]


class _FusedBF16Experts(nn.Module):
    """Tiny container exposing Qwen3-MoE-style fused expert tensors in BF16."""

    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int, dtype):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        # Stay on meta until weights are materialized by ``_set_tensors``.
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size,
                        dtype=dtype, device="meta"),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size,
                        dtype=dtype, device="meta"),
            requires_grad=False,
        )

    def gate_proj_view(self, expert_idx: int) -> torch.Tensor:
        return self.gate_up_proj.data[expert_idx, : self.intermediate_size, :]

    def up_proj_view(self, expert_idx: int) -> torch.Tensor:
        return self.gate_up_proj.data[
            expert_idx, self.intermediate_size : 2 * self.intermediate_size, :
        ]

    def down_proj_view(self, expert_idx: int) -> torch.Tensor:
        return self.down_proj.data[expert_idx, :, :]

    @torch.no_grad()
    def forward_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU expert MLP: ``silu(x @ gate.T) * (x @ up.T) @ down.T``."""
        gate = torch.matmul(x, self.gate_proj_view(expert_idx).T)
        up = torch.matmul(x, self.up_proj_view(expert_idx).T)
        h = torch.nn.functional.silu(gate) * up
        return torch.matmul(h, self.down_proj_view(expert_idx).T)


class DeepseekV4Wrapper(Qwen3MoeWrapper):
    """GSQ wrapper for DeepSeek-V4-Flash with online dequantization."""

    # Runtime-resolved MoE block attribute ("ffn" or "mlp").
    _MOE_BLOCK_ATTR = "ffn"

    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        # Bypass ``Qwen3MoeWrapper.__init__`` (Qwen-specific config fields);
        # we still inherit its MoE helpers (``_virtual_expert_linear`` etc.).
        BaseModelWrapper.__init__(
            self, model_name, tokenizer, batch_size, seqlen, device, dtype,
            strip_quantization_config=True,
        )

        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)

        self.num_experts = (
            getattr(text_cfg, "n_routed_experts", None)
            or getattr(text_cfg, "num_experts", None)
            or getattr(text_cfg, "num_local_experts", 0)
        )
        self.first_k_dense_replace = getattr(text_cfg, "first_k_dense_replace", 0)
        self.decoder_sparse_step = getattr(text_cfg, "decoder_sparse_step", 1)
        self.mlp_only_layers = list(getattr(text_cfg, "mlp_only_layers", []))

        # Detect the actual layers module (DeepSeek-V4-Flash places ``layers``
        # at the top level, matching ckpt keys ``layers.X...``).
        if hasattr(self.model, "layers") and hasattr(self.model.layers, "__len__"):
            self._layers_module = self.model.layers
            self.layer_prefix = "layers"
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers_module = self.model.model.layers
            self.layer_prefix = "model.layers"
        else:
            raise RuntimeError(
                "DeepseekV4Wrapper: could not locate a ``layers`` ModuleList on the empty model."
            )
        self.num_layers = len(self._layers_module)

        # HF实现中MoE块名可能是`ffn`或`mlp`，需在运行时探测。
        self._MOE_BLOCK_ATTR = self._detect_moe_block_attr()

        self.is_moe = True
        self.fused_experts = True
        self.fused_expert_intermediate_size = (
            getattr(text_cfg, "moe_intermediate_size", None)
            or getattr(text_cfg, "intermediate_size", None)
        )

        # Activate online dequant in BaseModelWrapper._set_tensors.
        self.mxfp4_experts = True
        self.fp8_dense = True

        # Replace per-layer experts modules with BF16 fused containers so
        # ``_write_fused_expert_slice`` has somewhere to write the
        # dequantized data.
        self._install_bf16_fused_experts()
        self._patch_moe_forward_if_needed()

    def _detect_moe_block_attr(self):
        for layer in self._layers_module:
            # 先看直接属性（最快）
            if hasattr(layer, "mlp"):
                return "mlp"
            if hasattr(layer, "ffn"):
                return "ffn"

            # 再看子模块命名（兼容包装层）
            submods = dict(layer.named_children())
            if "mlp" in submods:
                return "mlp"
            if "ffn" in submods:
                return "ffn"

        # 保守默认：多数HF实现更常见mlp。
        return "mlp"

    # ------------------------------------------------------------------ #
    # Checkpoint → model name mapping                                    #
    # ------------------------------------------------------------------ #
    def _ckpt_to_model_name(self, ckpt_name: str) -> str:
        """Map DeepSeek-V4 checkpoint key to HF model parameter path.

        DeepSeek-V4 uses ``layers.X``, ``embed.``, ``norm.`` etc. in the
        checkpoint, while the HF model has ``model.layers.X``,
        ``model.embed.``, ``model.norm.`` etc.
        """
        name = ckpt_name.replace(".language_model", "")

        # Prefix mappings (order matters: longest first to avoid partial match).
        for ckpt_prefix, model_prefix in _CKPT_PREFIX_MAP:
            if name.startswith(ckpt_prefix):
                name = model_prefix + name[len(ckpt_prefix):]
                break

        # Suffix mappings.
        for ckpt_suffix, model_suffix in _CKPT_SUFFIX_MAP:
            if name.endswith(ckpt_suffix):
                name = name[: -len(ckpt_suffix)] + model_suffix
                break

        # Normalize MoE block naming across HF variants.
        if self._MOE_BLOCK_ATTR == "mlp":
            name = name.replace(".ffn.", ".mlp.")
        elif self._MOE_BLOCK_ATTR == "ffn":
            name = name.replace(".mlp.", ".ffn.")

        # DeepSeek MLP-family aliases: w1/w3/w2 -> gate/up/down_proj.
        # Use segment-level regex mapping so it covers:
        # - *.w{1,2,3}.weight / .bias / .scale / .weight_scale
        # - potential trailing token forms ending with .w{1,2,3}
        name = re.sub(r"(?<=\.)w1(?=\.|$)", "gate_proj", name)
        name = re.sub(r"(?<=\.)w3(?=\.|$)", "up_proj", name)
        name = re.sub(r"(?<=\.)w2(?=\.|$)", "down_proj", name)

        return name

    # ------------------------------------------------------------------ #
    # Expert-container path resolution (override of base default)        #
    # ------------------------------------------------------------------ #
    def _fused_experts_module_path(self, layer_prefix):
        return f"{layer_prefix}.{self._MOE_BLOCK_ATTR}.experts"

    # ------------------------------------------------------------------ #
    # Layer indexing / naming                                            #
    # ------------------------------------------------------------------ #
    def get_layer_module(self, idx):
        return self._layers_module[idx]

    def move_embed_to(self, device):
        # DeepSeek-V4 ckpt key is `embed.weight` (mapped to model-internal name).
        pairs = self._names_from_ckpt(["embed"])
        if device == "cuda":
            self._set_tensors(pairs)
        else:
            model_names = [self._ckpt_to_model_name(n if isinstance(n, str) else n[0]) for n in pairs]
            self._offload_names_to_meta(model_names)

    def move_output_heads_to(self, device):
        # DeepSeek-V4 ckpt keys are `norm.*` and `head.weight`.
        pairs = []
        pairs += self._names_from_ckpt("norm")
        pairs += self._names_from_ckpt("head")
        if device == "cuda":
            self._set_tensors(pairs)
        else:
            model_names = [self._ckpt_to_model_name(n if isinstance(n, str) else n[0]) for n in pairs]
            self._offload_names_to_meta(model_names)

    def _is_moe_layer(self, layer_idx):
        if layer_idx in self.mlp_only_layers:
            return False
        if layer_idx < self.first_k_dense_replace:
            return False
        if self.num_experts <= 0:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        moe_attr = self._MOE_BLOCK_ATTR

        # DeepSeek-V4不同HF版本在子模块命名上差异较大（如self_attn/hc_attn_*）。
        # 采用“整层前缀”兜底加载，确保layernorm/attention/mlp权重都不会漏载。
        non_mlp = [base]

        if not self._is_moe_layer(layer_idx):
            return {"non_mlp": non_mlp, "mlp": []}

        mlp_offload_params = [
            f"{base}.{moe_attr}.experts.gate_up_proj",
            f"{base}.{moe_attr}.experts.down_proj",
        ]
        return {
            "non_mlp": non_mlp,
            "mlp": [],
            "mlp_offload_params": mlp_offload_params,
        }

    # ------------------------------------------------------------------ #
    # Attention call (DeepSeek MLA may return a tuple)                   #
    # ------------------------------------------------------------------ #
    def _run_attention(self, layer, layer_idx, hidden_states, additional_inputs):
        out = layer.self_attn(hidden_states, **additional_inputs)
        if isinstance(out, tuple):
            return out[0]
        return out

    # ------------------------------------------------------------------ #
    # Online dequant scaffolding                                         #
    # ------------------------------------------------------------------ #
    def _install_bf16_fused_experts(self):
        """Replace ``layer.ffn.experts`` with ``_FusedBF16Experts`` on every MoE layer."""
        if self.fused_expert_intermediate_size is None:
            raise RuntimeError(
                "fused_expert_intermediate_size is not set; cannot build BF16 expert containers."
            )
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        hidden_size = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size")

        moe_attr = self._MOE_BLOCK_ATTR
        for idx in range(self.num_layers):
            if not self._is_moe_layer(idx):
                continue
            layer = self.get_layer_module(idx)
            block = getattr(layer, moe_attr, None)
            if block is None:
                # Fallback to ``mlp`` if attribute differs from expectation.
                block = getattr(layer, "mlp", None)
            if block is None or not hasattr(block, "experts"):
                continue
            new_experts = _FusedBF16Experts(
                num_experts=self.num_experts,
                intermediate_size=self.fused_expert_intermediate_size,
                hidden_size=hidden_size,
                dtype=self.dtype,
            )
            block.experts = new_experts

    def _patch_moe_forward_if_needed(self):
        """Best-effort patch: rewire each MoE layer's MoE forward to use
        the new BF16 fused experts container.

        Generic top-k SwiGLU MoE forward; uses the original ``gate`` and
        ``shared_experts`` (if present) and the new ``_FusedBF16Experts``.
        Set ``self.disable_moe_patch = True`` before init to skip.
        """
        if getattr(self, "disable_moe_patch", False):
            return

        wrapper = self
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        topk = (
            getattr(text_cfg, "num_experts_per_tok", None)
            or getattr(text_cfg, "moe_topk", None)
            or 2
        )
        norm_topk_prob = getattr(text_cfg, "norm_topk_prob", True)

        def _moe_forward(self_block, hidden_states):
            orig_shape = hidden_states.shape
            x = hidden_states.reshape(-1, orig_shape[-1])
            x_dtype = x.dtype

            router_logits = self_block.gate(x.to(x_dtype))
            if isinstance(router_logits, tuple):
                router_logits = router_logits[0]
            scores = torch.softmax(router_logits.float(), dim=-1)

            topk_scores, topk_idx = torch.topk(scores, k=topk, dim=-1)
            if norm_topk_prob:
                topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
            topk_scores = topk_scores.to(x_dtype)

            out = torch.zeros_like(x)
            for k in range(topk):
                expert_ids = topk_idx[:, k]
                weights = topk_scores[:, k].unsqueeze(-1)
                for e in range(wrapper.num_experts):
                    mask = expert_ids == e
                    if not mask.any():
                        continue
                    sub = x[mask]
                    y = self_block.experts.forward_expert(e, sub.to(x_dtype))
                    out[mask] += weights[mask] * y

            shared = getattr(self_block, "shared_experts", None)
            if shared is not None:
                shared_out = shared(hidden_states)
                if isinstance(shared_out, tuple):
                    shared_out = shared_out[0]
                out = out + shared_out.reshape(-1, orig_shape[-1]).to(out.dtype)

            return out.reshape(orig_shape)

        moe_attr = self._MOE_BLOCK_ATTR
        for idx in range(self.num_layers):
            if not self._is_moe_layer(idx):
                continue
            layer = self.get_layer_module(idx)
            block = getattr(layer, moe_attr, None) or getattr(layer, "mlp", None)
            if block is None or not hasattr(block, "experts"):
                continue
            if not isinstance(block.experts, _FusedBF16Experts):
                continue
            block.forward = _moe_forward.__get__(block, block.__class__)
