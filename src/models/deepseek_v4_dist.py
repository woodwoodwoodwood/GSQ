"""DeepSeek-V4-Flash GSQ wrapper (distributed / expert-parallel).

Same online dequantization as ``DeepseekV4Wrapper`` (FP8 dense + MXFP4 experts)
plus the Qwen3-MoE distributed primitives (``ExpertSharder``, A2A dispatch).
Naming differences vs Qwen-MoE (``ffn``, top-level ``layers.``, ``.scale``
instead of ``.weight_scale``) and checkpoint→model prefix mapping are inherited
from ``DeepseekV4Wrapper``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .base import BaseModelWrapper
from .deepseek_v4 import DeepseekV4Wrapper
from .qwen3_moe_dist import Qwen3MoeDistributedWrapper


class DeepseekV4DistributedWrapper(DeepseekV4Wrapper, Qwen3MoeDistributedWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        # Run BaseModelWrapper.__init__ with the strip-quant-config flag,
        # then manually set DeepSeek-specific fields, install BF16 fused
        # experts, and finally attach the distributed sharder.
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

        if hasattr(self.model, "layers") and hasattr(self.model.layers, "__len__"):
            self._layers_module = self.model.layers
            self.layer_prefix = "layers"
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers_module = self.model.model.layers
            self.layer_prefix = "model.layers"
        else:
            raise RuntimeError(
                "DeepseekV4DistributedWrapper: could not locate ``layers`` ModuleList."
            )
        self.num_layers = len(self._layers_module)

        # HF implementations may expose MoE/attn/hc blocks with variant names.
        self._MOE_BLOCK_ATTR = self._detect_moe_block_attr()
        self._ATTN_BLOCK_ATTR = self._detect_attn_block_attr()
        self._HC_ATTN_ATTR = self._detect_optional_layer_attr(
            ["attn_hc", "hc_attn_base", "hc_attn"]
        )
        self._HC_FFN_ATTR = self._detect_optional_layer_attr(
            ["ffn_hc", "hc_ffn_base", "hc_mlp_base", "mlp_hc"]
        )
        self._refresh_model_tensor_name_cache()

        self.is_moe = True
        self.fused_experts = True
        self.fused_expert_intermediate_size = (
            getattr(text_cfg, "moe_intermediate_size", None)
            or getattr(text_cfg, "intermediate_size", None)
        )
        self.mxfp4_experts = True
        self.fp8_dense = True

        self._install_bf16_fused_experts()
        self._patch_moe_forward_if_needed()

        if not dist.is_initialized():
            raise RuntimeError(
                "DeepseekV4DistributedWrapper requires torch.distributed to be initialized."
            )
        from src.moe.placement import ExpertSharder

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(
            num_experts=self.num_experts,
            world_size=self.world_size,
        )
        self.local_expert_ids = [
            e for e in range(self.num_experts)
            if self.sharder.owner(e) == self.rank
        ]
        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long,
        )
        self.groupsize = 32

    # ------------------------------------------------------------------ #
    # Per-layer prefix list (rank-local)                                 #
    # ------------------------------------------------------------------ #
    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        moe_attr = self._MOE_BLOCK_ATTR

        if not self._is_moe_layer(layer_idx):
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm",
            ]
            local_expert = [f"{base}.{moe_attr}"]
            return {"non_mlp": non_mlp, "mlp": local_expert}

        non_mlp = [
            f"{base}.input_layernorm",
            f"{base}.self_attn",
            f"{base}.{moe_attr}.gate",
            f"{base}.{moe_attr}.shared_experts",
            f"{base}.post_attention_layernorm",
        ]
        local_expert = [
            f"{base}.{moe_attr}.experts.{e}"
            for e in range(self.num_experts)
            if self.sharder.owner(e) == self.rank
        ]
        mlp_offload_params = [
            f"{base}.{moe_attr}.experts.gate_up_proj",
            f"{base}.{moe_attr}.experts.down_proj",
        ]
        return {
            "non_mlp": non_mlp,
            "mlp": local_expert,
            "mlp_offload_params": mlp_offload_params,
        }

    # ------------------------------------------------------------------ #
    # PPL evaluation                                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        """DeepSeek-V4-Flash is structurally incompatible with the Qwen-MoE
        layer-by-layer PPL path:

        * Layer residuals are mHC (Manifold-Constrained Hyper-Connections, 4D
          ``[B, S, hc_mult, D]`` state with Sinkhorn-projected mixing matrices),
          not the standard Pre-LN ``x + attn(LN(x))`` / ``x + mlp(LN(x))`` that
          ``Qwen3MoeDistributedWrapper.ppl_evaluation`` assumes.
        * ``DeepseekV4Attention.forward`` requires ``position_embeddings`` (a
          ``{"main", "compress"}`` rotary dict) and ``s_aux=self.sinks`` —
          neither is plumbed by the layer-by-layer path.
        * The MoE block is a HashRouter that needs ``input_ids``.
        * Two attention sub-paths use compressors (HCA / CSA) with their own
          rolling-window state.

        Calling the inherited PPL would (and does) trigger asynchronous
        ``cudaErrorIllegalAddress`` from MLA + zero-initialised ``sinks`` /
        missing position embeddings. We disable it here and log clear guidance.

        Health monitoring during quantization should use:
          * ``gptq/avg_loss`` (per-layer, in W&B)
          * ``{layer}/val_hard_loss`` (per-epoch, in W&B; the Gumbel-softmax
            validation MSE is the same target the quantizer optimizes)

        End-to-end perplexity / lm-eval should be run post-training via
        ``eval_model.py`` against a vLLM server with the assembled checkpoint.
        """
        msg = (
            "[DeepseekV4DistributedWrapper] ppl_evaluation is disabled: the "
            "layer-by-layer PPL path is incompatible with mHC residuals, MLA "
            "position_embeddings, and HashRouter input_ids in DeepSeek-V4-Flash. "
            "Use post-training vLLM + eval_model.py for true PPL/benchmarks; "
            "monitor per-layer health via gptq/avg_loss and {layer}/val_hard_loss "
            "in W&B."
        )
        if self.rank == 0 and not getattr(self, "_ppl_disabled_logged", False):
            print(msg, flush=True)
            self._ppl_disabled_logged = True

        return float("nan")
