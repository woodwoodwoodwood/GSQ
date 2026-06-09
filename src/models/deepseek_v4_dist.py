"""DeepSeek-V4-Flash GSQ wrapper (distributed / expert-parallel).

Same online dequantization as ``DeepseekV4Wrapper`` (FP8 dense + MXFP4 experts)
plus the Qwen3-MoE distributed primitives (``ExpertSharder``, A2A dispatch).
Naming differences vs Qwen-MoE (``ffn``, top-level ``layers.``, ``.scale``
instead of ``.weight_scale``) and checkpoint→model prefix mapping are inherited
from ``DeepseekV4Wrapper``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .base import BaseModelWrapper
from .deepseek_v4 import DeepseekV4Wrapper
from .qwen3_moe_dist import Qwen3MoeDistributedWrapper
from src.moe.autograd_ops import AllToAllTokens


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

        # Manifold-Constrained Hyper-Connection (mHC) multiplicity. The hidden
        # state through every DeepseekV4DecoderLayer is shaped
        # ``[B, S, hc_mult, D]``, mixed in/out by ``attn_hc`` / ``ffn_hc``.
        # See ``modular_deepseek_v4.py`` lines 765-841 (paper §2.2). Mirrors
        # the same assignment in ``DeepseekV4Wrapper.__init__``; required here
        # because this class bypasses ``DeepseekV4Wrapper.__init__`` and goes
        # straight to ``BaseModelWrapper.__init__``.
        self.hc_mult = int(getattr(text_cfg, "hc_mult", 1))

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

    # ====================================================================== #
    # mHC (Manifold-Constrained Hyper-Connection) forward                     #
    # ====================================================================== #
    #
    # DeepSeek-V4-Flash decoder layers carry a 4D residual state
    # ``[B, S, hc_mult, D]`` instead of the standard 3D ``[B, S, D]``. Each
    # layer mixes the streams in/out via two ``DeepseekV4HyperConnection``
    # modules (``attn_hc`` / ``ffn_hc``) — see ``modular_deepseek_v4.py:1011-1021``.
    #
    # ``HyperConnection.forward(hidden_4D)`` returns ``(post, comb, collapsed)``
    # where ``collapsed`` is a 3D weighted sum across the ``hc_mult`` axis (the
    # input the sublayer attention/MLP sees) and ``post`` / ``comb`` are the
    # per-stream output projection / Sinkhorn-projected combine matrix used to
    # produce the next 4D state from the sublayer's 3D output.
    #
    # The overrides below replace the Pre-LN ``x + sublayer(LN(x))`` residual
    # used by ``Qwen3MoeDistributedWrapper`` with the correct mHC residual.

    @torch.no_grad()
    def _attn_site_forward(self, layer, hidden_4D, additional_layer_inputs):
        """Apply the attention site of one decoder layer with mHC residual.

        ``hidden_4D`` shape: ``[B, S, hc_mult, D]``. Returns the post-attention
        4D residual state (mirrors HF lines 1011-1015).
        """
        dtype = hidden_4D.dtype
        post, comb, collapsed = layer.attn_hc(hidden_4D)
        attn_out = layer.self_attn(layer.input_layernorm(collapsed), **additional_layer_inputs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        # post: [B, S, H], attn_out: [B, S, D], comb: [B, S, H, H], hidden_4D: [B, S, H, D]
        return (
            post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2)
            + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_4D)
        )

    @torch.no_grad()
    def _ffn_collapse(self, layer, hidden_4D):
        """Apply ffn_hc to get the 3D collapsed input for the MoE block.

        Returns ``(post, comb, collapsed_3D)``. The caller runs the MoE on
        ``collapsed_3D`` and combines via ``_ffn_combine`` to produce the
        next 4D state.
        """
        post, comb, collapsed = layer.ffn_hc(hidden_4D)
        return post, comb, collapsed

    def _ffn_combine(self, post, comb, hidden_4D, mlp_output_3D):
        """4D state, 3D MoE output, ffn_hc tensors → next 4D state.

        Mirrors HF lines 1019-1021.
        """
        dtype = hidden_4D.dtype
        return (
            post.to(dtype).unsqueeze(-1) * mlp_output_3D.unsqueeze(-2)
            + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_4D)
        )

    @torch.no_grad()
    def _build_layer_inputs(self):
        """Re-create ``additional_layer_inputs`` dict with captured kwargs."""
        kwargs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            kwargs[k] = v
        return kwargs

    # ---- Overrides --------------------------------------------------------- #

    @torch.no_grad()
    def get_mlp_input(self, batch):
        """Override: full attn site, 4D in → 4D out.

        Parent class assumes Pre-LN: ``x + attn(LN(x))``. mHC needs the
        attn_hc combine instead.
        """
        layer = self.get_layer_module(self.current_layer_idx)
        return self._attn_site_forward(layer, batch, self._build_layer_inputs())

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch, input_ids=None):
        """Override: full ffn site, 4D in → 4D out.

        ``mlp_input_batch`` here is the 4D post-attention residual produced
        by ``get_mlp_input`` above (NOT a 3D MoE input as in the parent class).
        """
        layer = self.get_layer_module(self.current_layer_idx)

        post, comb, collapsed = self._ffn_collapse(layer, mlp_input_batch)

        if not self._is_moe_layer(self.current_layer_idx):
            # Dense MLP path: this branch does NOT go through ``_dispatch_tokens``,
            # which is where the MoE branch picks up its ``post_attention_layernorm``.
            # To match HF line 1018 (one LN before the MLP), apply it explicitly here.
            mlp_out = self._moe_block(layer)(layer.post_attention_layernorm(collapsed))
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
        else:
            # Distributed MoE: ``run_expert_parallel`` (via ``_dispatch_tokens``)
            # applies ``post_attention_layernorm`` internally, so we MUST pass
            # the pre-LN ``collapsed`` here. Passing ``post_LN`` would cause a
            # double LN that does not match HF's single LN at line 1018.
            mlp_out = self.run_expert_parallel(
                collapsed,
                quantized_weights=None,
                input_ids=input_ids,
                skip_residual=True,
            )
        return self._ffn_combine(post, comb, mlp_input_batch, mlp_out)

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        """Override: 4D activation propagation via full mHC layer forward.

        Reads layer-input 4D from ``data_all['input']``, applies the full
        decoder layer (attn site + ffn site, both with mHC), writes layer-
        output 4D back to ``data_all['input']``.
        """
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        ids_buf = data_all.get('input_ids', None) if isinstance(data_all, dict) else None
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)

            # Attn site
            x = self._attn_site_forward(current_layer, x, self._build_layer_inputs())

            # Ffn site (real input_ids for HashRouter, None for TopKRouter)
            if ids_buf is not None:
                batch_ids = ids_buf[start_idx:end_idx].to(self.device, non_blocking=True)
            else:
                batch_ids = None
            x = self.get_mlp_output(x, input_ids=batch_ids)

            data_all['input'][start_idx:end_idx] = x.detach().cpu()

    def calculate_mse(self, mlp_input_batch, quantized_weights, self_attn=False, validation=False, accumulation_steps=1, input_ids=None):
        """Override: HC-aware MoE-output MSE.

        ``mlp_input_batch`` is the 4D post-attn residual produced by
        ``get_mlp_input_all``. We apply ``ffn_hc`` to get the 3D collapsed
        input that the MoE actually sees, then dispatch tokens and compute
        the MSE between quantized and unquantized MoE outputs (3D), exactly
        like the parent class — only the input shape and the ffn_hc step
        differ.
        """
        # Dense (non-MoE) layers — DeepSeek-V4-Flash has none, but be safe.
        if not self._is_moe_layer(self.current_layer_idx):
            return super().calculate_mse(
                mlp_input_batch, quantized_weights,
                self_attn=self_attn, validation=validation,
                accumulation_steps=accumulation_steps,
                input_ids=input_ids,
            )

        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device

        # mHC ffn-site collapse: 4D → 3D collapsed
        with torch.no_grad():
            post, comb, collapsed = self._ffn_collapse(layer, mlp_input_batch)
            hidden = layer.post_attention_layernorm(collapsed)

            B, T, H = hidden.shape
            x_flat = hidden.reshape(B * T, H)

            router = self._moe_block(layer).gate
            _, topi, top_k = self._router_topk(router, x_flat, input_ids=input_ids)

            tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
            eid_flat = topi.reshape(-1).to(torch.long)

            owner_lut = self._owner_lut.to(device)
            owners_flat = owner_lut[eid_flat]
            perm = torch.argsort(owners_flat, stable=True)
            owners_flat = owners_flat.index_select(0, perm)
            send_idx_flat = tok_idx_flat.index_select(0, perm)
            send_eid_flat = eid_flat.index_select(0, perm)
            send_x_flat = x_flat.index_select(0, send_idx_flat)

            world_size = self.world_size
            in_sizes_tensor = torch.bincount(owners_flat, minlength=world_size).to(torch.long)
            all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(world_size)]
            dist.all_gather(all_sizes, in_sizes_tensor)
            recv_sizes = torch.stack(all_sizes)[:, self.rank]
            out_split_sizes = recv_sizes.tolist()
            in_split_sizes = in_sizes_tensor.tolist()

            pg = dist.group.WORLD
            xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
            eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        with torch.no_grad():
            out_fp = self._batched_expert_forward(xin, eids, quantized_weights=None)
        out_q = self._batched_expert_forward(xin, eids, quantized_weights=quantized_weights)

        total_mse = self.loss_fn(out_q, out_fp)
        if not validation:
            # Same dead-batch guard as the parent (rank with 0 routed tokens).
            if total_mse.requires_grad and total_mse.grad_fn is not None:
                (total_mse / accumulation_steps).backward()

        # Free the unused HC tensors.
        del post, comb, collapsed
        return total_mse.item()

    @torch.no_grad()
    def _gptq_calib_step(self, layer, x, additional_layer_inputs, batch_input_ids, gpts):
        """Override: HC-aware GPTQ Hessian accumulation step.

        ``x`` is the 4D layer input ``[B, S, hc_mult, D]`` from ``gpt_all``.
        Build the correct 3D MoE input via attn site + ffn_hc, then dispatch
        through ``run_expert_parallel`` with ``gpts_calib=gpts`` so the per-
        expert linears see the production-time activation distribution.

        ``GSQ_CALIB_ROUND_ROBIN`` (default ``"1"``): when truthy, bypass the
        production TopKRouter during calib and use a deterministic round-
        robin token→expert schedule so all 256 experts receive Hessian
        samples. DSv4 layer 0-2 use HashRouter (``input_ids``-driven) which
        already covers all experts; layer 3+ use TopKRouter where the
        learned ``gate.weight`` projection consistently starves ~150 of 256
        experts on any reasonable English calib distribution. GSQ training
        and inference always use the real router and are unaffected.
        """
        # Attn site: 4D → 4D
        x = self._attn_site_forward(layer, x, additional_layer_inputs)

        # Ffn site: 4D → 3D collapsed → MoE Hessian.
        # ``run_expert_parallel`` (via ``_dispatch_tokens``) applies
        # ``post_attention_layernorm`` internally, so feed it pre-LN
        # ``collapsed`` to match HF's single LN at line 1018.
        if self._is_moe_layer(self.current_layer_idx):
            post, comb, collapsed = self._ffn_collapse(layer, x)
            del post, comb  # only need collapsed for Hessian
            force_rr = os.environ.get("GSQ_CALIB_ROUND_ROBIN", "1").strip().lower() in ("1", "true", "yes")
            _ = self.run_expert_parallel(
                collapsed,
                quantized_weights=None,
                gpts_calib=gpts,
                input_ids=batch_input_ids,
                skip_residual=True,
                force_round_robin=force_rr,
            )
        # Dense MLP layers: nothing to calibrate via expert parallel.
