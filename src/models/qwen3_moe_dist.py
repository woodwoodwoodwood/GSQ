import os
import time
import math
import torch
import torch.distributed as dist
import gc
from .qwen3_moe import Qwen3MoeWrapper
from src.moe.placement import ExpertSharder
from src.moe.autograd_ops import AllToAllTokens
from src.prior.gptq import *
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import (
    report_gptq_calib, report_ppl_layer,
)


class Qwen3MoeDistributedWrapper(Qwen3MoeWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long
        )

        # Route debug (off by default)
        self._route_debug = os.environ.get("GSQ_ROUTE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
        try:
            self._route_debug_interval = max(1, int(os.environ.get("GSQ_ROUTE_DEBUG_INTERVAL", "20")))
        except ValueError:
            self._route_debug_interval = 20
        self._route_debug_step = 0

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        moe_attr = self._moe_block_attr()
        if not self._is_moe_layer(layer_idx):
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            local_expert = [
                f"{base}.{moe_attr}"
            ]
        else:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.{moe_attr}.gate",
                f"{base}.{moe_attr}.shared_experts",
                f"{base}.post_attention_layernorm"
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
            return {"non_mlp": non_mlp, "mlp": local_expert, "mlp_offload_params": mlp_offload_params}
        return {"non_mlp": non_mlp, "mlp": local_expert}

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        ids_buf = data_all.get('input_ids', None) if isinstance(data_all, dict) else None
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)
            additional_layer_inputs = {"attention_mask": None}
            for k, v in self.kwargs.items():
                additional_layer_inputs[k] = v

            hidden_states = current_layer.input_layernorm(x)
            attn_out, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
            mlp_input = x + attn_out
            if not self._is_moe_layer(self.current_layer_idx):
                hidden_states = current_layer.post_attention_layernorm(mlp_input)
                hidden_states = self._moe_block(current_layer)(hidden_states)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                out = hidden_states + mlp_input
            else:
                # Pass real input_ids so hash-routed MoE (DeepSeek-V4-Flash)
                # uses ``tid2eid[input_ids]`` instead of arange dummies.
                if ids_buf is not None:
                    batch_ids = ids_buf[start_idx:end_idx].to(self.device, non_blocking=True)
                else:
                    batch_ids = None
                out = self.run_expert_parallel(mlp_input, input_ids=batch_ids)

            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch, input_ids=None):
        if not self._is_moe_layer(self.current_layer_idx):
            current_layer = self.get_layer_module(self.current_layer_idx)
            hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
            hidden_states = self._moe_block(current_layer)(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            return hidden_states + mlp_input_batch
        return self.run_expert_parallel(mlp_input_batch, input_ids=input_ids)

    def _router_topk(self, router, flat_hidden, input_ids=None):
        # Forward to the parent (Qwen3MoeWrapper) implementation which handles
        # the input_ids plumbing for hash-routed MoE (DeepSeek-V4-Flash) and
        # falls back to ``arange`` dummies for non-hash routers.
        return super()._router_topk(router, flat_hidden, input_ids=input_ids)

    def _dispatch_tokens(self, mlp_input_batch, input_ids=None, force_round_robin=False):
        """Route tokens to expert-owning ranks via all-to-all.

        ``input_ids`` (optional): real (B, T) vocab token ids; required by hash
        routers (DeepSeek-V4-Flash) that look up ``tid2eid[input_ids]`` for
        expert selection. Falls back to ``arange`` dummies when None — that
        fallback collapses every batch onto the same ~64 experts and starves
        the rest of GPTQ calibration data.

        ``force_round_robin`` (default False): bypass the real router and
        deterministically assign each token to ``top_k`` experts using a
        rotating-modulo schedule, so every expert receives roughly
        ``B*T*top_k / num_experts`` tokens per batch. Used only by GPTQ
        Hessian accumulation when the production router would starve a large
        fraction of experts of calibration data — GSQ training and inference
        always go through the real router and are unaffected.
        """
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)

        router = self._moe_block(layer).gate
        if force_round_robin:
            # Resolve top_k from the router (DSv4 stores ``top_k`` directly;
            # other models may use ``num_experts_per_tok``).
            top_k = (
                getattr(router, "top_k", None)
                or getattr(router, "num_experts_per_tok", None)
                or 6
            )
            num_experts = self.num_experts
            N = B * T
            # token i -> experts [(i*top_k + j + step_off) % num_experts for j in 0..top_k]
            # ``step_off`` rotates per call so different calib batches cover
            # different (token, expert) pairs, improving Hessian diversity.
            if not hasattr(self, "_rr_step_off"):
                self._rr_step_off = 0
                if self.rank == 0:
                    print(
                        f"[GPTQ-CALIB] round-robin enabled: top_k={top_k}, "
                        f"num_experts={num_experts}; bypassing real router for "
                        f"Hessian accumulation only (GSQ training/inference unaffected)",
                        flush=True,
                    )
            base = (
                torch.arange(N * top_k, device=device, dtype=torch.long)
                + self._rr_step_off
            )
            self._rr_step_off = (self._rr_step_off + N * top_k) % num_experts
            topi = (base % num_experts).view(N, top_k)
            # Hessian accumulation only depends on the *input* x_flat per
            # expert, not on the gating weights. Use uniform 1/top_k so the
            # downstream forward output stays sane for any unrelated paths.
            topw = torch.full(
                (N, top_k), 1.0 / float(top_k),
                device=device, dtype=self.dtype,
            )
        else:
            topw, topi, top_k = self._router_topk(router, hidden.reshape(-1, H), input_ids=input_ids)

        tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)
        w_flat = topw.reshape(-1).to(self.dtype)

        owner_lut = self._owner_lut.to(device)
        owners_flat = owner_lut[eid_flat]
        perm = torch.argsort(owners_flat, stable=True)
        owners_flat = owners_flat.index_select(0, perm)
        send_idx_flat = tok_idx_flat.index_select(0, perm)
        send_eid_flat = eid_flat.index_select(0, perm)
        send_w_flat = w_flat.index_select(0, perm)
        send_x_flat = x_flat.index_select(0, send_idx_flat)

        world_size = self.world_size
        in_sizes_tensor = torch.bincount(owners_flat, minlength=world_size).to(torch.long)
        all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(world_size)]
        dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
        recv_sizes = torch.stack(all_sizes)[:, self.rank]
        out_split_sizes = recv_sizes.tolist()
        in_split_sizes = in_sizes_tensor.tolist()

        # DeepseekV4DistributedWrapper does not call Qwen3MoeDistributedWrapper.__init__,
        # so these debug attrs may be absent; lazily initialize for compatibility.
        route_debug = getattr(self, "_route_debug", None)
        if route_debug is None:
            route_debug = os.environ.get("GSQ_ROUTE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
            self._route_debug = route_debug
        if not hasattr(self, "_route_debug_interval"):
            try:
                self._route_debug_interval = max(1, int(os.environ.get("GSQ_ROUTE_DEBUG_INTERVAL", "20")))
            except ValueError:
                self._route_debug_interval = 20
        if not hasattr(self, "_route_debug_step"):
            self._route_debug_step = 0

        if route_debug:
            self._route_debug_step += 1
            if self._route_debug_step % self._route_debug_interval == 0:
                # Per-rank local send/recv view
                print(
                    f"[ROUTE][rank={self.rank}][layer={self.current_layer_idx}][step={self._route_debug_step}] "
                    f"send={in_split_sizes} recv={out_split_sizes}",
                    flush=True,
                )

                # Rank-0 global imbalance summary
                if self.rank == 0:
                    # all_sizes: list of per-rank send vectors; stack -> [src_rank, dst_rank]
                    send_matrix = torch.stack(all_sizes)
                    recv_totals = send_matrix.sum(dim=0)  # total tokens each dst rank receives
                    max_recv = int(recv_totals.max().item())
                    min_recv = int(recv_totals.min().item())
                    imbalance = float(max_recv) / float(max(1, min_recv))
                    print(
                        f"[ROUTE][global][layer={self.current_layer_idx}][step={self._route_debug_step}] "
                        f"recv_totals={recv_totals.tolist()} imbalance(max/min)={imbalance:.2f}",
                        flush=True,
                    )

        xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
        win = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).to(self.dtype)
        eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        return x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H

    def _batched_expert_forward(self, xin, eids, quantized_weights=None, gpts_calib=None):
        return self._fused_expert_forward_batched(
            xin, eids, quantized_weights=quantized_weights, gpts_calib=gpts_calib)

    def run_expert_parallel(self, mlp_input_batch, quantized_weights=None, gpts_calib=None, input_ids=None, skip_residual=False, force_round_robin=False):
        """Distributed MoE forward.

        ``skip_residual=False`` (default): returns ``MoE_out + shared_out + mlp_input_batch``.
        Standard Pre-LN residual; ``mlp_input_batch`` is the post-attention residual.

        ``skip_residual=True``: returns ``MoE_out + shared_out`` only. Used by
        Hyper-Connection models (DeepSeek-V4-Flash) where the residual is
        combined externally via the ``ffn_hc`` mapping.

        ``force_round_robin=True``: bypass the real router and use a rotating
        round-robin assignment so all experts receive calibration data. See
        ``_dispatch_tokens`` for details. Only used during GPTQ calib.
        """
        pg = dist.group.WORLD

        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch, input_ids=input_ids, force_round_robin=force_round_robin)

        out_local = self._batched_expert_forward(xin, eids, quantized_weights, gpts_calib=gpts_calib)
        xin = out_local * win

        returned = AllToAllTokens.apply(xin, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)
        y = y_flat.view(B, T, H)

        moe_block = self._moe_block(self.get_layer_module(self.current_layer_idx))
        shared_experts = getattr(moe_block, "shared_experts", None)
        if shared_experts is not None:
            shared_out = shared_experts(hidden)
            if isinstance(shared_out, tuple):
                shared_out = shared_out[0]
            y = y + shared_out.view(B, T, H).to(y.dtype)

        if skip_residual:
            return y
        return y + mlp_input_batch

    @torch.no_grad()
    def _gptq_calib_step(self, layer, x, additional_layer_inputs, batch_input_ids, gpts):
        """One step of GPTQ Hessian accumulation on this layer.

        Default Pre-LN block forward used by Qwen3-MoE / DeepSeek-V3 etc.:
        ``x = x + attn(LN(x))`` then dispatch through MoE so the hooked
        per-expert linears accumulate ``H = X^T X``. Subclasses with a
        non-Pre-LN residual (e.g. DeepSeek-V4-Flash with mHC Hyper-Connection
        residuals) override this to build the correct MoE input distribution.

        ``GSQ_CALIB_ROUND_ROBIN`` (default ``"1"``): when truthy, bypass the
        production router during calib and use a deterministic round-robin
        token→expert schedule so every expert receives Hessian samples. GSQ
        training and inference always use the real router; only calib is
        affected.
        """
        hidden_states = layer.input_layernorm(x)
        attn_out, _ = layer.self_attn(hidden_states, **additional_layer_inputs)
        x = x + attn_out

        force_rr = os.environ.get("GSQ_CALIB_ROUND_ROBIN", "1").strip().lower() in ("1", "true", "yes")
        _ = self.run_expert_parallel(
            x,
            quantized_weights=None,
            gpts_calib=gpts,
            input_ids=batch_input_ids,
            force_round_robin=force_rr,
        )

    def calculate_mse(self, mlp_input_batch, quantized_weights, self_attn=False, validation=False, accumulation_steps=1, input_ids=None):
        if not self._is_moe_layer(self.current_layer_idx):
            return super(Qwen3MoeWrapper, self).calculate_mse(
                mlp_input_batch, quantized_weights,
                self_attn=self_attn, validation=validation,
                accumulation_steps=accumulation_steps
            )
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        with torch.no_grad():
            hidden = layer.post_attention_layernorm(mlp_input_batch)
            x_flat = hidden.reshape(B * T, H)

            router = self._moe_block(layer).gate
            _, topi, top_k = self._router_topk(router, hidden.reshape(-1, H), input_ids=input_ids)

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
            dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
            recv_sizes = torch.stack(all_sizes)[:, self.rank]
            out_split_sizes = recv_sizes.tolist()
            in_split_sizes = in_sizes_tensor.tolist()

            # Reuse route-debug logging in GSQ path as well.
            route_debug = getattr(self, "_route_debug", None)
            if route_debug is None:
                route_debug = os.environ.get("GSQ_ROUTE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
                self._route_debug = route_debug
            if not hasattr(self, "_route_debug_interval"):
                try:
                    self._route_debug_interval = max(1, int(os.environ.get("GSQ_ROUTE_DEBUG_INTERVAL", "20")))
                except ValueError:
                    self._route_debug_interval = 20
            if not hasattr(self, "_route_debug_step"):
                self._route_debug_step = 0

            if route_debug:
                self._route_debug_step += 1
                if self._route_debug_step % self._route_debug_interval == 0:
                    print(
                        f"[ROUTE][rank={self.rank}][layer={self.current_layer_idx}][step={self._route_debug_step}] "
                        f"send={in_split_sizes} recv={out_split_sizes}",
                        flush=True,
                    )
                    if self.rank == 0:
                        send_matrix = torch.stack(all_sizes)
                        recv_totals = send_matrix.sum(dim=0)
                        max_recv = int(recv_totals.max().item())
                        min_recv = int(recv_totals.min().item())
                        imbalance = float(max_recv) / float(max(1, min_recv))
                        print(
                            f"[ROUTE][global][layer={self.current_layer_idx}][step={self._route_debug_step}] "
                            f"recv_totals={recv_totals.tolist()} imbalance(max/min)={imbalance:.2f}",
                            flush=True,
                        )

            xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
            eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        with torch.no_grad():
            out_fp = self._batched_expert_forward(xin, eids, quantized_weights=None)
        out_q = self._batched_expert_forward(xin, eids, quantized_weights=quantized_weights)

        total_mse = self.loss_fn(out_q, out_fp)
        if not validation:
            # When dead0 is large on this layer, some calib batches route 100%
            # of tokens to experts owned by other ranks, leaving this rank with
            # ``xin.shape[0] == 0``.  ``loss_fn(empty, empty)`` then produces a
            # tensor that does not depend on any quantized weight (no grad_fn),
            # and ``backward()`` raises ``element 0 of tensors does not require
            # grad and does not have a grad_fn``.  In that case we skip the
            # backward — this rank simply contributes zero to the gradient for
            # this micro-batch, which is correct.
            if total_mse.requires_grad and total_mse.grad_fn is not None:
                (total_mse / accumulation_steps).backward()

        return total_mse.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if not self._is_moe_layer(self.current_layer_idx):
            return super(Qwen3MoeWrapper, self).get_layer_initialization(
                trainer, gpt_all, config, logging
            )
        if logging is not None:
            logging = logging.logger
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)
        rank = self.rank
        self.configure_quantization_compression(config)

        owned_experts = [e for e in range(self.num_experts) if self.sharder.owner(e) == rank]
        subset = {}

        for e in owned_experts:
            base_prefix = f"{self._moe_base_prefix(self.get_current_layer())}.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                subset[f"{base_prefix}.{proj}"] = self._virtual_expert_linear(layer_idx, e, proj)

        init_method = config.quantization.init_method
        gsq_enabled = config.quantization.gsq_enabled

        if init_method in ("rtn", "random"):
            quantize_fn = random_quantize if init_method == "random" else rtn_quantize
            for name in subset:
                Q, scales = quantize_fn(subset[name], config, self.device, self.dtype)
                if gsq_enabled:
                    trainer.setup_layer_training(name, Q, scales)
                else:
                    self.update_quantized_weights(name, (Q, scales))
            dist.barrier()
            return

        if init_method != "gptq":
            raise ValueError(
                f"Unknown init_method={init_method!r}. Supported: 'gptq', 'rtn', 'random'"
            )

        gpts = {}
        for name in subset:
            gpts[name] = GPTQ(subset[name], name, config, self.device, self.dtype)
            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    config.gptq.wbits, perchannel=True, sym=config.gptq.sym, mse=True, trits=config.gptq.trits
                )

        # ----- Temporarily zero e_score_correction_bias during calib -----
        # TopKRouter relies on ``scores + e_score_correction_bias`` for top-k
        # selection.  When the bias dominates the scores (common on models with
        # ``num_hash_layers > 0`` where the first TopKRouter layer has a strong
        # learnt bias), the same ~N experts are selected on every calib batch
        # regardless of the input distribution.  This starves 256−N experts of
        # calibration data → identity-Hessian fallback (RTN).
        #
        # During GPTQ calibration we temporarily zero the bias so that routing
        # depends purely on ``scores`` (i.e. on the calibration data).  This
        # gives every expert a fair chance to accumulate a Hessian.  After the
        # calib loop we restore the original bias — GSQ training and inference
        # are unaffected.
        _calib_bias_orig = None
        _calib_bias_gate = None
        if self._is_moe_layer(self.current_layer_idx):
            try:
                _moe = self._moe_block(layer)
                _gate = getattr(_moe, "gate", None)
                if _gate is not None and hasattr(_gate, "e_score_correction_bias"):
                    _bias = _gate.e_score_correction_bias
                    if torch.is_tensor(_bias) and _bias.abs().sum().item() > 0.0:
                        _calib_bias_orig = _bias.detach().clone()
                        _calib_bias_gate = _gate
                        _gate.e_score_correction_bias = torch.zeros_like(_bias)
            except Exception:
                pass

        n_hessian = config.gptq.nsamples // self.world_size
        calib_start = time.time()
        calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
        ids_buf = gpt_all.get('input_ids', None) if isinstance(gpt_all, dict) else None
        with torch.no_grad():
            for j in range(n_hessian):
                x = gpt_all['input'][j].unsqueeze(0).to(self.device, non_blocking=True)

                additional_layer_inputs = {"attention_mask": None}
                for k, v in self.kwargs.items():
                    additional_layer_inputs[k] = v

                # Real input_ids for hash-routed MoE (DeepSeek-V4-Flash). Falls
                # back to ``arange`` dummies when the dataset pipeline did not
                # populate ``input_ids`` (older runs / non-hash routers).
                if ids_buf is not None:
                    batch_input_ids = ids_buf[j].unsqueeze(0).to(self.device, non_blocking=True)
                else:
                    batch_input_ids = None

                self._gptq_calib_step(
                    layer=layer,
                    x=x,
                    additional_layer_inputs=additional_layer_inputs,
                    batch_input_ids=batch_input_ids,
                    gpts=gpts,
                )
                if rank == 0 and (j + 1) % calib_report_interval == 0:
                    report_gptq_calib(j + 1, n_hessian, time.time() - calib_start)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # ----- Per-expert routing diagnostic ---------------------------------
        # During calib, each rank's owned experts accumulate their Hessian via
        # tokens that get routed to them (top-k routing + A2A dispatch). When
        # routing skews (typically at deeper layers, especially when the
        # accumulated quantization error from earlier layers shifts the
        # router-input distribution), some experts may receive 0 or very few
        # tokens, forcing GPTQ.cholesky() into the identity-Hessian fallback
        # (effectively per-channel RTN with damping for that expert).
        #
        # We emit a single concise summary per layer on rank 0 so this is
        # easy to spot in the training log, and avoid one warning per dead
        # linear (which fires 3× per dead expert: gate/up/down).
        local_zero = 0       # experts with 0 calib tokens on this rank
        local_underfed = 0   # experts with 0 < nsamples < UNDERFED_THRESHOLD
        local_owned = 0
        local_min_pos = float("inf")  # min nsamples among non-zero owned
        local_max = 0
        local_sum = 0
        UNDERFED_THRESHOLD = max(16, config.gptq.groupsize // 4)
        for nm in gpts:
            if "gate_proj" not in nm:
                # Each expert has gate/up/down sharing the same token count;
                # count once per expert via gate_proj only.
                continue
            local_owned += 1
            nsamp = gpts[nm].nsamples
            local_sum += nsamp
            if nsamp == 0:
                local_zero += 1
            else:
                if nsamp < UNDERFED_THRESHOLD:
                    local_underfed += 1
                if nsamp < local_min_pos:
                    local_min_pos = nsamp
                if nsamp > local_max:
                    local_max = nsamp
        if local_min_pos == float("inf"):
            local_min_pos = 0

        if dist.is_initialized():
            stats = torch.tensor(
                [local_owned, local_zero, local_underfed, local_sum, local_max],
                device=self.device, dtype=torch.long,
            )
            min_pos_t = torch.tensor([int(local_min_pos)], device=self.device, dtype=torch.long)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            # MIN reduce of "min among nonzero" - replace 0 with INT64_MAX so
            # ranks with all-dead experts don't drag the min to 0.
            sentinel = torch.iinfo(torch.long).max
            min_pos_for_reduce = min_pos_t if min_pos_t.item() > 0 else torch.tensor(
                [sentinel], device=self.device, dtype=torch.long
            )
            dist.all_reduce(min_pos_for_reduce, op=dist.ReduceOp.MIN)
            max_t = torch.tensor([local_max], device=self.device, dtype=torch.long)
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX)

            g_owned, g_zero, g_under, g_sum, _ = stats.tolist()
            g_min_pos = min_pos_for_reduce.item()
            if g_min_pos == sentinel:
                g_min_pos = 0
            g_max = max_t.item()
        else:
            g_owned, g_zero, g_under, g_sum = local_owned, local_zero, local_underfed, local_sum
            g_min_pos, g_max = int(local_min_pos), local_max

        if self.rank == 0:
            g_mean = (g_sum / g_owned) if g_owned > 0 else 0.0
            severity = "OK"
            if g_zero > 0:
                severity = "DEAD"
            elif g_under > 0:
                severity = "UNDERFED"

            # Router introspection: distinguish "hash router missed input_ids"
            # from "topk router bias not loaded" from "real routing collapse".
            try:
                _moe_block = self._moe_block(layer)
                _router = getattr(_moe_block, "gate", None)
                router_kind = type(_router).__name__ if _router is not None else "None"
                bias_t = getattr(_router, "e_score_correction_bias", None)
                if bias_t is None:
                    bias_status = "no_bias"
                elif not torch.is_tensor(bias_t):
                    bias_status = "not_tensor"
                else:
                    nonzero = bias_t.abs().sum().item()
                    if not torch.isfinite(bias_t).all().item():
                        bias_status = "nonfinite"
                    elif nonzero == 0.0:
                        bias_status = "all_zero"
                    else:
                        bias_status = f"loaded_abs_sum={nonzero:.3f}"
                tid2eid_t = getattr(_router, "tid2eid", None)
                if tid2eid_t is None:
                    tid_status = "no_tid2eid"
                else:
                    tid_nz = tid2eid_t.abs().sum().item() if torch.is_tensor(tid2eid_t) else -1
                    tid_status = "all_zero" if tid_nz == 0 else "loaded"
            except Exception as _e:
                router_kind = "unknown"
                bias_status = f"introspect_err={_e}"
                tid_status = "?"

            print(
                f"[GPTQ-CALIB][layer={layer_idx}] expert tokens "
                f"min/mean/max={g_min_pos}/{int(g_mean)}/{g_max} "
                f"experts(total={g_owned}, dead0={g_zero}, "
                f"under{UNDERFED_THRESHOLD}={g_under}) [{severity}] "
                f"router={router_kind} bias={bias_status} tid2eid={tid_status}",
                flush=True,
            )
            # When dead0 is non-trivial and bias is loaded, print the bias
            # distribution so we can see whether bias is dominating the routing.
            if (
                g_zero > 0
                and bias_t is not None
                and torch.is_tensor(bias_t)
                and bias_t.numel() > 1
                and bias_t.abs().sum().item() > 0.0
            ):
                b = bias_t.cpu().float()
                b_sorted, b_idx = b.sort(descending=True)
                top_n = 12
                top_info = ", ".join(
                    f"e{b_idx[i].item():d}={b_sorted[i].item():.3f}"
                    for i in range(min(top_n, b_sorted.numel()))
                )
                n_positive = (b > 0).sum().item()
                n_negative = (b < 0).sum().item()
                print(
                    f"[GPTQ-CALIB][layer={layer_idx}] bias stats: "
                    f"min={b.min().item():.3f} max={b.max().item():.3f} "
                    f"mean={b.mean().item():.3f} std={b.std().item():.3f} "
                    f"pos={n_positive} neg={n_negative} "
                    f"| top12: {top_info}",
                    flush=True,
                )
            if g_zero > 0:
                # Categorize the likely cause based on router type / bias state.
                if "Hash" in router_kind and tid_status == "all_zero":
                    reason = (
                        "HashRouter.tid2eid not loaded (all zero) -> all tokens route to expert 0. "
                        "Fix: ensure 'ffn.gate.tid2eid' ckpt key reaches the model param."
                    )
                elif "Hash" in router_kind:
                    reason = (
                        "HashRouter receiving wrong/no input_ids. "
                        "Fix: ensure data_dict['input_ids'] is populated and threaded through."
                    )
                elif bias_status == "all_zero":
                    reason = (
                        "TopKRouter e_score_correction_bias not loaded (all zero); the load-balancing "
                        "bias is what keeps DeepSeek-style routing diverse on small calib batches. "
                        "Fix: ensure 'ffn.gate.bias' ckpt key maps to 'gate.e_score_correction_bias' "
                        "on the model. dead0 will plummet once it is loaded."
                    )
                else:
                    reason = (
                        "Real routing skew on this calibration data (not a loader bug). "
                        f"Increase config.gptq.nsamples (currently {config.gptq.nsamples}), "
                        "increase data.max_length, or switch to a more diverse calib dataset."
                    )
                print(
                    f"[GPTQ-CALIB][layer={layer_idx}] WARNING: {g_zero} expert(s) "
                    f"received 0 calib tokens; identity-Hessian fallback ⇒ effectively RTN "
                    f"for these experts. Likely cause: {reason}",
                    flush=True,
                )
        # ---------------------------------------------------------------------

        # Restore the original calibration bias now that Hessian accumulation
        # is done.  The restored bias will be used during GSQ training and
        # inference — neither of which should see zeroed routing.
        if _calib_bias_gate is not None and _calib_bias_orig is not None:
            _calib_bias_gate.e_score_correction_bias = _calib_bias_orig
            _calib_bias_orig = None
            _calib_bias_gate = None

        gptq_losses = []
        for name in gpts:
            if "up_proj" in name:
                continue
            Q, scales = gpts[name].fasterquant(
                logging,
                percdamp=config.gptq.percdamp,
                blocksize=config.gptq.blocksize,
                groupsize=config.gptq.groupsize,
                static_groups=config.gptq.static_groups,
                prunen=config.gptq.prunen,
                prunem=config.gptq.prunem
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if scales is not None and gsq_enabled:
                trainer.setup_layer_training(name, Q, scales)
            else:
                self.update_quantized_weights(name, (Q, scales) if scales is not None else Q)
            if "gate_proj" in name:
                base = name[: -len(".gate_proj")]
                new_name = f"{base}.up_proj"
                gpts[new_name].H = gpts[name].H
                gpts[new_name].dead = gpts[name].dead
                Q, scales = gpts[new_name].fasterquant(
                    logging,
                    percdamp=config.gptq.percdamp,
                    blocksize=config.gptq.blocksize,
                    groupsize=config.gptq.groupsize,
                    static_groups=config.gptq.static_groups,
                    calculate_cholesky=False,
                    prunen=config.gptq.prunen,
                    prunem=config.gptq.prunem
                )
                if hasattr(gpts[new_name], 'last_gptq_loss'):
                    gptq_losses.append(gpts[new_name].last_gptq_loss)
                if scales is not None and gsq_enabled:
                    trainer.setup_layer_training(new_name, Q, scales)
                else:
                    self.update_quantized_weights(new_name, (Q, scales) if scales is not None else Q)
                gpts[name].free()
                gpts[new_name].free()
            else:
                gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

        dist.barrier()

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk and self._is_moe_layer(layer_idx):
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("open_thoughts", self.tokenizer, ppl_max_samples=128)
        testloader = prepare_test_dataloader(
                dataset=dataset["test"],
                tokenizer=self.tokenizer,
                seqlen=self.model.seqlen,
                batch_size=4,
                world_size=self.world_size,
                rank=self.rank
            )

        pad_token_id = self.model.config.pad_token_id

        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)

        input_ids_cpu_list = []
        activations_cpu_list = []

        for batch in testloader:
            ids_cpu = batch["input_ids"].to("cpu", non_blocking=True).pin_memory()
            input_ids_cpu_list.append(ids_cpu)
            activations_cpu_list.append(None)
            del batch

        num_batches = len(input_ids_cpu_list)

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        idx_copy = self.current_layer_idx

        transfer_stream = torch.cuda.Stream(device=self.device)

        self.current_layer_idx = 0
        self._load_layer_for_eval(0, read_from_disk)
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                hidden = layer.input_layernorm(x)
                attn_out, _ = layer.self_attn(hidden, **additional_layer_inputs)
                x = x + attn_out
                if not self._is_moe_layer(i):
                    hidden_states = layer.post_attention_layernorm(x)
                    hidden_states = self._moe_block(layer)(hidden_states)
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    x = hidden_states + x
                else:
                    x = self.run_expert_parallel(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)

            if i + 1 < self.num_layers:
                with torch.cuda.stream(transfer_stream):
                    self._load_layer_for_eval(i + 1, read_from_disk)
                transfer_stream.synchronize()

            torch.cuda.empty_cache()

        local_nll_sum = torch.tensor(0.0, device=self.device)
        local_tok_cnt = torch.tensor(0.0, device=self.device)

        for b in range(num_batches):
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)

            logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            nll = (nll * mask).sum(dim=1)
            tok = mask.sum(dim=1)

            local_nll_sum += nll.sum()
            local_tok_cnt += tok.sum()

            del input_ids, x, x_cpu, logits, shift_labels, nll, tok

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")

        self.current_layer_idx = idx_copy

        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (local_nll_sum / local_tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl

    def save_moe_experts_to_disc(self):
        if not self._is_moe_layer(self.current_layer_idx):
            return
        layer_key = self.get_current_layer()
        owned = [e for e in range(self.num_experts) if self.sharder.owner(e) == self.rank]
        for e in owned:
            base = f"{self._moe_base_prefix(layer_key)}.experts.{e}"
            gk = f"{base}.gate_proj"
            if gk not in self.temp_weights:
                continue
            pairs = {
                "gate_proj": (self.temp_weights[gk], self.temp_weights[f"{gk}.scale"]),
                "up_proj": (
                    self.temp_weights[f"{base}.up_proj"],
                    self.temp_weights[f"{base}.up_proj.scale"],
                ),
                "down_proj": (
                    self.temp_weights[f"{base}.down_proj"],
                    self.temp_weights[f"{base}.down_proj.scale"],
                ),
            }
            self.save_to_disc(base, pairs)
