import os
import re
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
import json
from pathlib import Path
from types import SimpleNamespace
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM  
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from compressed_tensors import PackedQuantizationCompressor
from contextlib import ExitStack
from src.prior.quant import Quantizer
from src.prior.gptq import GPTQ, rtn_quantize
from src.utils.progress_reporter import report_gptq_calib, report_gptq_linear

class BaseModelWrapper(ABC):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype,
                 dummy=False, strip_quantization_config=False):
        self.ckpt_path = self.resolve_model_path(model_name)
        self.dtype = self.normalize_dtype(dtype)
        self.device = device
        self.save_dir = None

        cfg = AutoConfig.from_pretrained(self.ckpt_path, trust_remote_code=True)

        # When the source checkpoint is quantized in a foreign format (e.g.
        # MXFP4 for DeepSeek-V4-Flash), the embedded ``quantization_config``
        # would force ``from_config`` to materialize uint8/packed storage.
        # Stripping it lets us build a vanilla BF16/FP16 model and perform
        # online dequantization while loading the weights.
        if strip_quantization_config:
            for sub_cfg in (cfg, getattr(cfg, "text_config", None)):
                if sub_cfg is None:
                    continue
                for attr in ("quantization_config", "quantization", "compression_config",
                             "expert_dtype"):
                    if getattr(sub_cfg, attr, None) is not None:
                        try:
                            setattr(sub_cfg, attr, None)
                        except (AttributeError, TypeError):
                            # Frozen dataclass / __slots__: fall back to
                            # __dict__ manipulation so we actually strip it.
                            if hasattr(sub_cfg, "__dict__"):
                                sub_cfg.__dict__[attr] = None
                            else:
                                raise RuntimeError(
                                    f"Cannot strip '{attr}' from "
                                    f"{type(sub_cfg).__name__}"
                                ) from None

        # HF only sets _attn_implementation on the top-level config; propagate to
        # sub-configs so decoder layers pick up the same attention backend.
        # Some architectures (e.g. DeepSeek-V4 in certain HF versions) do not
        # support SDPA and require eager mode; auto-fallback to eager on failure.
        preferred_attn_impl = os.environ.get("GSQ_ATTN_IMPLEMENTATION", "sdpa")

        def _build_empty_model(attn_impl: str):
            if hasattr(cfg, 'text_config') and cfg.text_config is not None:
                cfg.text_config._attn_implementation = attn_impl
                return AutoModelForCausalLM.from_config(
                    cfg.text_config,
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                ).eval()
            return AutoModelForCausalLM.from_config(
                cfg,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            ).eval()

        with init_empty_weights():
            try:
                empty_model = _build_empty_model(preferred_attn_impl)
            except ValueError as e:
                msg = str(e)
                if preferred_attn_impl == "sdpa" and (
                    "does not support an attention implementation" in msg
                    or "scaled_dot_product_attention" in msg
                ):
                    print(
                        "[BaseModelWrapper] WARNING: model/config does not support "
                        "attn_implementation='sdpa'; fallback to 'eager'."
                    )
                    empty_model = _build_empty_model("eager")
                else:
                    raise

        self.model = empty_model

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.model.seqlen = seqlen
        self.model.config.use_cache = False

        self.current_layer_idx = 0
        self.loss_fn = torch.nn.MSELoss()

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dummy = dummy
        self.kwargs = {}
        self._attention_mask_1 = None  # single-item causal mask, expanded on demand
        self.meta_init_std = 0.02
        self.calib_report_divisor = 10
        self.batch_report_divisor = 5

        if dummy:
            self._name_to_shard = None
            self._single_shard = None
        else:
            self._name_to_shard, self._single_shard = self._load_safetensor_index(self.ckpt_path)
        self.compressor = PackedQuantizationCompressor()

        self.quantization_config = self.dict_to_ns({
            "config_groups": {
              "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": [
                  "Linear"
                ],
                "weights": {
                  "actorder": None,
                  "block_structure": None,
                  "dynamic": False,
                  "group_size": 32,
                  "num_bits": 4,
                  "observer": "minmax",
                  "observer_kwargs": {},
                  "strategy": "group",
                  "symmetric": True,
                  "type": "int"
                }
              }
            },
            "format": "pack-quantized",
            "ignore": [
              "lm_head",
              "re:.*self_attn.*"
            ],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
          })
        self.temp_weights = {}
        self.is_moe = False
        self.fused_experts = False
        self.fused_expert_intermediate_size = None

        # When ``True``, ``_set_tensors`` will pair ``...experts.{e}.w{1,2,3}.weight``
        # (uint8 packed E2M1 nibbles) with ``..scale`` (uint8 UE8M0 block
        # scales) and dequantize on-the-fly to ``self.dtype`` before writing
        # them into the fused BF16 expert slots. Subclasses (e.g.
        # ``DeepseekV4Wrapper``) flip this on.
        self.mxfp4_experts = False
        # Internal buffer used to pair packed/scale tensors that may arrive in
        # any order while iterating safetensors keys.
        self._mxfp4_pending = {}

        # When ``True``, ``_set_tensors`` will also detect FP8 E4M3 dense
        # linear weights (``float8_e4m3fn``) paired with ``.scale`` tensors
        # (UE8M0), dequantize on-the-fly, and write BF16 into the model.
        # Subclasses (e.g. ``DeepseekV4Wrapper``) flip this on.
        self.fp8_dense = False
        self._fp8_pending = {}

    _FUSED_EXPERT_WEIGHT_KEY_RE = re.compile(
        r"^(.+\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    _FUSED_EXPERT_MODULE_RE = re.compile(
        r"^(.+\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
    )

    # DeepSeek-style naming: ``w1`` (gate_proj), ``w3`` (up_proj), ``w2`` (down_proj),
    # with companion ``.weight_scale`` (DeepSeek-V3) or ``.scale`` (DeepSeek-V4-Flash)
    # tensor for MXFP4 block scales. The MoE submodule may be named ``mlp`` (Qwen)
    # or ``ffn`` (DeepSeek-V4); the leading ``model.`` prefix is also optional.
    _MXFP4_FUSED_EXPERT_KEY_RE = re.compile(
        r"^(.*layers\.\d+)\.(?:mlp|ffn)\.experts\.(\d+)\.(w1|w2|w3)\.(weight|weight_scale|scale)$"
    )
    _MXFP4_PROJ_ALIAS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    def _parse_fused_expert_weight_param_name(self, name):
        m = self._FUSED_EXPERT_WEIGHT_KEY_RE.match(name)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
        return None

    def _parse_fused_expert_module_base(self, base_name):
        m = self._FUSED_EXPERT_MODULE_RE.match(base_name)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
        return None

    def _parse_mxfp4_expert_key(self, name):
        """Match ``...experts.{e}.w{1,2,3}.{weight,weight_scale,scale}``.

        Returns ``(layer_prefix, expert_idx, canonical_proj_kind, kind)`` where
        ``canonical_proj_kind`` is one of ``gate_proj``/``up_proj``/``down_proj``
        and ``kind`` is normalized to ``"weight"`` or ``"weight_scale"``.
        """
        m = self._MXFP4_FUSED_EXPERT_KEY_RE.match(name)
        if not m:
            return None
        layer_prefix = m.group(1)
        expert_idx = int(m.group(2))
        proj_kind = self._MXFP4_PROJ_ALIAS[m.group(3)]
        raw_kind = m.group(4)
        kind = "weight_scale" if raw_kind in ("weight_scale", "scale") else "weight"
        return layer_prefix, expert_idx, proj_kind, kind

    def _try_dequant_mxfp4_pending(self, key):
        """If both packed weight and scale tensors for ``key`` are buffered,
        dequantize and write the BF16 result into the fused expert slot.
        """
        entry = self._mxfp4_pending.get(key)
        if entry is None:
            return
        if "weight" not in entry or "weight_scale" not in entry:
            return
        layer_prefix, expert_idx, proj_kind = key
        packed = entry["weight"]
        scales = entry["weight_scale"]

        from src.quant_utils import dequantize_mxfp4_to_dtype

        deq = dequantize_mxfp4_to_dtype(packed, scales, out_dtype=self.dtype)
        self._write_fused_expert_slice(layer_prefix, expert_idx, proj_kind, deq)

        # Free intermediate tensors aggressively; expert weights are large.
        del self._mxfp4_pending[key]
        del packed, scales, deq

    def _ensure_fused_expert_param_materialized(self, param_name):
        param_name = param_name.replace(".language_model", "")
        params = dict(self.model.named_parameters())
        if param_name not in params:
            return
        p = params[param_name]
        if p.device.type != "meta":
            return
        zeros = torch.zeros(p.shape, dtype=self.dtype, device=self.device)
        set_module_tensor_to_device(self.model, param_name, self.device, value=zeros, dtype=self.dtype)

    def _fused_experts_module_path(self, layer_prefix):
        """Return the dotted submodule path of the fused experts container.

        Default convention is Qwen-MoE: ``{layer_prefix}.mlp.experts``.
        Subclasses (e.g. DeepSeek-V4 with ``ffn``) can override.
        """
        return f"{layer_prefix}.mlp.experts"

    def _ckpt_to_model_name(self, ckpt_name: str) -> str:
        """Map a safetensors checkpoint key to the model's internal param name.

        The default implementation only strips ``.language_model`` prefixes.
        Subclasses should override when the checkpoint uses a different naming
        convention from the HF model (e.g. DeepSeek-V4 uses ``layers.X`` in
        the checkpoint but the model has ``model.layers.X``).
        """
        return ckpt_name.replace(".language_model", "")

    def _write_fused_expert_slice(self, layer_prefix, expert_idx, proj_kind, tensor_value):
        if self.fused_expert_intermediate_size is None:
            raise RuntimeError("fused_expert_intermediate_size must be set when fused_experts=True")
        intermediate_dim = self.fused_expert_intermediate_size
        # ``layer_prefix`` comes from checkpoint naming (e.g. ``layers.42``).
        # Map to model-internal naming (e.g. ``model.layers.42``) so that
        # ``get_submodule`` can find it.
        experts_path_ckpt = self._fused_experts_module_path(layer_prefix)
        experts_path = self._ckpt_to_model_name(experts_path_ckpt)
        gate_up_name = f"{experts_path}.gate_up_proj"
        down_name = f"{experts_path}.down_proj"
        self._ensure_fused_expert_param_materialized(gate_up_name)
        self._ensure_fused_expert_param_materialized(down_name)
        experts_mod = self.model.get_submodule(experts_path.replace(".language_model", ""))
        t = tensor_value.to(device=self.device, dtype=experts_mod.gate_up_proj.dtype)
        with torch.no_grad():
            if proj_kind == "gate_proj":
                experts_mod.gate_up_proj.data[expert_idx, :intermediate_dim, :].copy_(t)
            elif proj_kind == "up_proj":
                experts_mod.gate_up_proj.data[expert_idx, intermediate_dim : 2 * intermediate_dim, :].copy_(t)
            else:
                experts_mod.down_proj.data[expert_idx, :, :].copy_(t)

    def resolve_model_path(self, model_name):
        p = Path(model_name)

        if p.exists():
            if p.is_file():
                return str(p.parent.resolve())
            return str(p.resolve())

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=model_name, ignore_patterns=["*.bin", "*.pth"])
            return str(Path(local_dir).resolve())
        except Exception as e:
            raise FileNotFoundError(
                f"Could not resolve '{model_name}' as a local path or download it from the Hugging Face Hub. "
                f"Original error: {e}"
            )
        
    def dict_to_ns(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self.dict_to_ns(x) for x in d]
        else:
            return d
        
    @staticmethod
    def _load_safetensor_index(ckpt_path):
        index_json = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.exists(index_json):
            with open(index_json, "r") as f:
                idx = json.load(f)
            return idx["weight_map"], None

        single = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(single):
            return None, single

        raise FileNotFoundError(f"No safetensors found under {ckpt_path}")

    def _materialize_to_device(self, name_shard_pairs):
        params = dict(self.model.named_parameters())
        for n, _ in name_shard_pairs:
            if n not in params:
                continue
            if params[n].device.type == "meta":
                t = torch.randn(params[n].shape, dtype=self.dtype, device=self.device) * self.meta_init_std
                set_module_tensor_to_device(self.model, n, self.device, value=t, dtype=self.dtype)

    def _names_from_ckpt(self, prefixes):
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        def wanted(name):
            for p in prefixes:
                if name == p or name.startswith(p + "."):
                    return True
            return False

        if self.dummy:
            return [(n, None) for n, _ in self.model.named_parameters() if wanted(n)]

        pairs = []
        if self._name_to_shard is None:
            names = safe_load_file(self._single_shard, device="cpu").keys()
            for n in names:
                if wanted(n):
                    pairs.append((n, self._single_shard))
        else:
            for n, shard in self._name_to_shard.items():
                if wanted(n):
                    pairs.append((n, os.path.join(self.ckpt_path, shard)))
        return pairs

    def _set_tensors(self, name_shard_pairs):
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return

        # Clear per-layer pending buffers from any previous layer.
        self._mxfp4_pending.clear()
        self._fp8_pending.clear()

        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)

        for shard, names in by_shard.items():
            with safe_open(shard, framework="pt", device=str(self.device)) as f:
                for ckpt_name in names:
                    if ckpt_name.endswith("inv_freq"):
                        continue
                    model_name = self._ckpt_to_model_name(ckpt_name)

                    # ----- MXFP4 online dequant path (MoE experts) ----- #
                    if self.mxfp4_experts and self.fused_experts:
                        mx = self._parse_mxfp4_expert_key(ckpt_name)
                        if mx is not None:
                            layer_prefix, eid, proj, kind = mx
                            t = f.get_tensor(ckpt_name)
                            key = (layer_prefix, eid, proj)
                            self._mxfp4_pending.setdefault(key, {})[kind] = t
                            self._try_dequant_mxfp4_pending(key)
                            continue

                    # ----- FP8 dense online dequant path ----- #
                    if self.fp8_dense:
                        # Buffer ``.scale`` keys (companion to a ``.weight``).
                        if model_name.endswith(".scale"):
                            self._fp8_pending[model_name] = f.get_tensor(ckpt_name)
                            continue

                        # For ``.weight`` keys, check if the tensor is FP8
                        # and if a paired scale exists.
                        if model_name.endswith(".weight"):
                            t = f.get_tensor(ckpt_name)
                            if t.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                                scale_key = model_name.rsplit(".weight", 1)[0] + ".scale"
                                scale_t = self._fp8_pending.pop(scale_key, None)
                                if scale_t is not None:
                                    from src.quant_utils import dequantize_fp8_to_dtype
                                    t = dequantize_fp8_to_dtype(t, scale_t, out_dtype=self.dtype)
                                else:
                                    # No scale found – best-effort: upcast directly.
                                    t = t.to(self.dtype)
                                set_module_tensor_to_device(
                                    self.model, model_name, self.device,
                                    value=t, dtype=self.dtype,
                                )
                                continue
                            # Not FP8 → fall through to normal loading.

                    t = f.get_tensor(ckpt_name)
                    if self.fused_experts:
                        parsed = self._parse_fused_expert_weight_param_name(model_name)
                        if parsed:
                            layer_prefix, eid, proj = parsed
                            self._write_fused_expert_slice(layer_prefix, eid, proj, t)
                            continue
                    set_module_tensor_to_device(self.model, model_name, self.device, value=t, dtype=t.dtype)

        # Sanity: warn if any pending pairs are still incomplete.
        if self.mxfp4_experts and self._mxfp4_pending:
            stale = list(self._mxfp4_pending.keys())[:4]
            print(
                f"[BaseModelWrapper] WARNING: {len(self._mxfp4_pending)} MXFP4 expert "
                f"tensor(s) had no matching weight/scale partner. Examples: {stale}"
            )
            self._mxfp4_pending.clear()

        if self.fp8_dense and self._fp8_pending:
            stale = list(self._fp8_pending.keys())[:4]
            print(
                f"[BaseModelWrapper] WARNING: {len(self._fp8_pending)} FP8 scale "
                f"tensor(s) had no matching weight partner. Examples: {stale}"
            )
            self._fp8_pending.clear()

    def move_layer_to_gpu(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        pairs = self._names_from_ckpt(prefixes["non_mlp"] + prefixes["mlp"])
        self._set_tensors(pairs)

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        for n in names:
            if n.endswith("inv_freq"):
                continue
            n = n.replace(".language_model", "")
            set_module_tensor_to_device(self.model, n, "meta")
        torch.cuda.empty_cache()

    def offload_to_meta(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        non_pairs = self._names_from_ckpt(prefixes["non_mlp"])
        self._offload_names_to_meta(non_pairs)
        if self.fused_experts and prefixes.get("mlp_offload_params"):
            for pname in prefixes["mlp_offload_params"]:
                n = pname.replace(".language_model", "")
                set_module_tensor_to_device(self.model, n, "meta")
        else:
            pairs = self._names_from_ckpt(prefixes["mlp"])
            self._offload_names_to_meta(pairs)

    @torch.no_grad()
    def get_inputs(self, data_dict, data_loader):
        current_layer = self.get_layer_module(self.current_layer_idx)
        cache = {'index': 0}
        def store_input_hook(_, args, kwargs):
            start = cache['index'] * self.batch_size
            end = min(start + self.batch_size, data_dict['input'].shape[0])
            if isinstance(args, tuple):
                args = args[0]
            data_dict['input'][start:end] = args
            cache['index'] += 1
            for k, v in kwargs.items():
                if k == "attention_mask":
                    if v is not None:
                        self._attention_mask_1 = v.detach().cpu()
                elif k not in ("hidden_states", "past_key_values", "past_key_value"):
                    self.kwargs[k] = v
            raise ValueError
            
        total_batches = len(data_loader)
        handle = current_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for batch_idx, batch in enumerate(data_loader):
            try:
                self.model(batch.to(self.device))
            except ValueError:
                pass
            if self.rank == 0 and (batch_idx + 1) % max(1, total_batches // self.batch_report_divisor) == 0:
                print(f"  get_inputs: {batch_idx + 1}/{total_batches} batches", flush=True)
        handle.remove()

        self.kwargs["position_embeddings"] = (self.kwargs["position_embeddings"][0][0].unsqueeze(0), self.kwargs["position_embeddings"][1][0].unsqueeze(0))

    def _build_layer_inputs(self, batch_size):
        """Build kwargs dict for a layer forward, expanding attention_mask to batch_size."""
        inputs = dict(self.kwargs)
        inputs["attention_mask"] = None
        return inputs

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        for batch_idx in range((data_all['input'].size(0) + self.batch_size - 1) // self.batch_size):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, data_all['input'].shape[0])
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)

            additional_layer_inputs = self._build_layer_inputs(x.shape[0])
            out = current_layer(x, **additional_layer_inputs)
            if isinstance(out, tuple):
                out = out[0]
            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_input_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_input(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
    
    @torch.no_grad()
    def get_mlp_output_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_output(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
        
    @torch.no_grad()
    def get_loss(self, layer_inputs, layer_outputs):
        new_outputs = self.get_mlp_output(layer_inputs)
        val_loss = self.loss_fn(layer_outputs, new_outputs)
        return layer_inputs.detach().cpu(), new_outputs.detach().cpu(), val_loss.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        if logging is not None:
            logging = logging.logger
        current_layer = self.get_layer_module(self.current_layer_idx)
        self.groupsize = config.quantization.groupsize
        subset = {}
        
        if is_attn:
            self_attn = current_layer.self_attn
            base_prefix = f"{self.get_current_layer()}.self_attn"
            for name, module in self_attn.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        else:
            mlp = current_layer.mlp
            base_prefix = f"{self.get_current_layer()}.mlp"
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        
        skip_gptq = getattr(config.gptq, 'skip', False)

        if skip_gptq:
            for name in subset:
                Q, scales = rtn_quantize(subset[name], config, self.device, self.dtype)
                if "q_proj" in name or "k_proj" in name:
                    self.update_quantized_weights(name, (Q, scales))
                    if self.world_size > 1:
                        layer = self._get_layer_by_name(name)
                        dist.broadcast(layer.weight.data, src=0)
                else:
                    trainer.setup_layer_training(name, Q, scales)
            return

        gpts = {}
        for name in subset:
            gpts[name] = GPTQ(subset[name], name, config, self.device, self.dtype)
            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    config.gptq.wbits, perchannel=True, sym=config.gptq.sym, mse=True, trits=config.gptq.trits
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        calib_total = gpt_all['input'].shape[0]
        calib_report_interval = max(1, calib_total // self.calib_report_divisor)
        calib_start = time.time()
        for j in range(calib_total):
            inp = gpt_all['input'][j].unsqueeze(0).to(self.device)
            additional_layer_inputs = self._build_layer_inputs(1)
            _ = current_layer(inp, **additional_layer_inputs)
            if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                report_gptq_calib(j + 1, calib_total, time.time() - calib_start)
        for h in handles:
            h.remove()

        gptq_losses = []
        linear_names = list(gpts.keys())
        linear_start = time.time()
        for li, name in enumerate(linear_names):
            if self.rank == 0:
                report_gptq_linear(li + 1, len(linear_names), name,
                                   time.time() - linear_start)
            if self.world_size > 1:
                gpts[name].sync_H(self.world_size)
            Q, scales = gpts[name].fasterquant(
                logging, percdamp=config.gptq.percdamp, blocksize=config.gptq.blocksize, groupsize=config.gptq.groupsize, static_groups=config.gptq.static_groups, prunen=config.gptq.prunen, prunem=config.gptq.prunem
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if "q_proj" in name or "k_proj" in name:
                self.update_quantized_weights(name, (Q, scales))
                if self.world_size > 1:
                    layer = self._get_layer_by_name(name.replace(".language_model", ""))
                    dist.broadcast(layer.weight.data, src=0)
            else:
                trainer.setup_layer_training(name, Q, scales)
            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

    def calculate_mse(self, batch, quantized_weights, self_attn, validation=False, accumulation_steps=1):
        with torch.no_grad():
            out_fp = self.forward_with_quantized(batch, None, self_attn)
        out_q = self.forward_with_quantized(batch, quantized_weights, self_attn)

        mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (mse / accumulation_steps).backward()

        return mse.item()

    def forward_with_quantized(self, batch, quantized_weights, self_attn):
        class LinearWeightHook:
            def __init__(self, module, quantized_weight):
                self.module = module
                self.quantized_weight = quantized_weight
                self.original_forward = module.forward

            def __enter__(self):
                def new_forward(module_self, x):
                    return torch.nn.functional.linear(x, self.quantized_weight, self.module.bias)
                self.module.forward = new_forward.__get__(self.module, torch.nn.Linear)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.module.forward = self.original_forward

        current_layer = self.get_layer_module(self.current_layer_idx)
        hooks = []
        try:
            if quantized_weights is not None:
                if self_attn:
                    layer = current_layer.self_attn
                else:
                    layer = current_layer.mlp

                for name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if self_attn:
                            key = f"{self.get_current_layer()}.self_attn.{name}"
                        else: 
                            key = f"{self.get_current_layer()}.mlp.{name}"
                        if key in quantized_weights:
                            hooks.append(LinearWeightHook(module, quantized_weights[key]))
            else:
                for name, module in current_layer.self_attn.named_modules():
                    key = f"{self.get_current_layer()}.self_attn.{name}"
                    if key in self.temp_weights:
                        hooks.append(LinearWeightHook(module, self.temp_weights[key]))

            with ExitStack() as stack:
                for h in hooks: stack.enter_context(h)
                if self_attn:
                    output = self.get_mlp_input(batch)
                else:
                    additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
                    output = current_layer(batch, **additional_layer_inputs)
            return output
        finally:
            hooks.clear()

    def find_layers(self, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(self.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
        
    def _get_layer_by_name(self, layer_name):
        return self.model.get_submodule(layer_name)
        
    def update_quantized_weights(self, layer_name, quantized_weights):
        layer = self._get_layer_by_name(layer_name.replace(".language_model", ""))
        self.temp_weights[layer_name] = layer.weight.data
        self.temp_weights[f"{layer_name}.scale"] = quantized_weights[1]
        with torch.no_grad():
            layer.weight.data = quantized_weights[0].to(self.device).to(layer.weight.data.dtype)
        
    def get_current_layer(self):
        return f"{self.layer_prefix}.{self.current_layer_idx}"
        
    def move_to_next_layer(self):
        self.current_layer_idx += 1
        if self.current_layer_idx < self.num_layers:
            return self.get_current_layer()
        return None
    
    def normalize_dtype(self, x):
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, str):
            try:
                return getattr(torch, x)
            except AttributeError:
                raise ValueError(f"Unknown dtype string: {x}")
        raise TypeError(f"Expected torch.dtype or str, got {type(x)}")
    
    def _compress_weight(self, weight, scale):
        # Adapter for the post-0.15 compressed_tensors API: PackedQuantizationCompressor
        # is now state-dict / scheme based (compress(state_dict, scheme)) instead of
        # the old per-tensor compress_weight(weight=, scale=, quantization_args=).
        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
        scheme = self.quantization_config.config_groups.group_0
        return self.compressor.compress(
            {"weight": weight, "weight_scale": scale},
            scheme,
        )

    def save_to_disc(self, pfx, pairs):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        for name, pair in pairs.items():
            scale = pair[1][0] if isinstance(pair[1], tuple) else pair[1]
            scale_cpu = scale.detach().cpu()
            compressed = self._compress_weight(pair[0].detach().cpu(), scale_cpu)
            base = f"{pfx}.{name}"
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            to_save[base + ".weight_scale"]  = scale_cpu
            to_save[base + ".weight_shape"]  = compressed["weight_shape"]
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        self_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        for name in self_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
            if isinstance(self.temp_weights[f"{base}.scale"], tuple):
                self.temp_weights[f"{base}.scale"] = self.temp_weights[f"{base}.scale"][0]
            scale_cpu = self.temp_weights[f"{base}.scale"].detach().cpu()
            compressed = self._compress_weight(
                module.weight.data.detach().cpu(),
                scale_cpu,
            )
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            to_save[base + ".weight_scale"]  = scale_cpu
            to_save[base + ".weight_shape"]  = compressed["weight_shape"]
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_prefixes_to_disc(self, prefixes, exclude=[]):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if "self_attn" in exclude:
            current_layer = self.get_layer_module(self.current_layer_idx)
            if getattr(current_layer, "layer_type", None) == "linear_attn":
                exclude.append("linear_attn")
        prefixes = [p for p in prefixes if not any(x in p for x in exclude)]
        for pfx in prefixes:
            module = self.model.get_submodule(pfx.replace(".language_model", ""))
            sd = module.state_dict(keep_vars=True)
            to_save = {}
            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()
            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    def load_from_disc(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        files = {}
        for item in ("non_mlp", "mlp"):
            if item not in prefixes:
                continue
            for p in prefixes[item]:
                base = p.replace('.', '_')
                st = os.path.join(self.save_dir, f"{base}.safetensors")
                files[p] = st

        for p, path in files.items():
            tensors = safe_load_file(path, device="cpu")
            for name in tensors.keys():
                if name.endswith(".weight_shape") or name.endswith(".weight_scale") or name.endswith("inv_freq"):
                    continue
                if name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    compressed_data = {
                        "weight_packed": tensors[f"{base}.weight_packed"], 
                        "weight_scale": tensors[f"{base}.weight_scale"],
                        "weight_shape": tensors[f"{base}.weight_shape"]
                    }
                    decompressed = self.compressor.decompress(
                        compressed_data,
                        self.quantization_config.config_groups.group_0,
                    )
                    W_deq = decompressed["weight"]

                    weight_key = f"{base}.weight".replace(".language_model", "")
                    if self.fused_experts:
                        mod_base = base.replace(".language_model", "")
                        parsed = self._parse_fused_expert_module_base(mod_base)
                        if parsed:
                            lp, eid, proj = parsed
                            self._write_fused_expert_slice(lp, eid, proj, W_deq)
                            continue
                    set_module_tensor_to_device(self.model, weight_key, self.device, value=W_deq, dtype=self.dtype)
                    continue
                model_key = name.replace(".language_model", "")
                if self.fused_experts:
                    parsed = self._parse_fused_expert_weight_param_name(model_key)
                    if parsed:
                        lp, eid, proj = parsed
                        self._write_fused_expert_slice(lp, eid, proj, tensors[name])
                        continue
                set_module_tensor_to_device(self.model, model_key, self.device, value=tensors[name], dtype=self.dtype)
        current_layer = self.get_layer_module(self.current_layer_idx)

    @abstractmethod
    def get_mlp_input(self, layer_input):
        pass

    @abstractmethod
    def get_mlp_output(self, mlp_input_batch):
        pass
    
    @abstractmethod
    def get_layer_module(self, idx):
        pass

    @abstractmethod
    def move_embed_to(self, device):
        pass

    @abstractmethod
    def move_output_heads_to(self, device):
        pass

    @abstractmethod
    def _layer_prefixes(self, layer_name):
        pass

    @abstractmethod
    def ppl_evaluation(self, read_from_disk=-1):
        pass
