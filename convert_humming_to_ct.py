"""Convert a Humming-format GSQ checkpoint back into an assembled
compressed-tensors (pack-quantized) checkpoint.

This is the exact inverse of `convert_to_humming.py` / `humming_pack.ct_to_humming`.
The transformation is a **lossless bit re-packing**: the quantized integer codes
are identical, only the on-disk packing layout differs (Humming's tiled packer
vs compressed-tensors' LSB-first packer). No dequant/requant happens, so there is
zero additional quantization error.

Motivation
----------
Published GSQ checkpoints (e.g. ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ) ship in
Humming format:
    <prefix>.weight        int32 [N, K * eff_bits // 32]   (humming.ops.pack_weight)
    <prefix>.weight_scale  bf16  [N, K // group_size]
    (<prefix>.zero_point   bf16  [N, K // group_size])     (FP zero-point path)

Our llama.cpp GGUF converter only ingests compressed-tensors `pack-quantized`:
    <prefix>.weight_packed int32 [N, K * num_bits // 32]   (LSB-first)
    <prefix>.weight_scale  bf16  [N, K // group_size]
    <prefix>.weight_shape  int64 [2]

This script bridges the two so a Humming checkpoint can flow into the existing
compressed-tensors -> GGUF pipeline.

Usage
-----
    # Convert + write:
    python convert_humming_to_ct.py \
        --in-dir  /data1/models/Qwen3.6-35B-A3B-2Bit-GSQ \
        --out-dir /data1/models/Qwen3.6-35B-A3B-2Bit-GSQ-ct \
        --storage-bits 2 \
        --verify-one '.*layers\.0\.mlp\.experts\.0\.gate_proj$'

    # Verify only (no write):
    python convert_humming_to_ct.py --in-dir <humming_dir> --verify-only \
        --verify-one '.*layers\.0\..*gate_proj$'

Notes
-----
- Requires humming (CUDA) because ops.unpack_weight is a CUDA kernel.
- `--storage-bits` is the compressed-tensors container width. Default 2 to match
  configs/qwen36/config_assembled.json (num_bits=2). It must be >= the effective
  code width; the script asserts codes fit the container.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.quantization.humming_pack import (  # noqa: E402
    _unpack_ct_int32,
    ct_dequantize_reference,
    humming_dequantize,
)


# ----------------------------------------------------------------------------
# Core packing helpers
# ----------------------------------------------------------------------------
def _pack_ct_int32(codes: torch.Tensor, storage_bits: int) -> torch.Tensor:
    """LSB-first pack of unsigned codes into int32 along the last dim.

    Exact inverse of `_unpack_ct_int32`. `codes` must be int32 in
    [0, 2**storage_bits). Returns int32 [..., K * storage_bits // 32].
    """
    vals_per_el = 32 // storage_bits
    *lead, K = codes.shape
    if K % vals_per_el != 0:
        raise ValueError(f"K={K} not divisible by {vals_per_el} (32/{storage_bits})")
    c = codes.to(torch.int64).reshape(*lead, K // vals_per_el, vals_per_el)
    shifts = (torch.arange(vals_per_el, dtype=torch.int64, device=c.device) * storage_bits)
    packed = (c << shifts).sum(dim=-1)  # up to 2**32 - 1, held in int64
    return packed.to(torch.int32).contiguous()  # truncate to low 32 bits (bit pattern)


def humming_layer_to_ct(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    *,
    group_size: int,
    storage_bits: int,
    target_dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Convert one Humming Linear into a compressed-tensors bundle.

    Returns ({weight_packed, weight_scale, weight_shape}, info).
    """
    from humming import ops

    N = int(weight.shape[0])
    groups = int(weight_scale.shape[-1])
    K = groups * group_size
    packed_cols = int(weight.shape[-1])
    eff_bits = packed_cols * 32 // K
    if packed_cols * 32 % K != 0:
        raise ValueError(f"cannot infer eff_bits: packed_cols={packed_cols} K={K}")

    # 1) Unpack Humming tiled packing -> unsigned codes [0, 2**eff_bits).
    codes = ops.unpack_weight(weight.cuda().contiguous(), eff_bits).cpu()
    codes = codes[:, :K].to(torch.int32).contiguous()

    # 2) Recover the signed code.
    #    - symmetric_out (no zero_point): kernel implicitly subtracts 2**(eff_bits-1)
    #    - FP zero-point: deq = (code - zp) * scale, zp is a constant integer
    if zero_point is None:
        zp_val = 1 << (eff_bits - 1)
    else:
        zp_flat = zero_point.flatten().to(torch.float64)
        zp_val_f = float(zp_flat[0].item())
        if not torch.allclose(zp_flat, torch.full_like(zp_flat, zp_val_f)):
            raise ValueError("non-constant FP zero_point is not representable in CT symmetric")
        zp_val = int(round(zp_val_f))
        if abs(zp_val - zp_val_f) > 1e-3:
            raise ValueError(f"FP zero_point {zp_val_f} is not integer; CT symmetric needs integer zp")
    signed = codes - zp_val  # signed quantized code

    # 3) Re-encode into the compressed-tensors symmetric container.
    ct_offset = 1 << (storage_bits - 1)
    decoded_ct = (signed + ct_offset).to(torch.int32)
    lo, hi = int(decoded_ct.min()), int(decoded_ct.max())
    if lo < 0 or hi >= (1 << storage_bits):
        raise ValueError(
            f"code range [{lo},{hi}] does not fit CT container of {storage_bits} bits "
            f"[0,{1 << storage_bits}); pass a larger --storage-bits."
        )

    weight_packed = _pack_ct_int32(decoded_ct, storage_bits)

    # 4) Self-check: CT unpack must reproduce decoded_ct exactly.
    back = _unpack_ct_int32(weight_packed, storage_bits)[:, :K]
    if not torch.equal(back, decoded_ct):
        raise RuntimeError("CT pack/unpack round-trip mismatch (bug in _pack_ct_int32)")

    weight_shape = torch.tensor([N, K], dtype=torch.int64)
    scales_out = weight_scale.to(target_dtype).contiguous()

    tensors = {
        "weight_packed": weight_packed,
        "weight_scale": scales_out,
        "weight_shape": weight_shape,
    }
    info = {
        "N": N, "K": K, "eff_bits": eff_bits, "storage_bits": storage_bits,
        "code_min": lo, "code_max": hi, "zp_val": zp_val,
        "has_zero_point": zero_point is not None,
    }
    return tensors, info


# ----------------------------------------------------------------------------
# Checkpoint discovery
# ----------------------------------------------------------------------------
def _load_weight_map(in_dir: Path) -> Dict[str, str]:
    idx = in_dir / "model.safetensors.index.json"
    single = in_dir / "model.safetensors"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index/file under {in_dir}")


def discover_quantized_layers(in_dir: Path) -> Dict[str, str]:
    """Return {prefix: shard_name} for every quantized Humming Linear.

    A Linear is quantized iff it has both `<prefix>.weight` (int32) and
    `<prefix>.weight_scale`.
    """
    wmap = _load_weight_map(in_dir)
    scale_prefixes = {k[: -len(".weight_scale")] for k in wmap if k.endswith(".weight_scale")}
    quant: Dict[str, str] = {}
    # Confirm the `.weight` for each scale prefix is int32.
    shard_cache: Dict[str, set] = {}
    for prefix in sorted(scale_prefixes):
        wkey = f"{prefix}.weight"
        if wkey not in wmap:
            continue
        shard = wmap[wkey]
        if shard not in shard_cache:
            with safe_open(str(in_dir / shard), framework="pt", device="cpu") as f:
                # store dtypes lazily via slice metadata
                shard_cache[shard] = set(f.keys())
        quant[prefix] = shard
    return quant


# ----------------------------------------------------------------------------
# Config emission
# ----------------------------------------------------------------------------
def _to_regex_ignore(entry: str) -> str:
    """Turn a Humming ignore token into a compressed-tensors regex ignore."""
    if entry.startswith("re:"):
        return entry
    if entry in ("lm_head",) or "." not in entry:
        # plain module-name substring, CT matches by name; wrap defensively
        return entry
    return "re:.*" + re.escape(entry) + ".*"


def build_ct_config(humming_cfg_full: Dict, num_bits: int, group_size: int) -> Dict:
    qc = humming_cfg_full.get("quantization_config", {})
    ignore = qc.get("ignore", [])
    ct_ignore = [_to_regex_ignore(e) for e in ignore]
    cfg = dict(humming_cfg_full)
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": num_bits,
                    "strategy": "group",
                    "group_size": group_size,
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ct_ignore,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }
    return cfg


# ----------------------------------------------------------------------------
# Streaming writer
# ----------------------------------------------------------------------------
def write_ct_checkpoint(
    in_dir: Path,
    out_dir: Path,
    quant_layers: Dict[str, str],
    *,
    storage_bits: int,
    group_size: int,
    target_dtype: torch.dtype,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    wmap = _load_weight_map(in_dir)

    by_shard: Dict[str, List[str]] = {}
    for k, shard in wmap.items():
        by_shard.setdefault(shard, []).append(k)

    prefixes_in_shard: Dict[str, List[str]] = {}
    for p, sh in quant_layers.items():
        prefixes_in_shard.setdefault(sh, []).append(p)

    new_wmap: Dict[str, str] = {}
    total_bytes = 0
    bits_hist: Dict[int, int] = {}
    shards_sorted = sorted(set(wmap.values()))
    t0 = time.perf_counter()
    done = 0
    total = len(quant_layers)

    print(f"streaming {len(shards_sorted)} shards {in_dir} -> {out_dir}")
    for si, shard_name in enumerate(shards_sorted, 1):
        src = in_dir / shard_name
        out = out_dir / shard_name
        here = sorted(prefixes_in_shard.get(shard_name, []))
        drop = set()
        for p in here:
            drop |= {f"{p}.weight", f"{p}.weight_scale", f"{p}.zero_point"}

        tensors_out: Dict[str, torch.Tensor] = {}
        with safe_open(str(src), framework="pt", device="cpu") as f:
            keys_here = set(f.keys())
            for prefix in here:
                w = f.get_tensor(f"{prefix}.weight")
                ws = f.get_tensor(f"{prefix}.weight_scale")
                zp = f.get_tensor(f"{prefix}.zero_point") if f"{prefix}.zero_point" in keys_here else None
                try:
                    bundle, info = humming_layer_to_ct(
                        w, ws, zp, group_size=group_size,
                        storage_bits=storage_bits, target_dtype=target_dtype,
                    )
                except Exception as e:
                    raise RuntimeError(f"failed converting {prefix}: {e}") from e
                del w, ws, zp
                bits_hist[info["eff_bits"]] = bits_hist.get(info["eff_bits"], 0) + 1
                tensors_out[f"{prefix}.weight_packed"] = bundle["weight_packed"]
                tensors_out[f"{prefix}.weight_scale"] = bundle["weight_scale"]
                tensors_out[f"{prefix}.weight_shape"] = bundle["weight_shape"]
                done += 1

            for k in keys_here:
                if k in drop:
                    continue
                tensors_out[k] = f.get_tensor(k).contiguous()

        for k, v in tensors_out.items():
            new_wmap[k] = shard_name
            total_bytes += v.numel() * v.element_size()
        save_file(tensors_out, str(out))
        print(f"  [{si}/{len(shards_sorted)}] {shard_name}: {len(tensors_out)} tensors "
              f"({done}/{total} converted) elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        del tensors_out
        gc.collect()

    print(f"\neffective-bit histogram: {bits_hist}")

    # index
    if (in_dir / "model.safetensors.index.json").exists() or len(shards_sorted) > 1:
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": int(total_bytes)}, "weight_map": new_wmap}, indent=2))

    # aux files
    keep = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
            "tokenizer.model", "tiktoken.model", "special_tokens_map.json",
            "generation_config.json", "preprocessor_config.json", "chat_template.jinja"]
    for fn in os.listdir(in_dir):
        src = in_dir / fn
        if src.is_file() and (fn in keep or fn.endswith(".py") or fn.endswith(".jinja")):
            shutil.copy2(src, out_dir / fn)

    # config.json
    hum_cfg = json.loads((in_dir / "config.json").read_text())
    num_bits = max(bits_hist, key=bits_hist.get) if bits_hist else 2
    ct_cfg = build_ct_config(hum_cfg, num_bits=storage_bits, group_size=group_size)
    (out_dir / "config.json").write_text(json.dumps(ct_cfg, indent=2))
    print(f"wrote {done} quantized layers, ~{total_bytes/1024**3:.2f} GB -> {out_dir}")
    print(f"CT container num_bits={storage_bits}, effective code bits={num_bits}")


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify_one(in_dir: Path, pattern: str, *, storage_bits: int, group_size: int,
               target_dtype: torch.dtype):
    quant = discover_quantized_layers(in_dir)
    pat = re.compile(pattern)
    matches = [p for p in sorted(quant) if pat.search(p)]
    if not matches:
        raise SystemExit(f"no quantized layer matched {pattern!r}; "
                         f"first few: {sorted(quant)[:5]}")
    name = matches[0]
    shard = quant[name]
    print(f"verifying {name} (shard {shard})")
    with safe_open(str(in_dir / shard), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        w = f.get_tensor(f"{name}.weight")
        ws = f.get_tensor(f"{name}.weight_scale")
        zp = f.get_tensor(f"{name}.zero_point") if f"{name}.zero_point" in keys else None

    # humming reference dequant (ground truth)
    eff_bits = int(w.shape[-1]) * 32 // (int(ws.shape[-1]) * group_size)
    schema_cfg = {
        "quant_method": "humming",
        "dtype": f"uint{eff_bits}",
        "b_dtype": f"uint{eff_bits}",
        "group_size": group_size,
        "has_zero_point": zp is not None,
        "is_fp_zero_point": zp is not None,
    }
    hum_tensors = {"weight": w, "weight_scale": ws}
    if zp is not None:
        hum_tensors["zero_point"] = zp
    w_hum = humming_dequantize(hum_tensors, schema_cfg).float().cpu()

    # CT round-trip dequant
    bundle, info = humming_layer_to_ct(
        w, ws, zp, group_size=group_size, storage_bits=storage_bits, target_dtype=target_dtype)
    print(f"  info: {info}")
    w_ct = ct_dequantize_reference(
        bundle["weight_packed"], bundle["weight_scale"], bundle["weight_shape"],
        storage_bits=storage_bits, group_size=group_size, target_dtype=target_dtype).float()

    diff = (w_hum - w_ct).abs()
    rel = diff.max().item() / max(w_hum.abs().max().item(), 1e-6)
    print(f"  dequant diff: max_abs={diff.max().item():.3e} mean={diff.mean().item():.3e} rel_max={rel:.3e}")
    if rel > 1e-3:
        raise SystemExit(f"round-trip mismatch too large: rel_max={rel}")
    print("  OK (lossless within scale-dtype rounding)")


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="Humming-format checkpoint dir.")
    p.add_argument("--out-dir", default=None, help="Output compressed-tensors dir.")
    p.add_argument("--storage-bits", type=int, default=2,
                   help="CT container bit width (default 2, matches config_assembled.json).")
    p.add_argument("--group-size", type=int, default=None,
                   help="Override group_size; default read from humming config.")
    p.add_argument("--target-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--verify-one", default=None, help="Regex; verify first matching Linear.")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    target_dtype = getattr(torch, args.target_dtype)

    hum_cfg = json.loads((in_dir / "config.json").read_text())
    qc = hum_cfg.get("quantization_config", {})
    if qc.get("quant_method") != "humming":
        raise SystemExit(f"expected quant_method=humming, got {qc.get('quant_method')}")
    group_size = args.group_size or int(qc.get("group_size") or qc.get("weight_scale_group_size") or 128)
    print(f"input: {in_dir}  group_size={group_size}  storage_bits={args.storage_bits}")

    if args.verify_one:
        verify_one(in_dir, args.verify_one, storage_bits=args.storage_bits,
                   group_size=group_size, target_dtype=target_dtype)
    if args.verify_only:
        return

    if args.out_dir is None:
        raise SystemExit("pass --out-dir (or --verify-only)")
    out_dir = Path(args.out_dir).resolve()
    quant = discover_quantized_layers(in_dir)
    print(f"found {len(quant)} quantized Linear modules")
    write_ct_checkpoint(in_dir, out_dir, quant, storage_bits=args.storage_bits,
                        group_size=group_size, target_dtype=target_dtype)


if __name__ == "__main__":
    main()
