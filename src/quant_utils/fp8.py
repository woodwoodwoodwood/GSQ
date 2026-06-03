"""Pure PyTorch FP8 (E4M3 weight + UE8M0 block scales) dequantization.

Used to upcast the FP8 dense linear weights of DeepSeek-V4-Flash
(``float8_e4m3fn`` weights + per-block ``float8_e8m0fnu`` / ``uint8`` UE8M0
scales) to ``bfloat16`` while loading the safetensors checkpoint, so that GSQ
can run without a separate offline reconstruction pass.

Block size convention (from the HF config ``weight_block_size`` field):

* Typical value: ``[128, 128]``, meaning one scale per 128×128 weight block.
* The scale tensor shape is ``[out_features // block_r, in_features // block_c]``.
* For 1D per-channel scales the shape is ``[out_features]`` (or scalar for
  per-tensor).

We auto-detect the layout from the relative shapes of the weight and scale
tensors, so no explicit block-size argument is required.
"""

from __future__ import annotations

import torch

from .mxfp4 import ue8m0_to_float32


@torch.no_grad()
def dequantize_fp8_to_dtype(
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 E4M3 weight tensor with UE8M0 block scales.

    Parameters
    ----------
    weight : torch.Tensor
        FP8 E4M3 weight, typically ``float8_e4m3fn`` dtype.
    scale : torch.Tensor
        Block scales in UE8M0 format. May be ``uint8``, ``float8_e8m0fnu``,
        or already decoded ``float32``.
    out_dtype : torch.dtype
        Target dtype for the dequantized weight.

    Returns
    -------
    torch.Tensor
        Dequantized weight in ``out_dtype``.
    """
    # Decode UE8M0 scales to float32 regardless of input dtype.
    if scale.dtype == torch.float8_e8m0fnu:
        # Preserve raw exponent bytes; float8_e8m0fnu stores 2^(e-127)
        # but copy_/view would do a numeric conversion.
        scale = scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        scale_float = ue8m0_to_float32(scale)
    else:
        scale_float = scale.float()

    # Convert weight to float32 for the multiplication.
    w_float = weight.float()

    # Auto-detect scale layout and expand to match weight shape.
    scale_float = _expand_scale_to_weight(scale_float, w_float.shape)

    return (w_float * scale_float).to(out_dtype)


def _expand_scale_to_weight(
    scale: torch.Tensor,
    weight_shape: tuple[int, ...],
) -> torch.Tensor:
    """Expand a (possibly sub-channel-block) scale tensor to match the full
    weight shape so that element-wise multiplication works.

    Supported layouts (auto-detected from ``scale.shape`` vs ``weight_shape``):
    * **Per-tensor**  – ``scale`` is scalar or shape ``(1, 1)``.
    * **Per-channel** – ``scale`` has shape ``(out_features,)`` or
      ``(out_features, 1)``.
    * **2-D block**   – ``scale`` has shape
      ``(out_features // block_r, in_features // block_c)``.
    """
    if scale.numel() == 1:
        # Per-tensor: just multiply everything by the same scale.
        return scale.reshape(1, 1)

    if scale.ndim == 1:
        # Per-output-channel: shape [out_features] → [out_features, 1]
        return scale.unsqueeze(-1)

    if scale.ndim == 2 and len(weight_shape) >= 2:
        out_f, in_f = weight_shape[-2], weight_shape[-1]
        s_out, s_in = scale.shape
        if s_out == out_f and s_in == in_f:
            # Already full-shape.
            return scale
        # 2-D block quantization: repeat each scale over its block.
        block_r = out_f // s_out
        block_c = in_f // s_in
        return scale.unsqueeze(-1).unsqueeze(-1).expand(
            s_out, block_r, s_in, block_c
        ).reshape(out_f, in_f)

    # Fallback: try to broadcast.
    return scale
