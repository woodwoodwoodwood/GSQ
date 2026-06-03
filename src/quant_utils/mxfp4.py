"""Pure PyTorch MXFP4 (E2M1 packed nibbles + UE8M0 block scales) dequantization.

Used to upcast on-the-fly the MXFP4 expert weights of DeepSeek-V4-Flash
(``uint8`` packed weights + per-32-element ``uint8`` UE8M0 scales) to
``bfloat16`` while loading the safetensors checkpoint, so that GSQ can run
without a separate offline reconstruction pass.

Layout convention (matches the DeepSeek-V4 / NVFP4 community convention):

* ``packed`` :  ``uint8`` tensor with shape ``[..., K // 2]`` where the low
  nibble of each byte stores the *first* of the two FP4 codes and the high
  nibble stores the *second*.
* ``scales`` :  ``uint8`` tensor with shape ``[..., K // 32]`` interpreted as
  UE8M0 (i.e. an unsigned 8-bit exponent, value = ``2 ** (e - 127)``).
"""

from __future__ import annotations

import torch

# Sign-magnitude E2M1 lookup table:
#   bits 0..2  -> magnitude
#   bit  3     -> sign
# The 16 entries below cover ``(sign << 3) | magnitude`` indexing.
_E2M1_LUT_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@torch.no_grad()
def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a ``uint8`` E2M1 nibble tensor to ``float32``.

    The low nibble is decoded *before* the high nibble, doubling the size of
    the last dimension.
    """
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)

    lut = torch.tensor(_E2M1_LUT_VALUES, dtype=torch.float32, device=packed.device)
    p = packed.to(torch.int64)

    lo = lut[p & 0x0F]
    hi = lut[(p >> 4) & 0x0F]

    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@torch.no_grad()
def ue8m0_to_float32(scales: torch.Tensor) -> torch.Tensor:
    """Decode a ``uint8`` UE8M0 exponent tensor to ``float32`` (= ``2**(e-127)``)."""
    if scales.dtype != torch.uint8:
        scales = scales.view(torch.uint8)
    e = scales.to(torch.int32)
    # IEEE-754 trick: place the exponent into a float32 directly.
    return ((e << 23).view(torch.float32))


@torch.no_grad()
def dequantize_mxfp4_to_dtype(
    packed: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize an MXFP4 packed tensor with UE8M0 block scales to ``out_dtype``.

    ``packed.shape[-1]`` must equal ``2 * scales.shape[-1] * (block_size // 32)``.
    The leading dimensions of ``packed`` and ``scales`` must broadcast.
    """
    if packed.device != scales.device:
        scales = scales.to(packed.device)

    unpacked = unpack_fp4_e2m1(packed)  # float32, last_dim *= 2

    inferred_block = unpacked.shape[-1] // scales.shape[-1]
    if block_size != inferred_block:
        # Trust the data layout over the user-supplied default.
        block_size = inferred_block

    decoded_scales = ue8m0_to_float32(scales)
    expanded = decoded_scales.repeat_interleave(block_size, dim=-1)

    out = unpacked * expanded
    return out.to(out_dtype)
