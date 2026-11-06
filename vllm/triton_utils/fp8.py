# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def needs_software_e4m3fn(device: torch.device) -> bool:
    """Triton supports native E4M3FN conversions on CUDA starting at SM89."""
    return current_platform.is_cuda() and not current_platform.has_device_capability(
        89, device_id=device.index or 0
    )


@triton.jit
def fp8_e4m3fn_to_uint8(x, USE_SOFTWARE: tl.constexpr):
    """Encode float32 as E4M3FN bytes with finite saturation and RNE."""
    if not USE_SOFTWARE:
        return tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    bits = x.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    exponent = (magnitude >> 23).to(tl.int32)
    significand = (magnitude & 0x7FFFFF) | 0x800000

    # Drop 20 mantissa bits for normals, and one more per exponent step
    # below 2^-6. At 2^-10 (exponent 117) the tie rounds to zero.
    shift = 20 + tl.minimum(tl.maximum(121 - exponent, 0), 4)
    halfway = 1 << (shift - 1)
    rounded = (significand + halfway - 1 + ((significand >> shift) & 1)) >> shift
    code = (tl.maximum(exponent - 121, 0) << 3) + rounded
    code = tl.where(exponent < 117, 0, tl.minimum(code, 0x7E))
    code = tl.where(magnitude > 0x7F800000, 0x7F, code)
    return (code | ((bits >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def fp8_e4m3fn_from_uint8(x, USE_SOFTWARE: tl.constexpr):
    """Decode E4M3FN bytes to float32, preserving subnormals and signed zero."""
    if not USE_SOFTWARE:
        return x.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    bits = x.to(tl.uint16)
    magnitude = bits & 0x7F
    sign = bits & 0x80
    # The fp16 bit pattern reads as E4M3FN * 2^-8 (bias 15 versus 7).
    half_bits = ((sign << 8) | (magnitude << 7)).to(tl.uint16)
    value = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    nan = ((sign.to(tl.uint32) << 24) | 0x7FC00000).to(tl.float32, bitcast=True)
    return tl.where(magnitude == 0x7F, nan, value)
