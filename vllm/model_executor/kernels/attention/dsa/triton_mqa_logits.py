# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 paged MQA logits using BF16 tensor cores on SM80."""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8 import fp8_e4m3fn_from_uint8


@triton.jit(do_not_specialize=["width", "table_row_stride"])
def _paged_mqa_logits(
    Q,
    K,
    Weights,
    Table,
    Lengths,
    Starts,
    KeyToRequest,
    Logits,
    width,
    table_row_stride,
    table_col_stride: tl.constexpr,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    weight_row_stride: tl.constexpr,
    weight_head_stride: tl.constexpr,
    page_size: tl.constexpr,
    page_stride: tl.constexpr,
    num_heads: tl.constexpr,
    head_tile: tl.constexpr,
    key_tile: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(Lengths + row)
    if Starts is not None:
        # A prefill chunk's row starts index the packed key space, but the
        # page table and output columns use request-local coordinates.
        start = tl.load(Starts + row)
        request = tl.load(KeyToRequest + start, length > 0, 0).to(tl.int64)
    else:
        request = row

    h = tl.arange(0, head_tile)
    d = tl.arange(0, 128)
    q_bytes = tl.load(
        Q + row * q_row_stride + h[:, None] * q_head_stride + d[None, :],
        h[:, None] < num_heads,
        0,
    )
    q = fp8_e4m3fn_from_uint8(q_bytes, True).to(tl.bfloat16)
    weight = tl.load(
        Weights + row * weight_row_stride + h * weight_head_stride, h < num_heads, 0
    )
    for tile in range(tl.program_id(1), tl.cdiv(width, key_tile), tl.num_programs(1)):
        cols = tile * key_tile + tl.arange(0, key_tile)
        valid = (cols < width) & (cols < length)
        page_id = tl.load(
            Table + request * table_row_stride + (cols // page_size) * table_col_stride,
            valid,
            0,
        ).to(tl.int64)
        page = K + page_id * page_stride
        offset = cols % page_size
        key_bytes = tl.load(
            page[None, :] + offset[None, :] * 128 + d[:, None], valid[None, :], 0
        )
        key = fp8_e4m3fn_from_uint8(key_bytes, True).to(tl.bfloat16)
        scales = tl.load(
            (page + page_size * 128).to(tl.pointer_type(tl.float32)) + offset,
            valid,
            0,
        )
        per_head = tl.maximum(tl.dot(q, key), 0.0)
        scores = tl.sum(per_head * weight[:, None], 0) * scales
        tl.store(
            Logits + row * width + cols,
            tl.where(valid, scores, -float("inf")),
            cols < width,
        )


def paged_mqa_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
    starts: torch.Tensor | None = None,
    key_to_request: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score each query over its own causal, request-local FP8 K columns."""
    rows, num_heads, head_dim = q.shape
    assert q.dtype == torch.float8_e4m3fn and head_dim == 128 and q.stride(2) == 1
    assert cache.dtype == torch.uint8 and cache.shape[2] == head_dim + 4
    assert lengths.shape == (rows,) and lengths.is_contiguous()
    assert (starts is None) == (key_to_request is None)
    if starts is not None:
        assert starts.shape == (rows,) and starts.is_contiguous()
    logits = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if not rows or not width:
        return logits
    key_tile = 32
    _paged_mqa_logits[(rows, min(triton.cdiv(width, key_tile), 512))](
        q.view(torch.uint8),
        cache,
        weights,
        block_table,
        lengths,
        starts,
        key_to_request,
        logits,
        width,
        *block_table.stride(),
        q.stride(0),
        q.stride(1),
        *weights.stride(),
        cache.shape[1],
        cache.stride(0),
        num_heads,
        max(16, triton.next_power_of_2(num_heads)),
        key_tile,
        num_warps=4,
    )
    return logits
