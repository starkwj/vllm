# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v41.common.ops.cache_utils import V4_BYTES_PER_TOKEN
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8 import fp8_e4m3fn_from_uint8


@triton.jit
def _load_packed_kv(cache, slot, d, page_size: tl.constexpr, page_stride: tl.constexpr):
    # The V4 record places the scale rows after all data rows in the page.
    nope_dim: tl.constexpr = 448
    data_bytes: tl.constexpr = 576
    scale_bytes: tl.constexpr = 8
    valid = slot >= 0
    page = cache + (slot // page_size).to(tl.int64) * page_stride
    row = slot % page_size
    data = page + row * data_bytes
    scales = page + page_size * data_bytes + row * scale_bytes

    is_nope = d < nope_dim
    raw = tl.load(data[:, None] + d[None, :], valid[:, None] & is_nope[None, :], 0)
    exponent = tl.load(
        scales[:, None] + d[None, :] // 64,
        valid[:, None] & is_nope[None, :],
        127,
    )
    nope = fp8_e4m3fn_from_uint8(raw, True) * tl.exp2(exponent.to(tl.float32) - 127)
    rope = tl.load(
        (data + nope_dim).to(tl.pointer_type(tl.bfloat16))[:, None]
        + (d[None, :] - nope_dim),
        valid[:, None] & ~is_nope[None, :],
        0,
    )
    return tl.where(is_nope[None, :], nope.to(tl.bfloat16), rope)


@triton.jit
def _sparse_mla(
    Q,
    Out,
    Partials,
    Sink,
    Positions,
    SWACache,
    SWAIndices,
    SWALens,
    CompressedCache,
    CompressedIndices,
    CompressedLens,
    q_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    swa_page_size: tl.constexpr,
    swa_page_stride: tl.constexpr,
    swa_index_stride: tl.constexpr,
    compressed_page_size: tl.constexpr,
    compressed_page_stride: tl.constexpr,
    compressed_index_stride: tl.constexpr,
    topk: tl.constexpr,
    compress_ratio: tl.constexpr,
    heads: tl.constexpr,
    sm_scale: tl.constexpr,
    head_tile: tl.constexpr,
    key_tile: tl.constexpr,
    num_splits: tl.constexpr,
    partial_stride: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * head_tile + tl.arange(0, head_tile)
    d = tl.arange(0, 512)
    q = tl.load(
        Q + row * q_row_stride + h[:, None] * 512 + d[None, :], h[:, None] < heads, 0
    )
    # A finite initial maximum handles fully masked tiles without inf-inf.
    neg_large: tl.constexpr = -3.402823466e38
    if num_splits == 1:
        sink = tl.load(Sink + h, h < heads, -float("inf"))
        m = tl.where(sink == -float("inf"), neg_large, sink)
        denom = tl.where(sink == -float("inf"), 0.0, 1.0)
    else:
        m = tl.full((head_tile,), neg_large, tl.float32)
        denom = tl.zeros((head_tile,), tl.float32)
    acc = tl.full((head_tile, 512), 0, tl.float32)

    swa_len = tl.load(SWALens + row)
    swa_tiles = tl.cdiv(swa_len, key_tile)
    compressed_width = 0
    if compress_ratio > 0:
        pos = tl.load(Positions + row)
        has_topk = tl.load(CompressedLens + row) > 0
        compressed_width = tl.where(
            has_topk, tl.minimum(topk, (pos + 1) // compress_ratio), 0
        ).to(tl.int32)
    num_tiles = swa_tiles + tl.cdiv(compressed_width, key_tile)
    tile_start = 0
    tile_end = num_tiles
    if num_splits > 1:
        split = tl.program_id(2)
        tile_start = split * num_tiles // num_splits
        tile_end = (split + 1) * num_tiles // num_splits
    for tile in range(tile_start, tile_end):
        cols = tl.arange(0, key_tile)
        if compress_ratio == 0 or tile < swa_tiles:
            cols += tile * key_tile
            slot = tl.load(
                SWAIndices + row * swa_index_stride + cols, cols < swa_len, -1
            )
            kv = _load_packed_kv(SWACache, slot, d, swa_page_size, swa_page_stride)
        else:
            cols += (tile - swa_tiles) * key_tile
            slot = tl.load(
                CompressedIndices + row * compressed_index_stride + cols,
                cols < compressed_width,
                -1,
            )
            kv = _load_packed_kv(
                CompressedCache, slot, d, compressed_page_size, compressed_page_stride
            )

        scores = tl.dot(q, tl.trans(kv)) * sm_scale
        scores = tl.where(slot[None, :] >= 0, scores, -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp(m - next_m)
        p = tl.exp(scores - next_m[:, None])
        denom = denom * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), kv, acc * alpha[:, None])
        m = next_m

    if num_splits == 1:
        result = acc / tl.where(denom > 0, denom, 1.0)[:, None]
        tl.store(
            Out + row * out_row_stride + h[:, None] * 512 + d[None, :],
            result,
            h[:, None] < heads,
        )
    else:
        partial = Partials + ((row * heads + h) * num_splits + split) * partial_stride
        tl.store(partial[:, None] + d[None, :], acc, h[:, None] < heads)
        tl.store(partial + 512, m, h < heads)
        tl.store(partial + 513, denom, h < heads)


@triton.jit
def _merge_sparse_mla_splits(
    Partials,
    Sink,
    Out,
    out_row_stride: tl.constexpr,
    heads: tl.constexpr,
    num_splits: tl.constexpr,
    partial_stride: tl.constexpr,
    reduce_tile: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.program_id(2) * reduce_tile + tl.arange(0, reduce_tile)
    splits = tl.arange(0, num_splits)
    partial = Partials + ((row * heads + h) * num_splits + splits) * partial_stride
    m = tl.load(partial + 512)
    denom = tl.load(partial + 513)
    sink = tl.load(Sink + h)
    # Merge the split states and add the zero-valued sink once.
    maximum = tl.maximum(tl.max(m, 0), sink)
    weights = tl.exp(m - maximum)
    denom = tl.sum(weights * denom, 0) + tl.exp(sink - maximum)
    acc = tl.load(partial[:, None] + d[None, :])
    result = tl.sum(acc * weights[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(
        Out + row * out_row_stride + h * 512 + d,
        result,
    )


def sparse_mla_sm80(
    q: torch.Tensor,
    output: torch.Tensor,
    sink: torch.Tensor,
    positions: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lens: torch.Tensor | None = None,
    compress_ratio: int = 0,
    *,
    is_decode: bool = False,
) -> None:
    """Attend directly to paged SWA and selected compressed KV rows."""
    if not q.shape[0]:
        return
    assert q.dtype == output.dtype == torch.bfloat16
    assert q.shape == output.shape and q.shape[2] == 512
    assert q.stride()[1:] == output.stride()[1:] == (512, 1)
    assert swa_cache.dtype == torch.uint8
    assert swa_cache.stride()[1:] == (V4_BYTES_PER_TOKEN, 1)
    if compressed_cache is not None:
        assert compressed_cache.dtype == torch.uint8
        assert compressed_cache.stride()[1:] == swa_cache.stride()[1:]
        assert compressed_indices is not None and compressed_lens is not None
    head_tile = 16
    rows, heads, _ = q.shape
    head_blocks = triton.cdiv(heads, head_tile)
    num_splits, key_tile, num_stages = 1, 32, 3
    # Include multi-request DSpark verification grids, and reduce the split
    # count as more query rows fill the device.
    if is_decode:
        split_limit = 32 if compress_ratio else 8
        num_splits = min(
            split_limit, max(1, 256 // triton.next_power_of_2(rows * head_blocks))
        )
        key_tile, num_stages = 16, 1
    # Two FP32 statistics follow each partial vector. Keep the vectors aligned.
    partial_stride = 516
    partials = (
        torch.empty(
            (rows, heads, num_splits, partial_stride),
            dtype=torch.float32,
            device=q.device,
        )
        if num_splits > 1
        else None
    )
    _sparse_mla[(rows, head_blocks, num_splits)](
        q,
        output,
        partials,
        sink,
        positions,
        swa_cache,
        swa_indices,
        swa_lens,
        compressed_cache,
        compressed_indices,
        compressed_lens,
        q.stride(0),
        output.stride(0),
        swa_cache.shape[1],
        swa_cache.stride(0),
        swa_indices.stride(0),
        compressed_cache.shape[1] if compressed_cache is not None else 0,
        compressed_cache.stride(0) if compressed_cache is not None else 0,
        compressed_indices.stride(0) if compressed_indices is not None else 0,
        compressed_indices.shape[1] if compressed_indices is not None else 0,
        compress_ratio,
        heads,
        scale,
        head_tile,
        key_tile,
        num_splits,
        partial_stride,
        num_warps=4,
        num_stages=num_stages,
    )
    if partials is not None:
        reduce_tile = 128
        _merge_sparse_mla_splits[(rows, heads, 512 // reduce_tile)](
            partials,
            sink,
            output,
            output.stride(0),
            heads,
            num_splits,
            partial_stride,
            reduce_tile,
            num_warps=4,
        )
