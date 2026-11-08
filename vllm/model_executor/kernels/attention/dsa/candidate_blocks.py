# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import has_deep_select
from vllm.model_executor.layers.indexer_topk import (
    deep_select_topk,
    get_deep_select_stride_requirement,
    get_indexer_topk,
    is_deep_select_supported,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    valid_blocks,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    stride_scores,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
):
    # Writes each row's valid block count; FULL_WIDTH also scores the -inf tail.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    col_end = tl.minimum(end, width)
    num_valid = tl.cdiv(tl.maximum(col_end - start, 0), BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(valid_blocks + row, num_valid)
    num_blocks = nblocks if FULL_WIDTH else num_valid
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    num_tiles = tl.cdiv(num_blocks, TILE)
    for tile_id in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        blocks = tile_id * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (offsets[None, :] < BLOCK_SIZE) & (cols < col_end),
            other=-float("inf"),
        ).to(tl.float32)
        reduced = tl.reduce(values, 1, _max_with_nan)
        # Pin the newest block. DeepSelect leaves NaN rows partially written,
        # so also pin NaN blocks (torch.topk ranks NaN first too).
        pinned = (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE)
        if not FULL_WIDTH:
            pinned |= reduced != reduced
        reduced = tl.where(pinned, float("inf"), reduced)
        tl.store(scores + row * stride_scores + blocks, reduced, blocks < num_blocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit
def _mask_invalid_topk_kernel(
    logits,
    lengths,
    indices,
    logit_stride,
    length_stride,
    index_stride,
    width,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    index = tl.load(indices + row * index_stride + cols, cols < K, -1)
    length = tl.minimum(tl.load(lengths + row * length_stride), width)
    valid = (cols < K) & (index >= 0) & (index < length)
    score = tl.load(logits + row * logit_stride + index, valid, -float("inf"))
    tl.store(
        indices + row * index_stride + cols,
        tl.where(valid & (score != -float("inf")), index, -1),
        cols < K,
    )


def mask_invalid_topk(
    logits: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> None:
    """Discard masked selections without reading unwritten, out-of-range columns."""
    rows, k = indices.shape
    assert logits.stride(1) == indices.stride(1) == 1
    _mask_invalid_topk_kernel[(rows, triton.cdiv(k, 256))](
        logits,
        lengths,
        indices,
        logits.stride(0),
        lengths.stride(0),
        indices.stride(0),
        logits.shape[1],
        k,
        256,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    MASK_TAIL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    valid_blocks = nblocks
    if not MASK_TAIL:
        start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
        end = tl.load(ends + row // ROW_REPEAT * stride_end)
        valid_blocks = tl.cdiv(
            tl.maximum(tl.minimum(end, width) - start, 0), BLOCK_SIZE
        )
    for tile in range(tl.cdiv(valid_blocks, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots < valid_blocks)
    tl.store(flags + row * (nblocks + 1) + nblocks, 0)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(
        flags + row * (nblocks + 1) + block,
        1,
        (cols < K) & (block >= 0) & ((block < valid_blocks) | (block == nblocks)),
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    MASK_TAIL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    limit = width if MASK_TAIL else tl.minimum(end, width)
    for tile in range(tl.program_id(1), tl.cdiv(limit, TILE), tl.num_programs(1)):
        cols = tile * TILE + tl.arange(0, TILE)
        valid = (cols >= start) & (cols < end) & (cols < width)
        block = (cols - start) // BLOCK_SIZE
        keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
        edge = tl.load(flags + row * (nblocks + 1) + nblocks)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            (cols < limit) & ~(valid & keep),
        )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    score_dtype: torch.dtype | None = None,
    *,
    ragged_topk: bool = False,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    score_dtype = score_dtype or logits.dtype
    if ragged_topk:
        assert score_dtype == torch.float32
        assert out.dtype == torch.int32 and out.is_contiguous()
    # Logits are max_model_len wide (CUDA-graph static) but rows are shorter:
    # DeepSelect stops at each row's valid block count. Order does not matter.
    use_deep_select = has_deep_select() and nblocks >= topk_blocks
    if use_deep_select:
        in_align, out_align = get_deep_select_stride_requirement()
        padded = round_up(nblocks, in_align // score_dtype.itemsize)
        scores = logits.new_empty((rows, padded), dtype=score_dtype)[:, :nblocks]
        use_deep_select = (
            out.stride(1) == 1
            and out.stride(0) * out.element_size() % out_align == 0
            and is_deep_select_supported(scores, topk_blocks)
        )
    else:
        scores = logits.new_empty((rows, nblocks), dtype=score_dtype)
    valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _block_scores_kernel[(rows, 32)](
        logits,
        row_ks,
        row_ke,
        scores,
        valid_blocks,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        scores.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        TILE=128,
        FULL_WIDTH=not (use_deep_select or ragged_topk),
    )
    if ragged_topk:
        get_indexer_topk("per_row")(scores, valid_blocks, 1, out, topk_blocks, nblocks)
        mask_invalid_topk(scores, valid_blocks, out)
        return
    if use_deep_select:
        deep_select_topk(scores, topk_blocks, end=valid_blocks, output_idx=out)
        return
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
    *,
    mask_tail: bool = True,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        row_ke,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
        mask_tail,
    )
    grid = triton.cdiv(width, 1024)
    if not mask_tail:
        grid = min(grid, 32)
    _mask_candidates_kernel[(rows, grid)](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
        mask_tail,
    )
