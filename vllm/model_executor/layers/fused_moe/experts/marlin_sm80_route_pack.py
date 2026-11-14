# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton

# Larger sort tiles lose most of the single-warp latency advantage on SM80.
MAX_SM80_ROUTE_PACK_ROUTES = 128


@triton.jit
def _max_int(x, y):
    return tl.maximum(x, y)


@triton.jit
def _route_pack_sm80(
    topk_ptr,
    map_ptr,
    sorted_ptr,
    expert_ptr,
    count_ptr,
    ROUTES: tl.constexpr,
    GLOBAL_EXPERTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
):
    slot = tl.arange(0, BLOCK_R)
    global_id = tl.load(topk_ptr + slot, slot < ROUTES, -1).to(tl.int32)
    in_range = (global_id >= 0) & (global_id < GLOBAL_EXPERTS)
    if HAS_EXPERT_MAP:
        local_id = tl.load(map_ptr + global_id, in_range, -1)
    else:
        local_id = tl.where(in_range, global_id, -1)
    # Sort by (local expert, route slot), with nonlocal and padded routes last.
    key = tl.where(local_id >= 0, local_id * BLOCK_R + slot, 0x7FFF_FFFF)
    packed = tl.sort(key, descending=False)
    expert = packed // BLOCK_R
    valid = packed != 0x7FFF_FFFF

    # Round each expert run to eight rows without sharing state across CTAs.
    previous = tl.gather(expert, tl.maximum(slot - 1, 0), axis=0)
    following = tl.gather(expert, tl.minimum(slot + 1, BLOCK_R - 1), axis=0)
    first = (slot == 0) | (expert != previous)
    last = ((slot == BLOCK_R - 1) | (expert != following)) & valid
    start = tl.associative_scan(tl.where(first, slot, 0), 0, _max_int)
    intra = slot - start
    padding = tl.where(last, (-intra - 1) & 7, 0)
    out = slot + tl.cumsum(padding, axis=0) - padding
    tl.store(sorted_ptr + out, packed % BLOCK_R, valid)
    tl.store(expert_ptr + out // 8, expert, valid & ((intra & 7) == 0))

    lanes = tl.arange(0, 8)
    tl.store(
        sorted_ptr + out[:, None] + lanes[None, :] + 1,
        ROUTES,
        last[:, None] & (lanes[None, :] < padding[:, None]),
    )
    tl.store(count_ptr, tl.sum(valid.to(tl.int32) + padding, axis=0))


def marlin_sm80_route_pack(
    topk_ids: torch.Tensor, num_experts: int, expert_map: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack local routes into Marlin's eight-row blocks on one CTA."""
    routes = topk_ids.numel()
    # Match Marlin's default capacity. Entries past count are never read.
    capacity = min(routes * 8, routes + num_experts * 7)
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=topk_ids.device)
    experts = torch.empty(
        triton.cdiv(capacity, 8), dtype=torch.int32, device=topk_ids.device
    )
    count = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    _route_pack_sm80[(1,)](
        topk_ids,
        expert_map,
        sorted_ids,
        experts,
        count,
        routes,
        num_experts,
        triton.next_power_of_2(routes),
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=1,
    )
    return sorted_ids, experts, count
