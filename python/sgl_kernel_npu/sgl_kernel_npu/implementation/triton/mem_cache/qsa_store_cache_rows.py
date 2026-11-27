# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from the vLLM NPU plugin, qwen4_exp/qsa.py."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    cache_capacity = num_blocks * PAGE_SIZE
    valid = (row < num_rows) & (slot >= 0) & (slot < cache_capacity)
    # The NPU Triton backend can validate addresses before applying the store mask.
    # Keep every lane in bounds even when graph padding supplies an invalid
    # positive slot. Cast before multiplying by a potentially large stride so
    # the physical cache offset cannot overflow int32 either.
    safe_slot = tl.minimum(tl.maximum(slot, 0), cache_capacity - 1).to(tl.int64)
    block = safe_slot // PAGE_SIZE
    token = safe_slot % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a strided QSA cache."""
    if rows.dtype != cache.dtype or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("rows must match the cache dtype and slots must be integers")
    if rows.device != cache.device or slot_mapping.device != cache.device:
        raise ValueError("cache, slots and rows must share a device")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.ndim != 2 or rows.shape[1] != cache.shape[3]:
        raise ValueError("QSA cache rows have an incompatible width")

    slot_mapping = slot_mapping.reshape(-1).contiguous()
    num_updates = min(slot_mapping.shape[0], rows.shape[0])
    slot_mapping = slot_mapping[:num_updates]
    rows = rows[:num_updates]
    if not num_updates:
        return
    _store_qsa_rows_kernel[(num_updates,)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        num_updates,
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )
