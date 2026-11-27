# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from the vLLM NPU plugin, qwen4_exp/qsa.py."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_compress_group_slots(
    keys,
    rope,
    groups,
    pooled,
    first_positions,
    KEY_ROW: tl.constexpr,
    KEY_COL: tl.constexpr,
    ROPE_ROW: tl.constexpr,
    ROPE_COL: tl.constexpr,
    GROUP_ROW: tl.constexpr,
    GROUP_COL: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    # Groups can refer to nonadjacent slots in the key cache.
    for offset in tl.range(0, COMPRESS_RATIO):
        slot = tl.load(groups + row * GROUP_ROW + offset * GROUP_COL).to(tl.int64)
        values = tl.load(
            keys + slot * KEY_ROW + dims * KEY_COL, mask=dims < WIDTH, other=0.0
        ).to(tl.float32)
        accumulator += values
    tl.store(
        pooled + row * WIDTH + dims, accumulator / COMPRESS_RATIO, mask=dims < WIDTH
    )
    first_slot = tl.load(groups + row * GROUP_ROW).to(tl.int64)
    axes = tl.arange(0, 4)
    position = tl.load(
        rope + first_slot * ROPE_ROW + axes * ROPE_COL, mask=axes < 3, other=0
    )
    tl.store(first_positions + row * 3 + axes, position, mask=axes < 3)


def qsa_compress_group_slots(keys, positions, groups):
    if keys.ndim == 3:
        if keys.shape[1] != 1:
            raise ValueError("keys must have one head")
        keys = keys[:, 0]
    if keys.ndim != 2 or keys.dtype != torch.bfloat16 or keys.shape[1] != 128:
        raise ValueError("keys must have BF16 rows of width 128")
    if positions.shape != (keys.shape[0], 3) or positions.dtype != torch.int64:
        raise ValueError("positions must be int64 [slots, 3]")
    if (
        groups.ndim != 2
        or groups.shape[1] != 4
        or groups.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("groups must be integer [rows, 4]")
    if keys.device != positions.device or keys.device != groups.device:
        raise ValueError("keys, positions and groups must share a device")
    pooled = torch.empty(
        (groups.shape[0], 1, keys.shape[1]), dtype=keys.dtype, device=keys.device
    )
    first = torch.empty(
        (groups.shape[0], 3), dtype=positions.dtype, device=positions.device
    )
    if groups.shape[0]:
        _qsa_compress_group_slots[(groups.shape[0],)](
            keys,
            positions,
            groups,
            pooled,
            first,
            KEY_ROW=keys.stride(0),
            KEY_COL=keys.stride(1),
            ROPE_ROW=positions.stride(0),
            ROPE_COL=positions.stride(1),
            GROUP_ROW=groups.stride(0),
            GROUP_COL=groups.stride(1),
            WIDTH=128,
            BLOCK_D=128,
            COMPRESS_RATIO=4,
            num_warps=4,
        )
    return pooled, first
