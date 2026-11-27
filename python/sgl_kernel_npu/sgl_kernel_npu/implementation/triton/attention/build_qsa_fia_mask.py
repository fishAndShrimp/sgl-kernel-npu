# SPDX-License-Identifier: Apache-2.0
"""Adapted from the vLLM NPU plugin, qwen4_exp/fia.py."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["mask_words"])
def _initialize_qsa_mask_words_kernel(
    words,
    mask_words,
    word_stride,
    WORD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    for word_start in range(0, mask_words, WORD_BLOCK):
        word = word_start + tl.arange(0, WORD_BLOCK)
        tl.store(words + row * word_stride + word, 0x01010101, word < mask_words)


@triton.jit(do_not_specialize=["kv_length", "mask_words"])
def _prepare_qsa_mask_words_kernel(
    compact_indices,
    visible_lens,
    scatter_indices,
    updates,
    has_selection,
    stride_row,
    stride_column,
    scatter_stride_row,
    updates_stride_row,
    kv_length,
    mask_words,
    update_width,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(visible_lens + row), 0), kv_length)
    group = tl.load(
        compact_indices + row * stride_row + columns * stride_column,
        columns < GROUPS,
        other=-1,
    )
    valid = (columns < GROUPS) & (group >= 0) & (group < visible // 4)
    residual = visible % 4
    valid_count = tl.sum(valid.to(tl.int32), 0)
    # Reserve one extra update for the incomplete tail even if all groups are selected.
    selected = (valid_count > 0) | (residual != 0)
    # Each complete four-token group is one aligned int32 mask word. Keep
    # row-local targets so ScatterElementsV2 can use its efficient dim=1 path.
    target = tl.where(valid, group, mask_words + columns)
    tail = (columns == GROUPS) & (residual != 0)
    tail_value = (0x01010101 << (residual * 8)).to(tl.int32)
    target = tl.where(tail, visible // 4, target)
    value = tl.where(tail, tail_value, 0)
    # An empty row attends token zero and is explicitly zeroed after FIA.
    dummy = (columns == 0) & (~selected)
    target = tl.where(dummy, 0, target)
    value = tl.where(dummy, 0x01010100, value)
    store_mask = columns < update_width
    tl.store(
        scatter_indices + row * scatter_stride_row + columns,
        target.to(tl.int64),
        store_mask,
    )
    tl.store(updates + row * updates_stride_row + columns, value, store_mask)
    tl.store(has_selection + row, selected)


def build_qsa_fia_mask(
    compact_indices: torch.Tensor,
    visible_lens: torch.Tensor,
    kv_length: int,
    mask_width: int,
    workspace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill and return a reusable [T, KV] mask plus nonempty-row flags."""
    if compact_indices.ndim != 2 or compact_indices.dtype != torch.int32:
        raise ValueError("compact_indices must be int32 [rows, groups]")
    rows = compact_indices.shape[0]
    if (
        visible_lens.shape != (rows,)
        or visible_lens.dtype not in (torch.int32, torch.int64)
        or not visible_lens.is_contiguous()
    ):
        raise ValueError(
            "visible_lens must be a contiguous integer vector with one entry per row"
        )
    if (
        compact_indices.device != visible_lens.device
        or workspace.storage.device != compact_indices.device
    ):
        raise ValueError("workspace and inputs must share a device")
    if not 0 < kv_length <= mask_width:
        raise ValueError("mask_width must cover the positive KV length")
    # Dense DMA stores must not share a 32-byte block with another row.
    mask_words = triton.cdiv(mask_width, 32) * 8
    groups = compact_indices.shape[1]
    update_width = triton.cdiv(groups + 1, 8) * 8
    storage, words, indices, updates, selected = workspace.views(
        rows, mask_words, update_width
    )
    if not rows:
        return words.view(torch.bool)[:, :mask_width], selected
    word_stride = mask_words + update_width
    word_block = min(4096, triton.next_power_of_2(mask_words))
    _initialize_qsa_mask_words_kernel[(rows,)](
        storage, mask_words, word_stride, WORD_BLOCK=word_block
    )
    _prepare_qsa_mask_words_kernel[(rows,)](
        compact_indices,
        visible_lens,
        indices,
        updates,
        selected,
        compact_indices.stride(0),
        compact_indices.stride(1),
        indices.stride(0),
        updates.stride(0),
        kv_length=kv_length,
        mask_words=mask_words,
        update_width=update_width,
        GROUPS=groups,
        BLOCK=triton.next_power_of_2(update_width),
    )
    storage.scatter_(1, indices, updates)
    words.copy_(storage[:, :mask_words])
    # The NPU's little-endian int32 representation is four bool bytes. This is
    # a dtype view, not a numerical conversion (which would collapse the bytes).
    mask = words.view(torch.bool)
    return mask[:, :mask_width], selected
