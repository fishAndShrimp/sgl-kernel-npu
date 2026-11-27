# SPDX-License-Identifier: Apache-2.0
"""Adapted from the vLLM NPU plugin, qwen4_exp/fia.py."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


class QSAFIAMaskWorkspace:
    """Caller-owned scratch buffers. Reuse on one stream after consuming the previous mask."""

    def __init__(
        self, device: torch.device, max_rows: int, max_bytes: int, max_update_width: int
    ) -> None:
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.max_update_width = max_update_width
        self.mask_word_capacity = max_bytes // torch.int32.itemsize
        self.word_capacity = self.mask_word_capacity + max_rows * max_update_width
        self.storage = torch.empty(
            (self.word_capacity,), dtype=torch.int32, device=device
        )
        self.words = torch.empty(
            (self.mask_word_capacity,), dtype=torch.int32, device=device
        )
        self.indices = torch.empty(
            (max_rows, max_update_width), dtype=torch.int64, device=device
        )
        self.updates = torch.empty(
            (max_rows, max_update_width), dtype=torch.int32, device=device
        )
        self.selected = torch.empty((max_rows,), dtype=torch.bool, device=device)

    def views(
        self, rows: int, mask_words: int, update_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_size = rows * mask_words
        word_stride = mask_words + update_width
        storage_size = rows * word_stride
        if (
            rows > self.max_rows
            or mask_size > self.mask_word_capacity
            or storage_size > self.word_capacity
            or update_width > self.max_update_width
        ):
            raise ValueError(
                f"QSA FIA mask workspace cannot cover rows={rows}, words={mask_words}, updates={update_width}"
            )
        storage = self.storage[:storage_size].view(rows, word_stride)
        words = self.words[:mask_size].view(rows, mask_words)
        return (
            storage,
            words,
            self.indices[:rows, :update_width],
            self.updates[:rows, :update_width],
            self.selected[:rows],
        )


OP = "attention.build_qsa_fia_mask"

_triton_fn = None
_torch_native_fn = None


@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(compact_indices, visible_lens, kv_length, mask_width, workspace):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.build_qsa_fia_mask import (
            build_qsa_fia_mask,
        )

        _triton_fn = build_qsa_fia_mask
    return _triton_fn(compact_indices, visible_lens, kv_length, mask_width, workspace)


@register_impl(OP, "torch_native")
def _torch_native_impl(compact_indices, visible_lens, kv_length, mask_width, workspace):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.build_qsa_fia_mask import (
            build_qsa_fia_mask,
        )

        _torch_native_fn = build_qsa_fia_mask
    return _torch_native_fn(
        compact_indices, visible_lens, kv_length, mask_width, workspace
    )


@register_operator(OP)
def build_qsa_fia_mask(
    compact_indices, visible_lens, kv_length, mask_width, workspace, *, impl="auto"
):
    """Build a bool [rows,mask_width] exclusion mask and [rows] selection flags in caller-owned workspace.

    The default implementation is Triton.
    """
    return get_impl_by_name(OP, impl)(
        compact_indices, visible_lens, kv_length, mask_width, workspace
    )
