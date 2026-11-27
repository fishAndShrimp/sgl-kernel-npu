# SPDX-License-Identifier: Apache-2.0
"""Adapted from the vLLM NPU plugin, qwen4_exp/fia.py."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_qsa_fia_output_kernel(
    result,
    selected,
    output,
    stride_row,
    stride_head,
    stride_dim,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(result + row * WIDTH + columns, columns < WIDTH, other=0)
    valid = tl.load(selected + row)
    if CONTIGUOUS:
        offset = row * WIDTH + columns
    else:
        offset = (
            row * stride_row
            + columns // HEAD_DIM * stride_head
            + columns % HEAD_DIM * stride_dim
        )
    tl.store(output + offset, tl.where(valid, values, 0), columns < WIDTH)


def copy_qsa_fia_output(
    result: torch.Tensor, selected: torch.Tensor, output: torch.Tensor
) -> None:
    if (
        output.ndim != 3
        or result.shape != output.shape
        or not result.is_contiguous()
        or result.dtype != output.dtype
    ):
        raise ValueError(
            "result must be contiguous and match output [rows, heads, dim] and dtype"
        )
    if (
        selected.shape != (output.shape[0],)
        or selected.dtype != torch.bool
        or not selected.is_contiguous()
    ):
        raise ValueError(
            "selected must be a contiguous bool vector with one entry per row"
        )
    if selected.device != result.device or output.device != result.device:
        raise ValueError("result, selected and output must share a device")
    if not output.numel():
        return
    width = output.shape[1] * output.shape[2]
    _copy_qsa_fia_output_kernel[(output.shape[0], triton.cdiv(width, 1024))](
        result,
        selected,
        output,
        *output.stride(),
        HEAD_DIM=output.shape[2],
        WIDTH=width,
        BLOCK=1024,
        CONTIGUOUS=output.is_contiguous(),
    )
