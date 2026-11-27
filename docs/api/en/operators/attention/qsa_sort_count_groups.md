# qsa_sort_count_groups

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

qsa_sort_count_groups(indices, *, impl="auto")
```

The default selects Triton.

## Description

Sort nonnegative block IDs in ascending order. Count each valid input and pad unused slots with 16777215.

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `indices` | int32 [T,512] | Contiguous. Negative entries are invalid. Supply unique valid IDs below 16777215 for GBSA. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| int32 [1,T,512] | Sorted IDs with padding. |
| int32 [1,T] | Valid counts. |

## Triton

Sort and count share one launch, processing eight rows per tile.

## torch_native

Masks invalid entries, sorts the IDs and counts valid inputs.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

ids = torch.arange(511, -1, -1, device="npu", dtype=torch.int32)[None]
blocks, counts = qsa_sort_count_groups(ids, impl="triton")
```
