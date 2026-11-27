# qsa_compress_group_slots

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

qsa_compress_group_slots(keys, positions, groups, *, impl="auto")
```

The default selects Triton.

## Description

Average each four-slot group in FP32 and cast to BF16. Copy the three position IDs from its first slot.

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `keys` | BF16 [S,128] / [S,1,128] | One head. |
| `positions` | int64 [S,3] | Three coordinates per slot. |
| `groups` | int32/int64 [G,4] | Slot IDs in [0,S), repeats allowed. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| BF16 [G,1,128] | Pooled keys. |
| int64 [G,3] | First-slot coordinates. |

## Triton

One program reads the four supplied slots directly.

## torch_native

Gathers keys by group and averages across slots.

## Example

```python
import torch
from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

k = torch.ones(4, 128, device="npu", dtype=torch.bfloat16)
p = torch.arange(12, device="npu").reshape(4, 3)
g = torch.arange(4, device="npu").reshape(1, 4)
keys, positions = qsa_compress_group_slots(k, p, g, impl="triton")
```
