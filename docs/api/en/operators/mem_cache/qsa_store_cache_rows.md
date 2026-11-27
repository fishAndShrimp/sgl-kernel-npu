# qsa_store_cache_rows

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

qsa_store_cache_rows(cache, slot_mapping, rows, *, impl="auto")
```

The default selects Triton.

## Description

Pair rows and flattened slot IDs up to the shorter input. Skip slots outside [0,P*S).

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `cache` | Tensor [P,S,1,D] | Writable, nonempty paged cache. BF16 keys or int64 positions. |
| `slot_mapping` | int32/int64 | Flattened slot IDs. Live destinations must be unique. |
| `rows` | Tensor [N,D] / [N,1,D] | Same dtype as cache. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| None | Updates cache in place. |

## Triton

Uses the actual strides of the paged cache.

## torch_native

Uses masked indexing with page and row coordinates.

## Example

```python
import torch
from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

cache = torch.zeros(4, 8, 1, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([3, -1], device="npu")
rows = cache.new_ones((2, 128))
qsa_store_cache_rows(cache, slots, rows, impl="triton")
```
