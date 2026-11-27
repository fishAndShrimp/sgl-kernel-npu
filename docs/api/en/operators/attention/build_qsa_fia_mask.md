# build_qsa_fia_mask

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

build_qsa_fia_mask(compact_indices, visible_lens, kv_length, mask_width, workspace, *, impl="auto")
```

The default selects Triton.

## Description

Unmask selected complete four-token groups and the visible one-to-three-token tail. For empty selections allow token zero for FIA and return a false flag.

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `compact_indices` | int32 [T,K] | Logical group IDs. Ignore invalid and nonvisible full groups. |
| `visible_lens` | int32/int64 [T] | Contiguous. Clamped to [0,kv_length]. |
| `kv_length` | int | Positive KV length. |
| `mask_width` | int | At least kv_length, includes padding. |
| `workspace` | QSAFIAMaskWorkspace | Allocate on the input device with max_rows ≥ T, max_bytes ≥ T × round_up(mask_width,32), max_update_width ≥ round_up(K+1,8). Reuse sequentially on one stream. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| bool [T,mask_width] | True excludes a token. Aliases workspace until its next use. |
| bool [T] | True for a nonempty real selection. Also aliases workspace. |

## Triton

Initialize packed words, prepare row-local updates and scatter them into the mask.

## torch_native

Builds the boolean attention mask directly from visible token positions.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

from sgl_kernel_npu.operators.attention import QSAFIAMaskWorkspace

ids = torch.tensor([[1, -1]], dtype=torch.int32, device="npu")
lens = torch.tensor([10], dtype=torch.int32, device="npu")
scratch = QSAFIAMaskWorkspace(ids.device, 1, 32, 8)
mask, selected = build_qsa_fia_mask(ids, lens, 10, 32, scratch, impl="triton")
```
