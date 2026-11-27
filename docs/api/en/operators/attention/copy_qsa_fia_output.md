# copy_qsa_fia_output

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

copy_qsa_fia_output(result, selected, output, *, impl="auto")
```

The default selects Triton.

## Description

Copy FIA results for selected rows and write zero for rows without any visible selection.

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `result` | BF16/FP16 [T,H,D] | Contiguous FIA output. |
| `selected` | bool [T] | Contiguous selection flags returned with the mask. |
| `output` | Tensor [T,H,D] | Writable, same dtype as result. Supports strided, nonoverlapping rows. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| None | Writes output in place. |

## Triton

Fuses selection and the strided copy.

## torch_native

Uses torch.where and copies into output.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

x = torch.ones(2, 4, 256, device="npu", dtype=torch.bfloat16)
selected = torch.tensor([True, False], device="npu")
out = torch.empty_like(x)
copy_qsa_fia_output(x, selected, out, impl="triton")
```
