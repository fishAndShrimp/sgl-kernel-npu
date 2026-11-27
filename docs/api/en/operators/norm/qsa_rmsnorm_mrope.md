# qsa_rmsnorm_mrope

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

qsa_rmsnorm_mrope(tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions=None, *, impl="auto")
```

The default selects Triton.

## Description

Compute RMSNorm in FP32 with scale 1 + weight. Apply split-half rotary embedding to the first rope_dim features. Normalized values and rotary products round to BF16.

## Parameters and return values

Use tensors on the same NPU.

| Parameter | Type and shape | Constraints |
| --- | --- | --- |
| `tensor` | BF16 [T,H,D] | Unit feature stride. |
| `weight` | BF16 [D] | Contiguous. |
| `cos_sin` | BF16/FP32 | [T,R], [3,T,R], or lookup table [P,R]. Cosines precede sines. Unit last stride. |
| `eps` | float | Added to the mean square. |
| `mrope_section` | list[int] | Three interleaved section widths for MRoPE. Ignored for [T,R]. |
| `rope_dim` | int | R is a power of two, 2 ≤ R ≤ D. |
| `positions` | int32/int64 [3,T] | Optional table indices in [-P,P). Default None uses preselected rows. |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| Return | Meaning |
| --- | --- |
| BF16 [T,H,D] | New normalized, rotated tensor. |

## Triton

One program per token and head. Direct table lookup rounds the table entries to BF16.

## torch_native

Uses PyTorch reductions and explicit rotary products.

## Example

```python
import torch
from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

x = torch.ones(2, 4, 128, dtype=torch.bfloat16, device="npu")
w = x.new_zeros(128)
trig = x.new_ones((2, 64))
y = qsa_rmsnorm_mrope(x, w, trig, 1e-6, [], 64, impl="triton")
```
