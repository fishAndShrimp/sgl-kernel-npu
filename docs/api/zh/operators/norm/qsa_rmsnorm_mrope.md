# qsa_rmsnorm_mrope

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

qsa_rmsnorm_mrope(tensor, weight, cos_sin, eps, mrope_section, rope_dim, positions=None, *, impl="auto")
```

默认使用 Triton。

## 功能说明

按 FP32 计算 RMSNorm，缩放系数为 1 + weight。对前 rope_dim 维做半区配对的旋转编码，归一化结果和旋转乘积按 BF16 舍入。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `tensor` | BF16 [T,H,D] | 最后一维连续。 |
| `weight` | BF16 [D] | 连续。 |
| `cos_sin` | BF16/FP32 | [T,R]、[3,T,R] 或查找表 [P,R]。前半为 cos，后半为 sin，末维连续。 |
| `eps` | float | 加到平方均值上。 |
| `mrope_section` | list[int] | 三轴交错旋转的分段宽度。[T,R] 输入忽略此项。 |
| `rope_dim` | int | R 为 2 的幂，2 ≤ R ≤ D。 |
| `positions` | int32/int64 [3,T] | 可选查表索引，范围 [-P,P)。默认 None 表示已取出对应行。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| BF16 [T,H,D] | 新分配的归一化并旋转后的张量。 |

## Triton

每个 token 的每个头用一个 program。直接查表时先把表值转为 BF16。

## torch_native

用 PyTorch 归约、分段乘加计算。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.norm import qsa_rmsnorm_mrope

x = torch.ones(2, 4, 128, dtype=torch.bfloat16, device="npu")
w = x.new_zeros(128)
trig = x.new_ones((2, 64))
y = qsa_rmsnorm_mrope(x, w, trig, 1e-6, [], 64, impl="triton")
```
