# copy_qsa_fia_output

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

copy_qsa_fia_output(result, selected, output, *, impl="auto")
```

默认使用 Triton。

## 功能说明

有有效选择的行复制 FIA 输出，空行写零。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `result` | BF16/FP16 [T,H,D] | 连续的 FIA 输出。 |
| `selected` | bool [T] | 生成掩码时返回的有效行标志，连续。 |
| `output` | Tensor [T,H,D] | 可写，类型与 result 相同。支持有步长且不重叠的视图。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| None | 原地写入 output。 |

## Triton

合并行选择和带步长写入。

## torch_native

用 torch.where 选择，再复制到 output。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention import copy_qsa_fia_output

x = torch.ones(2, 4, 256, device="npu", dtype=torch.bfloat16)
selected = torch.tensor([True, False], device="npu")
out = torch.empty_like(x)
copy_qsa_fia_output(x, selected, out, impl="triton")
```
