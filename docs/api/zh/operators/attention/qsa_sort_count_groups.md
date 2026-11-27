# qsa_sort_count_groups

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

qsa_sort_count_groups(indices, *, impl="auto")
```

默认使用 Triton。

## 功能说明

对非负块号升序排列，统计有效项数，用 16777215 填充余位。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `indices` | int32 [T,512] | 连续。负数为无效项。供 GBSA 使用的有效块号应互异且小于 16777215。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| int32 [1,T,512] | 排序并填充的块号。 |
| int32 [1,T] | 各行有效项数。 |

## Triton

一次启动完成排序和计数，每个 tile 处理 8 行。

## torch_native

通过掩码替换无效项，再排序和计数。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention import qsa_sort_count_groups

ids = torch.arange(511, -1, -1, device="npu", dtype=torch.int32)[None]
blocks, counts = qsa_sort_count_groups(ids, impl="triton")
```
