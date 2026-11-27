# qsa_compress_group_slots

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

qsa_compress_group_slots(keys, positions, groups, *, impl="auto")
```

默认使用 Triton。

## 功能说明

每组 4 个槽位的 key 按 FP32 求平均并转回 BF16，同时取第一个槽位的三轴位置。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `keys` | BF16 [S,128] / [S,1,128] | 单个头。 |
| `positions` | int64 [S,3] | 每个槽位的三轴坐标。 |
| `groups` | int32/int64 [G,4] | 槽号在 [0,S) 内，允许重复。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| BF16 [G,1,128] | 分组平均后的 key。 |
| int64 [G,3] | 每组首槽坐标。 |

## Triton

每个 program 直接读取该组的 4 个槽。

## torch_native

先按槽号取 key，再沿组内维度求均值。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.indexer import qsa_compress_group_slots

k = torch.ones(4, 128, device="npu", dtype=torch.bfloat16)
p = torch.arange(12, device="npu").reshape(4, 3)
g = torch.arange(4, device="npu").reshape(1, 4)
keys, positions = qsa_compress_group_slots(k, p, g, impl="triton")
```
