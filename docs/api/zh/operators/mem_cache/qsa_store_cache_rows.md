# qsa_store_cache_rows

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

qsa_store_cache_rows(cache, slot_mapping, rows, *, impl="auto")
```

默认使用 Triton。

## 功能说明

按行配对输入和展平的槽号，取两者较短的长度。超出 [0,P*S) 的槽跳过。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `cache` | Tensor [P,S,1,D] | 可写、非空的分页缓存，可存 BF16 key 或 int64 位置。 |
| `slot_mapping` | int32/int64 | 按展平顺序取槽号。有效写入目标应互不重复。 |
| `rows` | Tensor [N,D] / [N,1,D] | 类型与 cache 一致。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| None | 原地更新 cache。 |

## Triton

根据分页缓存的实际步长写入。

## torch_native

用有效槽对应的页号和行号索引赋值。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.mem_cache import qsa_store_cache_rows

cache = torch.zeros(4, 8, 1, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([3, -1], device="npu")
rows = cache.new_ones((2, 128))
qsa_store_cache_rows(cache, slots, rows, impl="triton")
```
