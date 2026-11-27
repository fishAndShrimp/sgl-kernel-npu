# build_qsa_fia_mask

## 芯片支持

| 后端 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

build_qsa_fia_mask(compact_indices, visible_lens, kv_length, mask_width, workspace, *, impl="auto")
```

默认使用 Triton。

## 功能说明

放开选中的完整四 token 块和末尾可见的 1 至 3 个 token。空行临时允许看 token 0，并返回 False 标志供 FIA 输出清零。

## 参数与返回值

输入、输出和工作区位于同一颗 NPU。

| 参数 | 类型与形状 | 约束 |
| --- | --- | --- |
| `compact_indices` | int32 [T,K] | 逻辑块号。忽略无效或尚未完整可见的块。 |
| `visible_lens` | int32/int64 [T] | 连续，裁剪至 [0,kv_length]。 |
| `kv_length` | int | KV 长度，正整数。 |
| `mask_width` | int | 含填充的宽度，至少为 kv_length。 |
| `workspace` | QSAFIAMaskWorkspace | 与输入同设备，至少容纳 T 行、T × 向上对齐到 32 的 mask_width 字节，以及向上对齐到 8 的 K+1 个更新列。同一流内顺序复用。 |
| `impl` | str | `"auto"`, `"triton"`, `"torch_native"` |

| 返回值 | 含义 |
| --- | --- |
| bool [T,mask_width] | True 表示屏蔽。与工作区共用存储，下次使用时覆盖。 |
| bool [T] | True 表示有真实选择。与工作区共用存储。 |

## Triton

初始化掩码字，生成各行更新位置并散写到掩码。

## torch_native

根据可见 token 和块号直接构造布尔掩码。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention import build_qsa_fia_mask

from sgl_kernel_npu.operators.attention import QSAFIAMaskWorkspace

ids = torch.tensor([[1, -1]], dtype=torch.int32, device="npu")
lens = torch.tensor([10], dtype=torch.int32, device="npu")
scratch = QSAFIAMaskWorkspace(ids.device, 1, 32, 8)
mask, selected = build_qsa_fia_mask(ids, lens, 10, 32, scratch, impl="triton")
```
