---
title: triton_language.topk
---

```python
triton.language.topk(x, k: core.constexpr, dim: core.constexpr = None, descending: core.constexpr = True)
```

返回输入张量沿指定维度的 k 个最大（或最小）元素。

这些元素按排序后的顺序返回（最大者优先）。

**参数：**

* **x** (*Tensor*) - 输入张量。
* **k** (*int*) - 要返回的前若干个元素的数量。必须是正的 2 的幂，且不能大于所选维度的长度。
* **dim** (*int*, *可选*) - 用于查找前 k 个元素的维度。如果为 None，则使用最后一个维度。目前仅支持最后一个维度。
* **descending** (*bool*, *可选*) - 如果设置为 True，返回 k 个最大元素。如果设置为 False，返回 k 个最小元素。

**返回：** 包含沿指定维度的 k 个最大元素的张量。

## 示例

```python
# Get top 4 elements from a 1D tensor
# 从一维张量中获取前 4 个元素
x = tl.arange(0, 16)
top4 = tl.topk(x, 4)  # Returns [15, 14, 13, 12]
# 返回 [15, 14, 13, 12]
```
