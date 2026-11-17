---
title: triton_language.max
---

```python
triton.language.max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False, propagate_nan: core.constexpr = core.PropagateNan.NONE)
```

返回 `input` 张量沿指定 `axis` 的所有元素的最大值。

归约操作应当满足结合律与交换律。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。
* **return_indices**（*bool*）- 如果为 true，返回对应最大值位置的索引。
* **return_indices_tie_break_left**（*bool*）- 如果为 true，在出现平局的情况下（即多个元素具有相同的最大值），对于非 NaN 的值返回最左边的索引。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.max(...)` 而不是 `max(x, ...)`。
