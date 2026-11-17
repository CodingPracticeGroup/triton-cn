---
title: triton_language.argmax
---

```python
triton.language.argmax(input, axis, tie_break_left=True, keep_dims=False)
```

返回 `input` 张量沿指定 `axis` 的所有元素的最大索引。

归约操作应当满足结合律与交换律。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。
* **tie_break_left**（*bool*）- 如果为 true，在出现平局的情况下（即多个元素具有相同的最大索引值），对于非 NaN 的值返回最左边的索引。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.argmax(...)` 而不是 `argmax(x, ...)`。
