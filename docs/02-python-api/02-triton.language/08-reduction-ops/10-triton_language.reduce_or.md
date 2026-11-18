---
title: triton_language.reduce_or
---

```python
triton.language.reduce_or(input, axis, keep_dims=False)
```

返回 `input` 张量沿指定 `axis` 对所有元素做 reduce_or 的结果。

归约操作应当满足结合律与交换律。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.reduce_or(...)` 而不是 `reduce_or(x, ...)`。
