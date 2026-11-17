---
title: triton_language.sum
---

```python
triton.language.sum(input, axis=None, keep_dims=False, dtype=None)
```

返回 `input` 张量沿指定 `axis` 的所有元素的总和。

归约操作应当满足结合律与交换律。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。
* **dtype**（*tl.dtype*）- 期望的返回张量数据类型。如果指定，则在执行该操作之前先把输入张量转换为 :code:`dtype`。这有助于避免数据溢出。如果不指定，位宽窄于 32 位的有符号整数类型会被加宽为 :code:`tl.int32`，位宽窄于 32 位的无符号整数类型与 bool 类型会被加宽为 :code:`tl.uint32`。其他类型保持原样。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.sum(...)` 而不是 `sum(x, ...)`。
