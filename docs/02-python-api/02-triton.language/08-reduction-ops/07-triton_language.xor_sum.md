---
title: triton_language.xor_sum
---

```python
triton.language.xor_sum(input, axis=None, keep_dims=False)
```

返回 `input` 张量沿指定 `axis` 的所有元素的异或和。

归约操作应当满足结合律与交换律。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.xor_sum(...)` 而不是 `xor_sum(x, ...)`。
