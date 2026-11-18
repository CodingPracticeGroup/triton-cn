---
title: triton_language.any
---

```python
triton.language.any(input, axis=None, keep_dims=False)
```

返回 `input` 中的元素沿指定 `axis` 是否存在非零值。

结果的数据类型为 `int1`。NaN 与无穷大被视为非零值。

**参数：**

* **input**（*Tensor*）- 输入值。
* **axis**（*int | None*）- 要归约的维度。如果为 None，则归约所有维度。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.any(...)` 而不是 `any(x, ...)`。
