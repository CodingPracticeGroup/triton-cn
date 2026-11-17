---
title: triton_language.reduce
---

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False)
```

将 combine_fn 应用于 `input` 中各张量沿指定 `axis` 的所有元素。

**参数：**

* **input**（*Tensor*）- 输入张量，或张量的元组。
* **axis**（*int | None*）- 应在该维度上执行归约。如果为 None，则归约所有维度。
* **combine_fn**（*Callable*）- 用于组合两组标量张量的函数（必须使用 @triton.jit 标记）。该函数及其调用的被调函数必须是纯的，例外只有 :code:`assume`、:code:`device_assert` 与 :code:`device_print`。它们必须能在不占用共享内存暂存空间的前提下完成降级（lower）。
* **keep_dims**（*bool*）- 如果为 true，保留长度为 1 的归约维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.reduce(...)` 而不是 `reduce(x, ...)`。
