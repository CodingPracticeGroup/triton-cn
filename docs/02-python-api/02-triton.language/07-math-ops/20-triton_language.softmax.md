---
title: triton_language.softmax
---

```python
triton.language.softmax(x, dim=None, *, keep_dims=None, ieee_rounding=False)
```

沿给定维度对 `x` 计算 softmax。

**参数：**

* **x**（*Block*）- 输入值。
* **dim**（*int | None*）- 要沿之进行归一化的维度。默认为 0——请注意这与 `torch.softmax` 不同，这里的默认值不是最后一个维度。
* **keep_dims**（*bool | None*）- 已弃用且会被忽略。softmax 始终保留输入的形状。默认为 `None`；传入其他任何值都会发出警告。必须以关键字参数方式传入。
* **ieee_rounding**（*bool*）- 最后的除法是否使用符合 IEEE 规范的舍入。必须以关键字参数方式传入。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.softmax(...)` 而不是 `softmax(x, ...)`。
