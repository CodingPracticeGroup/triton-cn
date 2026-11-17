---
title: triton_language.cumprod
---

```python
triton.language.cumprod(input, axis=0, reverse=False, dtype=None)
```

返回沿指定 `axis` 的 `input` 张量中所有元素的累积乘积。

**参数：**

* **input** (*Tensor*) - 输入值。
* **axis** (*int*) - 应进行扫描的维度。
* **reverse** (*bool*) - 如果为 true，则沿反方向执行扫描。
* **dtype** (*tl.dtype*) - 期望的返回张量的数据类型。如果指定，则在执行该运算之前先将输入张量转换为 `dtype`。如果未指定，位宽小于 32 位的有符号整数 dtype 会被提升为 `tl.int32`，位宽小于 32 位的无符号整数和 bool dtype 会被提升为 `tl.uint32`。其他 dtype 保持原样。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.cumprod(...)` 而不是 `cumprod(x, ...)`。
