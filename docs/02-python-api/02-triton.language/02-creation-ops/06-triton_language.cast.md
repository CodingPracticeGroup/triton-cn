---
title: triton_language.cast
---

```python
triton.language.cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False)
```

将张量转换为指定的 `dtype`。

**参数：**

* **dtype**（*tl.dtype*）- 目标数据类型。
* **fp_downcast_rounding**（*str, optional*）- 浮点值向下转换时的舍入模式。仅当 `self` 是浮点张量且 `dtype` 是位宽更小的浮点类型时才会使用该参数。支持的取值为 `"rtne"`（round to nearest, ties to even，四舍五入到最接近的值，相同幅度时取偶数）与 `"rtz"`（round towards zero，向零舍入）。
* **bitcast**（*bool, optional*）- 如果为 true，则对张量执行 `bitcast`，将其按位重新解释为给定的 `dtype`，而不是进行数值转换。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.cast(...)` 而不是 `cast(x, ...)`。
