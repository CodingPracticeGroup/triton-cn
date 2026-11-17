---
title: triton_language.maximum
---

```python
triton.language.maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE)
```

计算 `x` 与 `y` 的逐元素最大值。

**参数：**

* **x**（*Block*）- 第一个输入张量。
* **y**（*Block*）- 第二个输入张量。
* **propagate_nan**（*tl.PropagateNan*）- 是否传播 NaN 值。

> 另请参阅  
> tl.PropagateNan
