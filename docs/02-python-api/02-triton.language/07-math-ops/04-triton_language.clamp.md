---
title: triton_language.clamp
---

```python
triton.language.clamp(x, min, max, propagate_nan=PropagateNan.NONE)
```

将输入张量 `x` 钳制在 [min, max] 范围内。当 `min` > `max` 时，行为未定义。

**参数：**

* **x** (*Block*) - 输入张量。
* **min** (*Block*) - 钳制操作的下界值。
* **max** (*Block*) - 钳制操作的上界值。
* **propagate_nan** (*tl.PropagateNan*) - 是否传播 NaN 值。仅对张量 `x` 生效。如果 `min` 或 `max` 中任一值为 NaN，则结果未定义。

> 另请参阅  
> tl.PropagateNan
