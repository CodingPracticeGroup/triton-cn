---
title: triton_language.where
---

```python
triton.language.where(condition, x, y)
```

根据 `condition` 返回由 `x` 或 `y` 中的元素组成的张量。

注意：无论 `condition` 的取值如何，`x` 和 `y` 都会被求值。

如果想避免非预期的内存操作，请改用 `triton.load` 与 `triton.store` 中的 `mask` 参数。

`x` 和 `y` 的形状都会被广播到 `condition` 的形状。`x` 和 `y` 必须具有相同的数据类型。

**参数：**

* **condition**（*Block of triton.bool*）- 为 True（非零）时取 x，否则取 y。
* **x** - 在 `condition` 为 True 的索引处选用的值。
* **y** - 在 `condition` 为 False 的索引处选用的值。
