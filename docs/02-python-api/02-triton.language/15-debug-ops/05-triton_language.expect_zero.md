---
title: triton_language.expect_zero
---

```python
triton.language.expect_zero(x, mask)
```

标记那些预期已经下溢为 0 的值。

在常规编译中，它会保留 `x`。Debug 构建会在 `mask` 为 true 的位置断言 `x` 为 0。在浮点检查器（FpSan）下，它变成 `where(mask, 0, x)`，因此经过插桩的执行会观察到预期的浮点下溢。

**参数：**

* **x** - 在 FpSan 模式之外需要保留的值。
* **mask** - 预期 `x` 为 0 的位置。
