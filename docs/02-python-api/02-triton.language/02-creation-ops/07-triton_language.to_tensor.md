---
title: triton_language.to_tensor
---

```python
triton.language.to_tensor(x)
```

把一个 Python 标量转换为 0 维的 `tensor`。

如果 `x` 已经是一个 `tensor`，则原样返回它。结果的 dtype 由该值推断：Python 的 `bool` 会变成 `tl.int1`；`int` 会变成 `tl.int32`、`tl.uint32`、`tl.int64` 或 `tl.uint64` 中能够表示该值的最小类型；`float` 会变成 `tl.float32`（当它超出 `float32` 的取值范围时为 `tl.float64`）。

**参数：**

* **x** - 任意数值。
