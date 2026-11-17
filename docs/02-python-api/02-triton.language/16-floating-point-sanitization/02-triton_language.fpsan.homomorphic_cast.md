---
title: triton_language.fpsan.homomorphic_cast
---

```python
triton.language.fpsan.homomorphic_cast(input, dtype: core.dtype, fp_downcast_rounding: str | None = None)
```


转换到同宽度或更窄的浮点类型，并保留 FpSan 的载荷代数。

启用 FpSan 时，保留输入载荷的低位。这保留了加法与乘法，使一次向下转换可以穿过算术运算移动而不改变签名。使用 `barrier` 可以标出一个边界，跨越该边界移动这次转换应当是可观察的。

未启用 FpSan 时，执行的是一次普通的浮点类型转换。不支持加宽（widening）与整数转换。输入的形状与布局会被保留。


**参数：**

* **input** - 一个浮点标量或张量。

* **dtype** - 一个位宽不大于输入的浮点目标类型。

* **fp_downcast_rounding** - 普通执行时的舍入模式：`"rtne"`（默认）或 `"rtz"`。仅对更窄的目标类型有效；FpSan 的载荷截断与此选择无关。
