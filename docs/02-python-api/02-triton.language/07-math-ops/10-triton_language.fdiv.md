---
title: triton_language.fdiv
---

```python
triton.language.fdiv(x, y, ieee_rounding, approx)
```

计算 `x` 除以 `y` 的逐元素除法。

**参数：**

* **x** - 分子。
* **y** - 分母。
* **ieee_rounding** - 若为 True，使用 IEEE 就近舍入（round-to-nearest）除法。
* **approx** - 若为 True，允许对 float32 操作数进行近似除法。在 NVIDIA 上使用 `div.approx.f32`，当分母量级处于 [2**-126, 2**126] 内时最大误差为 2 ULP；该范围之外的结果可能与全范围除法不同。在 AMD 上使用 `llvm.amdgcn.fdiv.fast`，最大误差为 2.5 ULP，且不支持次正规（denormal）输入或结果。不能与 `ieee_rounding=True` 组合使用。
