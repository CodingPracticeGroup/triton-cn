---
title: triton_language.clmul
---

```python
triton.language.clmul(x, y)
```

计算 `x` 与 `y` 的逐元素无进位乘法（carry-less multiplication）。

操作数被解释为 GF(2) 上的二元多项式，因此乘法中的加法被替换为 XOR。结果包含乘积的低 N 位，其中 N 为整数类型提升后的位宽。带符号操作数按其位模式解释。输入必须为整数，并遵循常规的广播与类型提升规则。

**参数：**

* **x** - 第一个输入。
* **y** - 第二个输入。
