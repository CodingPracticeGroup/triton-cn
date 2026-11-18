---
title: triton_language.fpsan.barrier
---

```python
triton.language.fpsan.barrier(input)
```


把一个中间表达式标记为对浮点检查（floating-point sanitization）有意义。

启用 FpSan 时，对输入的载荷施加一个确定性的、非同态的变换。例如，`barrier(a * b) * c` 可能不同于 `a * barrier(b * c)`，`barrier(a * b) + c` 也可能不同于 `fma(a, b, c)`。未启用 FpSan 时，原样返回输入。

高位的载荷位会被混合进低位，因此把一次 `homomorphic_cast` 移动到 barrier 的另一侧，可能改变签名。

这不是同步屏障或优化屏障，在普通执行中它也不会强制浮点舍入或阻止 FMA 融合。


**参数：**

* **input** - 一个浮点标量或张量。


**返回：** 与 `input` 具有相同类型、形状与布局的值。
