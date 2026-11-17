---
title: triton_language.dot_scaled
---

```python
triton.language.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=float32)
```

返回 2 个采用微缩放（microscaling）格式的块的矩阵乘积。

lhs 和 rhs 使用此处所描述的微缩放格式：
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

软件模拟使得该操作也可以用于没有原生微缩放运算支持的硬件架构。目前对于这种情况，微缩放的 lhs/rhs 会先向上转换（upcast）为 `bf16` 元素类型再进行 dot 计算，但有一个例外：具体到 AMD CDNA3，如果其中一个输入是 `fp16` 元素类型，则另一个输入也会向上转换为 `fp16` 元素类型。此行为是实验性的，将来可能会改变。

**参数：**

* **lhs**（*2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.*）- 第 1 个要相乘的张量。
* **lhs_scale**（*e8m0 type represented as an uint8 tensor, or None.*）- lhs 张量的缩放因子。当 lhs 为 [M, K] 时，形状应为 [M, K//group_size]，其中如果缩放因子类型为 `e8m0`，则 group_size 为 32。
* **lhs_format**（*str*）- lhs 张量的格式。可用格式：{`e2m1`, `e4m3`, `e5m2`, `bf16`, `fp16`}。
* **rhs**（*2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.*）- 第 2 个要相乘的张量。
* **rhs_scale**（*e8m0 type represented as an uint8 tensor, or None.*）- rhs 张量的缩放因子。当 rhs 为 [K, N] 时，形状应为 [N, K//group_size]。重要：不要转置 rhs_scale。
* **rhs_format**（*str*）- rhs 张量的格式。可用格式：{`e2m1`, `e4m3`, `e5m2`, `bf16`, `fp16`}。
* **acc** - 累加器张量。如果不为 None，则将结果添加到该张量中。
* **lhs_k_pack**（*bool, optional*）- 如果为 false，则 lhs 张量沿 M 维度被打包进 uint8。
* **rhs_k_pack**（*bool, optional*）- 如果为 false，则 rhs 张量沿 N 维度被打包进 uint8。
