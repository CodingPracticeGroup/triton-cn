---
title: triton_language.dot
---

```python
triton.language.dot(input, other, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype)
```

返回 2 个块的矩阵乘积。

这 2 个块必须都是二维或三维的，并且具有兼容的内部维度。对于三维的块，`tl.dot` 执行批量矩阵乘积，其中每个块的第一维表示批量维度。

:::warning
使用 TF32 精度时，float32 输入可能在未经舍入的情况下被截断为 TF32 格式（19 位浮点），这可能使结果产生偏差。为了获得最佳结果，必须显式地舍入到 TF32，或者使用带 `round_f32_to_tf32=True` 的 `TensorDescriptor` 来加载数据。
:::

**参数：**

* **input**（*2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}*）- 第 1 个要相乘的张量。
* **other**（*2D or 3D tensor of scalar-type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}*）- 第 2 个要相乘的张量。
* **acc**（*2D or 3D tensor of scalar-type in {`float16`, `float32`, `int32`}*）- 累加器张量。如果不为 None，则将结果添加到该张量中。
* **input_precision**（*string. Available options for nvidia: `"tf32"`, `"tf32x3"`, `"ieee"`, `"bf16x3"`, `"bf16x6"`. Default: `"tf32"`. Available options for amd: `"ieee"`, `"bf16x3"`, `"bf16x6"`, (CDNA3 only) `"tf32"`.*）- 控制如何调用 Tensor Cores 进行 f32 x f32 运算。如果设备没有 Tensor Cores，或者输入不是 dtype f32，则此选项将被忽略。对于具有 Tensor Cores 的设备，默认精度为 tf32。
* **allow_tf32** - 已弃用。如果为 true，则 `input_precision` 被设置为 `"tf32"`。只能指定 `input_precision` 和 `allow_tf32` 中的 1 个（即至少 1 个必须为 `None`）。
