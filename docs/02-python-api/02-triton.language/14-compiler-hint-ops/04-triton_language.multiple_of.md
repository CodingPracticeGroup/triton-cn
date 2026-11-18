---
title: triton_language.multiple_of
---

```python
triton.language.multiple_of(input, values)
```

告知编译器：`values[d]` 是能整除 `input` 沿维度 `d` 的每个连续组的第 1 个元素的、最大的 2 的幂（连续组的定义见 `max_contiguous`）。`values` 必须为 `input` 的每个维度各提供 1 个条目。

对于连续度为 1 的 1 维输入，这等同于说 `input` 的每个元素都是 `values[0]` 的倍数。例如，如果 `values` 是 `[16]` 且 `input` 是 `[64, 80, 96, 112]`，该提示是有效的，因为 16 能整除 64。

该提示可以启用依赖对齐的优化，例如向量化的内存访问。
