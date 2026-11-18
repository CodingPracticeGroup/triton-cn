---
title: triton_language.max_contiguous
---

```python
triton.language.max_contiguous(input, values)
```

告知编译器：`input` 沿维度 `d` 的元素构成长度为 `values[d]` 的连续组。`values` 必须为 `input` 的每个维度各提供 1 个条目，且每个条目都必须是 2 的幂。

含 `N` 个元素、连续度（contiguity）为 `C` 的 1 维数组会被看作 `N/C` 段、每段 `C` 个整数，其中同 1 段内的整数在数值上顺序连续。例如，如果 `values` 是 `[4]`，那么数组 `[0, 1, 2, 3, 8, 9, 10, 11]` 满足该提示，因为它由 2 段各 4 个连续值组成。

与 `multiple_of` 一起使用时，这类提示可以支持对连续、对齐的区域进行向量化的加载与存储。
