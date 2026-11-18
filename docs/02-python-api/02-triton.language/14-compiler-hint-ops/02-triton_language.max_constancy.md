---
title: triton_language.max_constancy
---

```python
triton.language.max_constancy(input, values)
```

告知编译器：`input` 沿维度 `d` 的元素构成长度为 `values[d]` 的常量组。`values` 必须为 `input` 的每个维度各提供 1 个条目，且每个条目都必须是 2 的幂。

含 `N` 个元素、常量度（constancy）为 `C` 的 1 维数组会被看作 `N/C` 段，每段是 `C` 个相同的值。例如，如果 `values` 是 `[4]`，那么数组 `[0, 0, 0, 0, 1, 1, 1, 1]` 满足该提示。
