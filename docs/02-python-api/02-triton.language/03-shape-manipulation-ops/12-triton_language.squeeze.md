---
title: triton_language.squeeze
---

```python
triton.language.squeeze(x, dim: core.constexpr)
```


从 `x` 中移除索引 `dim` 处长度为 1 的维度。


被选中的维度必须长度为 1（在编译时检查）。这是 `unsqueeze` 的逆操作。


**参数：**

* **x** - 输入张量。
* **dim** (*int*) - 要移除的维度的索引。
