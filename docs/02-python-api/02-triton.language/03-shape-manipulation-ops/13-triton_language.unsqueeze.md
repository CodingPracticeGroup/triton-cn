---
title: triton_language.unsqueeze
---

```python
triton.language.unsqueeze(x, dim)
```


在索引 `dim` 处向 `x` 插入一个长度为 1 的维度。


这是 `squeeze` 的逆操作，并且与 `expand_dims` 等价。


**参数：**

* **x** - 输入张量。
* **dim** (*int*) - 在其处插入新维度的索引。
