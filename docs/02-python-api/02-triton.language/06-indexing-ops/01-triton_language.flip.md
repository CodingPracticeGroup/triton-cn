---
title: triton_language.flip
---

```python
triton.language.flip(x, dim=None)
```

沿维度 `dim` 翻转张量 `x`。

**参数：**

* **x**（*Block*）- 第一个输入张量。
* **dim**（*int*）- 要沿其进行翻转的维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.flip(...)` 而不是 `flip(x, ...)`。
