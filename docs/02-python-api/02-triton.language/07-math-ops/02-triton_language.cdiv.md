---
title: triton_language.cdiv
---

```python
triton.language.cdiv(x, div)
```


计算 `x` 除以 `div` 的向上取整除法（ceiling division）。


**参数：**

* **x** (*Block*) - 被除数。
* **div** (*Block*) - 除数。


:::note
本条目在文档站中的名字是 `triton.language.cdiv`，实现在 `python/triton/language/standard.py`。
另有一个同名但不同的函数 `triton.cdiv(x, y)`（定义在 `python/triton/__init__.py`，作用于 Python
整数），二者不是同一个对象。
:::

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.cdiv(...)` 而不是 `cdiv(x, ...)`。
