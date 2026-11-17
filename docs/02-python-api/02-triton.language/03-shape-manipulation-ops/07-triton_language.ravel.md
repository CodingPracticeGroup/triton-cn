---
title: triton_language.ravel
---


```python
triton.language.ravel(x, can_reorder=False)
```


返回 `x` 的连续扁平视图。


**参数：**

* **x** (*Block*) - 输入张量。
* **can_reorder** - 已忽略。为向后兼容而保留。

这个函数也可作为 `tensor` 的成员函数调用，例如 `x.ravel(...)` 代替 `ravel(x, ...)`。


