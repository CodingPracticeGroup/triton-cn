---
title: triton_language.gather
---

```python
triton.language.gather(src, index, axis)
```

沿指定维度从张量中收集（gather）元素。

**参数：**

* **src** (*Tensor*) - 源张量。
* **index** (*Tensor*) - 索引张量。
* **axis** (*int*) - 要沿其进行收集的维度。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.gather(...)` 而不是 `gather(x, ...)`。
