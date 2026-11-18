---
title: triton_language.tensor_descriptor
---

```python
triton.language.tensor_descriptor(handle, shape, strides, block_type, fp6_format=None)
```


表示全局内存中一个张量的描述符。


**参数：**

* **handle** - IR 句柄。
* **shape** (*List\[tensor\]*) - 全局形状。
* **strides** (*List\[tensor\]*) - 步长。
* **block_type** (*block_type*) - 块类型。
* **fp6_format** - 上游未说明该参数。

`__init__` 不会被用户代码调用。


说明：上游文档站的这一条目由 Sphinx autodoc 从 `python/triton/language/core.py` 中的类 `tensor_descriptor` 生成。类与 `__init__` 的 docstring 只有上面这两句，没有逐个参数的说明；上面的参数含义是按 `__init__` 的赋值与该处源码注释给出的，不是上游文档原文。张量描述符的用法请见 `make_tensor_descriptor` 与 `load_tensor_descriptor`、`store_tensor_descriptor`。
