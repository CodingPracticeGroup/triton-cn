---
title: triton.language.advance
---

:::note
上游当前版本的 `triton.language` 文档列表已不再收录 `advance`；上游 `tl.load`/`tl.store` 的签名中也不再出现块指针相关的参数。该函数在上游源码中仍然存在，本页保留，供阅读旧代码时查阅。
:::

```python
triton.language.advance(base, offsets)
```
推进 1 个块指针。

**参数：**

* **base** - 要推进的块指针。
* **offsets** - 要推进的偏移量，是一个按维度划分的元组。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.advance(...)` 的方式而不是 `advance(x, ...)`。


