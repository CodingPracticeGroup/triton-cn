---
title: triton_language.make_block_ptr
---

:::note
上游当前版本的 `triton.language` 文档列表已不再收录 `make_block_ptr`；上游 `tl.load`/`tl.store` 的签名中也不再出现块指针相关的参数。该函数在上游源码中仍然存在，本页保留，供阅读旧代码时查阅。
:::

```python
triton.language.make_block_ptr(base, shape, strides, offsets, block_shape, order)
```


返回指向父张量中 1 个块的指针。


**参数：**

* **base** - 父张量的基础指针。
* **shape** - 父张量的形状。
* **strides** - 父张量的步幅。
* **offsets** - 块的偏移量。
* **block_shape** - 块的形状。
* **order** - 原始数据格式的顺序。

