---
title: triton_language.store
---

```python
triton.language.store(pointer, value, mask, *, cache_modifier, eviction_policy)
```

将数据张量存储到由 `pointer` 定义的内存位置。

1. 如果 `pointer` 是单元素指针，则存储 1 个标量。在这种情况下：

   * `mask` 必须也是标量。

2. 如果 `pointer` 是 1 个 N 维指针张量，则存储 1 个 N 维块。在这种情况下：

   * `mask` 会被隐式地广播到 `pointer.shape`。

`value` 会被隐式地广播到 `pointer.shape`，并转换为 `pointer.dtype.element_ty` 类型。

**参数：**

* **pointer**（*`triton.PointerType`, or block of `dtype=triton.PointerType`*）- 存储 `value` 中各元素的内存位置。
* **value**（*Block*）- 要存储的元素张量。
* **mask**（*Block of triton.int1, optional*）- 如果 `mask[idx]` 为 false，则不将 `value[idx]` 存储到 `pointer[idx]` 处。
* **cache_modifier**（*str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt" stands for cache write-through, see [cache operator](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) for more details.*）- 在 NVIDIA PTX 中更改缓存选项。
* **eviction_policy**（*str, optional, should be one of {"", "evict_first", "evict_last"}*）- 在 NVIDIA PTX 中更改驱逐策略。
