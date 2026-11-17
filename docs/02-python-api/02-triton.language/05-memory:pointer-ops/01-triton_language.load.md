---
title: triton_language.load
---

```python
triton.language.load(pointer, mask, other, *, cache_modifier, eviction_policy, volatile)
```

返回 1 个数据张量，其值从由 `pointer` 所定义的内存位置处加载：

1. 如果 `pointer` 是单元素指针，则加载 1 个标量。在这种情况下：

   * `mask` 和 `other` 必须也是标量，
   * `other` 会隐式地转换为 `pointer.dtype.element_ty` 类型。

2. 如果 `pointer` 是 1 个 N 维指针张量，则加载 1 个 N 维张量。在这种情况下：

   * `mask` 和 `other` 会被隐式地广播到 `pointer.shape`，
   * `other` 会隐式地转换为 `pointer.dtype.element_ty` 类型。

**参数：**

* **pointer**（*`triton.PointerType`, or block of `dtype=triton.PointerType`*）- 指向要加载的数据的指针。
* **mask**（*Block of `triton.int1`, optional*）- 如果 `mask[idx]` 为 false，则不加载地址 `pointer[idx]` 处的数据。
* **other**（*Block, optional*）- 如果 `mask[idx]` 为 false，则返回 `other[idx]`。如果 `other` 为 `None`，则被掩码遮蔽处的值未定义。
* **cache_modifier**（*str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1), and ".cv" means don’t cache and fetch again. see [cache operator](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) for more details.*）- 在 NVIDIA PTX 中更改缓存选项。
* **eviction_policy**（*str, optional*）- 在 NVIDIA PTX 中更改驱逐策略。
* **volatile**（*bool, optional*）- 在 NVIDIA PTX 中更改易失性选项。
