---
title: triton_language.atomic_load
---

```python
triton.language.atomic_load(pointer, mask=None, sem=None, scope=None)
```

从由 `pointer` 指定的内存位置原子地加载数据。

**参数：**

* **pointer**（*Block of dtype=triton.PointerDType*）- 要从中加载数据的内存位置。
* **mask**（*Block of triton.int1, optional*）- 如果 `mask[idx]` 为 false，则不执行对应位置的加载，其结果未指定。
* **sem**（*str, optional*）- 指定内存语义。可接受的值为 "acquire"（默认）与 "relaxed"。
* **scope**（*str, optional*）- 定义可观察到同步效果的线程范围。可接受的值为 "gpu"（默认）、"cta" 与 "sys"。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.atomic_load(...)` 而不是 `atomic_load(x, ...)`。
