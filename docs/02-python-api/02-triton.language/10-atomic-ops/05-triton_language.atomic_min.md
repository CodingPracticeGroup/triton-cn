---
title: triton_language.atomic_min
---

```python
triton.language.atomic_min(pointer, val, mask=None, sem=None, scope=None)
```

在由 `pointer` 指定的内存位置执行原子最小值操作。

返回原子操作执行前保存在 `pointer` 处的数据。

**参数：**

* **pointer**（*Block of dtype=triton.PointerDType*）- 要操作的内存位置。
* **val**（*Block of dtype=pointer.dtype.element_ty*）- 用于执行该原子操作的值。
* **sem**（*str, optional*）- 指定该操作的内存语义。可接受的值为 "acquire"、"release"、"acq_rel"（代表 "ACQUIRE_RELEASE"）与 "relaxed"。如果未提供，则函数默认使用 "acq_rel" 语义。
* **scope**（*str, optional*）- 定义可观察到该原子操作同步效果的线程范围。可接受的值为 "gpu"（默认）、"cta"（cooperative thread array，即线程块）与 "sys"（代表 "SYSTEM"）。默认值为 "gpu"。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.atomic_min(...)` 而不是 `atomic_min(x, ...)`。
