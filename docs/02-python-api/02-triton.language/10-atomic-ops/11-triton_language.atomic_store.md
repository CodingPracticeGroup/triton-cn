---
title: triton_language.atomic_store
---

```python
triton.language.atomic_store(pointer, val, mask=None, sem=None, scope=None)
```

将 `val` 原子地存储到由 `pointer` 指定的内存位置。

**参数：**

* **pointer**（*Block of dtype=triton.PointerDType*）- 要存储到的内存位置。
* **val**（*Block of dtype=pointer.dtype.element_ty*）- 要存储的值。
* **mask**（*Block of triton.int1, optional*）- 如果 `mask[idx]` 为 false，则不执行对应位置的存储。
* **sem**（*str, optional*）- 指定内存语义。可接受的值为 "release"（默认）与 "relaxed"。
* **scope**（*str, optional*）- 定义可观察到同步效果的线程范围。可接受的值为 "gpu"（默认）、"cta" 与 "sys"。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.atomic_store(...)` 而不是 `atomic_store(x, ...)`。
