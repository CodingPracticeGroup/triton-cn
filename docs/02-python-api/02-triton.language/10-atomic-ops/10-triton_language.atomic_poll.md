---
title: triton_language.atomic_poll
---

```python
triton.language.atomic_poll(pointer, expected_value, sem=None, scope=None, timeout_ns=None)
```

等待 `pointer` 处的值等于 `expected_value`。

这会对每个指定的指针自旋等待，直到其值等于预期值，或者该操作超时。该 block（线程块）会等待所有轮询结束。超时的元素返回 false，且不获得任何结果。

**参数：**

* **pointer**（*triton.PointerDType*）- 指向 16 位、32 位或 64 位整数的指针，或由这类指针组成的块。
* **expected_value**（*pointer.dtype.element_ty*）- 结束每轮轮询循环的值，会被广播到 `pointer` 的形状。
* **sem**（*str, optional*）- 指定一次成功的轮询是否具有 acquire 语义。可接受的值为 "acquire"（默认）与 "relaxed"。
* **scope**（*str, optional*）- 定义可观察到该轮询同步效果的线程范围。可接受的值为 "gpu"（默认）、"cta"（cooperative thread array，即线程块）与 "sys"（system）。
* **timeout_ns**（*int, optional*）- 整个操作共享的轮询时间预算，由 GPU 全局计时器以纳秒为单位计量。如果省略，则轮询没有超时限制。即使超时值为 0，每个元素也至少会被加载一次。

**返回：**

形状与 `pointer` 相同的布尔值；对于观察到预期值的元素为 true，若其超时先到则为 false。返回类型为 `triton.language.tensor`。

这个函数也可作为 `tensor` 的成员函数调用，使用 `x.atomic_poll(...)` 而不是 `atomic_poll(x, ...)`。
