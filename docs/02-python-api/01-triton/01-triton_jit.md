---
title: triton.jit
---

```python
triton.jit(fn: T)→ JITFunction[T]
triton.jit(*, version=None, repr: Callable | None = None, launch_metadata: Callable | None = None, do_not_specialize: Iterable[int | str] | None = None, do_not_specialize_on_alignment: Iterable[int | str] | None = None, debug: bool | None = None, noinline: bool | None = None)→ Callable[[T], JITFunction[T]]
```


使用 Triton 编译器的 JIT 编译函数的装饰器。


**注意：**


* 当调用 JIT 编译的函数时，如果参数具有 `.data_ptr()` 方法和 `.dtype` 属性，则会隐式转换为指针。

**注意：**

* 具名元组（Namedtuple）参数类可以将 `__triton_do_not_specialize__` 设置为一个由字段名组成的元组。这些字段以及它们嵌套的元组元素会跳过按取值特化（value specialization）与按对齐特化（alignment specialization）。显式的 `constexpr` 值仍会特化。

**注意：**

* 此函数将在 GPU 上编译和运行。它只能访问以下内容：
   * Python 原语，
   * Triton 包内的内置函数，
   * 此函数的参数，
   * 其他 JIT 编译的函数。

**参数：**

* **fn** (*Callable*) - 要进行 JIT 编译的函数
