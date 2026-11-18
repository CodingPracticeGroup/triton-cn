---
title: triton_language.device_print
---

```python
triton.language.device_print(prefix, *args, hex=False)
```

在运行时从设备端打印数值。字符串格式化对运行时的值不起作用，因此你要把想打印的值作为参数提供。第 1 个值必须是字符串，其后的所有值必须是标量或张量。

调用 Python 内置的 `print` 等同于调用这个函数，并且参数的要求与本函数一致（而不是普通 `print` 的要求）。

```python
tl.device_print("pid", pid)
print("pid", pid)
```

在 CUDA 中，printf 的输出通过一个大小有限的缓冲区（buffer）流式传输（在一台主机上，我们测得默认大小为 6912 KiB，但该取值在不同 GPU 与 CUDA 版本之间可能不一致）。如果你发现有些 printf 被丢弃了，可以调用下面的接口增大缓冲区大小。

```python
triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)
```

如果你在运行过使用 printf 的内核之后尝试修改这个值，CUDA 可能会引发错误。这里设置的值可能只影响当前设备（因此如果你有多个 GPU，就需要多次调用）。

**参数：**

* **prefix** - 在数值之前打印的前缀。必须是字符串字面值。
* **args** - 要打印的值，可以是任何张量或标量。
* **hex** - 以十六进制打印整数，以十六进制浮点记法打印浮点值。
