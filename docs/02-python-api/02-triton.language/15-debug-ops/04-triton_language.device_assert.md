---
title: triton_language.device_assert
---

```python
triton.language.device_assert(cond, msg="", mask=None)
```

在运行时从设备端断言该条件。要让这个断言生效，需要把环境变量 `TRITON_DEBUG` 设置为 `0` 以外的值。

使用 Python 的 `assert` 语句等同于调用这个函数，只不过第 2 个参数必须提供且必须是 1 个字符串，例如 `assert pid == 0, "pid != 0"`。这个 `assert` 语句同样必须设置该环境变量才会生效。

```python
tl.device_assert(pid == 0)
assert pid == 0, f"pid != 0"
```

**参数：**

* **cond** - 要断言的条件。必须是 1 个布尔张量。
* **msg** - 断言失败时要打印的消息。必须是 1 个字符串字面值。
