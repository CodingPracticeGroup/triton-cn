---
title: triton_language.sub
---

```python
triton.language.sub(x, y, sanitize_overflow: constexpr = True)
```

计算 `x` 与 `y` 的逐元素差。

这是 `-` 运算符的函数形式。

**参数：**

* **x**（*Block*）- 第一个输入张量。
* **y**（*Block*）- 第二个输入张量。
* **sanitize_overflow**（*bool*）- 当编译时启用了溢出净化（overflow sanitization）时，插入整数溢出检查；设为 `False` 则生成普通的回绕算术（wrapping arithmetic）。对浮点操作数无效果。
