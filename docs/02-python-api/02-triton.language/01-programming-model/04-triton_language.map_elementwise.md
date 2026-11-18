---
title: triton_language.map_elementwise
---

```python
triton.language.map_elementwise(scalar_fn, *args, pack=1)
```

将 1 个标量函数映射到张量上。

输入张量 `args` 会被隐式广播到相同的形状。

这可能有助于在张量的单个元素上使用控制流，例如有 1 个分支开销更大的多分支函数。使用 `tl.where` 时你必须计算分支两侧的取值，而使用 if 时只执行其中一侧。

```python
@triton.jit
def selu_scalar(x, alpha):
    if x > 0:
        return a
    else:
        return alpha * (tl.exp(x) - 1)

@triton.jit
def selu(x, alpha):
    return tl.map_elementwise(selu_scalar, x, alpha)
```

**参数：**

* **scalar_fn** - 要被映射的函数。该函数及其调用的函数必须是纯函数，`assume`、`device_assert` 与 `device_print` 除外。它们在降低（lower）时不得需要共享内存暂存空间。
* **pack** - 1 次函数调用处理的元素数量。

**返回：** 1 个张量或 1 组张量构成的元组，取决于被映射的函数。
