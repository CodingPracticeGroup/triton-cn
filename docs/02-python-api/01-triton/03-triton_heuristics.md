---
title: triton.heuristics
---


```python
triton.heuristics(values)
```


用于指定如何计算某些元参数值的装饰器。这在自动调优成本过高或不适用的情况下非常有用。


```python
# smallest power-of-two >= x_size  最小的 2 的幂 >= x_size
@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...
```

**参数：**

* **values** (*dict[str, Callable[[dict[str, Any]], Any]]*) - 包含元参数名称和计算元参数值的函数的字典。每个这样的函数都以内核全部参数构成的字典作为输入，该字典以参数名为键。
