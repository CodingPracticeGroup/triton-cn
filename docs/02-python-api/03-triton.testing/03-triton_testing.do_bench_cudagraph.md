---
title: triton_testing.do_bench_cudagraph
---

```python
triton.testing.do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean')
```


对提供的函数进行运行时间基准测试。


**参数****：**

* **fn** (*Callable*) - 用于基准测试的函数。

* **rep** (*int*) - 重复时间（以毫秒为单位）。

* **grad_to_none** (*torch.tensor*, *optional*) - 将提供的张量的梯度重置为 None。

* **return_mode** (*str*) - 要返回的统计量度。选项为「min」、「max」、「mean」、「median」或「all」。默认为「mean」。


**返回：**（*float | list[float]*）以毫秒为单位的运行时长：对于标量的 `return_mode` 返回单个 float；如果设置了 `quantiles` 或 `return_mode="all"`，则返回 float 列表。
