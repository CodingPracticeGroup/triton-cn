---
title: triton_testing.do_bench
---

```python
triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode='mean')
```


对所提供的函数的运行时间进行基准测试。默认情况下，以单个 float 的形式返回 `fn` 的平均运行时间（以毫秒为单位）。


**参数****：**

* **fn** (*Callable*) - 用于基准测试的函数。

* **warmup** (*int*) - 预热时间（以毫秒为单位）。控制开始计时之前该函数运行的时长。

* **rep** (*int*) - 重复时间（以毫秒为单位）。控制被计时的这些运行的总时长。

* **grad_to_none** (*list[torch.Tensor]*, *optional*) - 在每次运行之前，将提供的张量的梯度重置为 None，以避免梯度累积影响计时。

* **quantiles** (*list[float]*, *optional*) - 如果提供，则返回运行时长的这些分位数，而不是汇总统计量。例如，`[0.2, 0.5, 0.8]` 返回第 20、第 50 与第 80 百分位的运行时长（单位为 ms）。设置该参数后，`return_mode` 会被忽略。

* **return_mode** (*str*) - 未设置 `quantiles` 时要返回的汇总统计量。`"mean"` 与 `"median"` 返回单个 float。`"min"` 与 `"max"` 分别返回最快与最慢的一次运行。`"all"` 返回每次运行时长（单位为 ms）构成的原始列表。默认为 `"mean"`。


**返回：**（*float | list[float]*）以毫秒为单位的运行时长：对于标量的 `return_mode` 返回单个 float；如果设置了 `quantiles` 或 `return_mode="all"`，则返回 float 列表。
