---
title: triton_testing.do_bench_proton
---

```python
triton.testing.do_bench_proton(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode='mean')
```


使用 Proton 性能分析器（profiler）对所提供的函数所发起的内核的运行时间进行基准测试。


对于受 CPU 开销影响的短内核，测得的运行时间通常比 `do_bench` 更准确。注意，与 `do_bench` 相比，该函数有若干限制：

* 它不测量内核以外的 GPU 操作（例如内存拷贝、同步等）。

* 它仅支持 AMD 与 NVIDIA GPU。


**参数****：**

* **fn** (*Callable*) - 用于基准测试的函数。

* **warmup** (*int*) - 预热时间（以毫秒为单位）。

* **rep** (*int*) - 重复时间（以毫秒为单位）。

* **grad_to_none** (*torch.Tensor*, *optional*) - 将提供的张量（一个或多个）的梯度重置为 `None`。

* **quantiles** (*list[float]*, *optional*) - 除中位数外要返回的性能百分位数。

* **return_mode** (*str*) - 要返回的统计量度。选项为「min」、「max」、「mean」、「median」或「all」。默认为「mean」。


**返回：**（*float | list[float]*）以毫秒为单位的运行时长：对于标量的 `return_mode` 返回单个 float；如果设置了 `quantiles` 或 `return_mode="all"`，则返回 float 列表。
