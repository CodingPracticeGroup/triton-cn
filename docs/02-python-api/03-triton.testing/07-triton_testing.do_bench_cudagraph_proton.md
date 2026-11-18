---
title: triton_testing.do_bench_cudagraph_proton
---

```python
triton.testing.do_bench_cudagraph_proton(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean')
```


使用 Proton 性能分析器（profiler）与 CUDA 图（CUDA graph）对所提供的函数所发起的内核的运行时间进行基准测试。
该函数与 `do_bench_cudagraph` 类似，同样是重放包含所提供函数多次迭代的 CUDA 图来避开 CPU 开销，
但它使用 Proton 性能分析器测量图中每个内核的运行时长，而不是用 CUDA 事件测量整个图的总运行时长。
这让我们能更细粒度地测量内核运行时长，并把缓存清除排除在测量之外。
注意，与 `do_bench_cudagraph` 相比，该函数有若干限制：

* 它不测量内核以外的 GPU 操作（例如内存拷贝、同步等）。

* 它仅支持 NVIDIA GPU。AMD GPU 的支持仍是待办事项（TODO）。


**参数****：**

* **fn** (*Callable*) - 用于基准测试的函数。

* **rep** (*int*) - 重复时间（以毫秒为单位）。

* **grad_to_none** (*torch.tensor*, *optional*) - 将提供的张量的梯度重置为 None。

* **return_mode** (*str*) - 要返回的统计量度。选项为「min」、「max」、「mean」、「median」或「all」。默认为「mean」。


**返回：**（*float | list[float]*）以毫秒为单位的运行时长：对于标量的 `return_mode` 返回单个 float；如果设置了 `quantiles` 或 `return_mode="all"`，则返回 float 列表。
