---
title: triton.testing
---

|Benchmark|这类由 perf_report 函数使用，用于通过简洁的 API 生成折线图|
|:----|:----|
|do_bench|对提供的函数进行运行时基准测试|
|do_bench_cudagraph|对提供的函数进行运行时基准测试|
|perf_report|标记一个用于基准测试的函数|
|assert_close|断言两个输入在一定公差范围内接近|
|do_bench_proton|使用 Proton 性能分析器对所提供的函数所发起的内核的运行时间进行基准测试|
|do_bench_cudagraph_proton|使用 Proton 性能分析器与 CUDA 图对所提供的函数所发起的内核的运行时间进行基准测试|


