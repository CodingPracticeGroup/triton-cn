---
title: triton_language.extra.cuda.gdc_launch_dependents
---

```python
triton.language.extra.cuda.gdc_launch_dependents()
```


该操作在以程序化依赖启动（programmatic dependent launch）启动时发出这样的信号：一旦当前内核中的所有程序都调用了该函数或已经完成，下一个程序就可以启动。

对该函数的重复调用，在首次调用之后不再产生任何效果；程序员应当把首次调用视作向运行时系统发出的、让其启动下一个内核的提示。

在禁用程序化依赖启动时，该指令也可以安全执行。

更多细节请参见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol 。
