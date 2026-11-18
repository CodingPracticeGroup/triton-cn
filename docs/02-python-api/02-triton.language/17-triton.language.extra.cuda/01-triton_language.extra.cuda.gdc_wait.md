---
title: triton_language.extra.cuda.gdc_wait
---

```python
triton.language.extra.cuda.gdc_wait()
```


GDC wait 是一条阻塞指令，它会等待前一个内核中的所有指令执行完毕后才继续。这确保了在等待之前发生的所有内存操作对它之后的指令可见，例如：如果前一个内核写入了地址 "x"，那么在本内核中，该等待之后这些新值就是可见的。

在禁用程序化依赖启动（programmatic dependent launch）时，该指令也可以安全执行。

更多细节请参见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol 。
