---
title: 程序化依赖启动 (Programmatic Dependent Launch)
---

本脚本以使用 Triton 编写的向量相加示例为基础，演示程序化依赖启动（programmatic dependent launch，PDL）的用法。

关于 programmatic dependent launch 的 CUDA 参考，请参见 https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization。

关于 programmatic dependent launch 的 PTX 参考，请参见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol。

```bash
python 11-programmatic-dependent-launch.py
```

```python
import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_pdl():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# In this example
# 在本示例中
@triton.jit
def add_kernel(x_ptr,  #
               y_ptr,  #
               output_ptr,  #
               n_elements,  #
               BLOCK_SIZE: tl.constexpr,  #
               USE_GDC: tl.constexpr,  #
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        # GDC wait waits for ALL programs in the the prior kernel to complete before continuing.
        # GDC wait 会等待前一个内核中的所有程序（ALL）完成之后才继续。
        # This ensures any memory operations happen before the wait in program order,
        # 这确保任何内存操作按程序顺序发生在该等待之前，
        # e.g. if the prior kernel writes to x or y the new values will be visible.
        # 例如，如果前一个内核写入了 x 或 y，新值将是可见的。
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        # GDC launch dependents 向运行时系统发出提示，让其启动依赖内核。
        # These dependent kernels must also be launched with PDL enabled.
        # 这些依赖内核也必须以启用 PDL 的方式启动。
        # Once GDC launch has been issued by ALL programs or
        # 一旦所有程序都发出了 GDC launch，或者
        # programs have finished, the dependent grid can begin if there are enough resources.
        # 程序已经结束，那么在有足够资源的情况下，依赖 grid 可以开始。
        # Note: this by itself provides no additional memory-ordering guarentees, unlike `gdc_wait`
        # 注意：与 `gdc_wait` 不同，这本身不提供额外的内存顺序保证
        tl.extra.cuda.gdc_launch_dependents()
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, launch_pdl: bool = True):
    output = torch.empty_like(x)
    assert x.device == y.device and output.device == x.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=1024,
        USE_GDC=launch_pdl,  # set constexpr in kernel to use grid dependence control
        # 在内核中将 constexpr 置位，以使用网格依赖控制（grid dependence control）
        launch_pdl=launch_pdl,  # launch kernel with PDL flag set enabled
        # 启动内核时设置并启用 PDL 标志
    )
    return output


def validate(n_elements):
    x = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    y = torch.rand(n_elements, device="cuda", dtype=torch.float32)

    torch_result = x + y
    add_result = add(x, y)

    torch_vs_add = "✅" if torch.allclose(torch_result, add_result, atol=1.0) else "❌"
    print(f"Number of Elements={n_elements} verification naive vs: ", end="")
    print(f"add: {torch_vs_add}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(23, 28, 1)],
        x_log=False,
        line_arg="provider",
        line_vals=["pdl-fp32", "fp32"],
        line_names=["PDL", "No PDL"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel='GB/s',
        plot_name="pdl-performance",
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    fn = lambda: add(x, y, "pdl" in provider)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles, rep=100)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    if supports_pdl():
        validate(1024)
        benchmark.run(print_data=True, show_plots=True, save_path=".")
    else:
        print("PDL is not supported on this device")
```
