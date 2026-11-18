---
title: 向量相加
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/YSztKYdMWSL)

在本教程中，你将使用 Triton 编写一个简单的向量相加 (vector addition) 程序。

你将了解：

- Triton 的基本编程模型
- 用于定义 Triton 内核的 `triton.jit` 装饰器 (decorator)
- 验证和基准测试自定义算子与原生参考实现的最佳实践

## 计算内核

```python
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               # 指向第一个输入向量的指针。
               y_ptr,  # *Pointer* to second input vector.
               # 指向第二个输入向量的指针。
               output_ptr,  # *Pointer* to output vector.
               # 指向输出向量的指针。
               n_elements,  # Size of the vector.
               # 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # 每个程序应处理的元素数量。
               # NOTE: `constexpr` so it can be used as a shape value.
               # 注意：`constexpr` 因此它可以用作形状值。
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    # 有多个“程序”处理不同的数据。需要确定的是哪一个程序：
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # 使用 1D 启动网格，因此轴为 0。
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    # 该程序将处理相对初始数据偏移的输入。
    # 例如，如果你有一个长度为 256、块大小为 64 的向量，各程序将分别访问 [0:64, 64:128, 128:192, 192:256] 的元素。
    # 注意 offsets 是指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    # 创建掩码以防止内存操作超出边界访问。
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    # 从 DRAM 加载 x 和 y，如果输入不是块大小的整数倍，则屏蔽掉任何多余的元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    # 将 x + y 写回 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)
```

创建一个辅助函数从而： (1) 生成 `z` 张量，(2) 用适当的 grid/block sizes 将上述内核加入队列：

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    # 需要预分配输出。
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # SPMD 启动网格表示并行运行的内核实例的数量。
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # 它类似于 CUDA 启动网格。它可以是 Tuple[int]，也可以是 Callable(metaparameters) -> Tuple[int]。
    # In this case, we use a 1D grid where the size is the number of blocks:
    # 在这种情况下，使用 1D 网格，其中大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    # 注意：
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    #  - 每个 torch.tensor 对象都会隐式转换为其第一个元素的指针。
    #  - `triton.jit` 函数可以通过启动网格索引来获得可调用的 GPU 内核。
    #  - 不要忘记以关键字参数传递元参数。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    # 返回 z 的句柄，但由于 `torch.cuda.synchronize()` 尚未被调用，此时内核仍在异步运行。
    return output
```

使用上述函数计算两个 `torch.tensor` 对象的 element-wise sum，并测试其正确性：

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

现在准备就绪。

## 基准测试

在 size 持续增长的向量上对自定义算子进行基准测试，从而比较其与 PyTorch 的性能差异。为了方便操作，Triton 提供了一系列内置工具，允许开发者简洁地绘制自定义算子在不同问题规模 (problem sizes) 下的性能图。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        # 用作绘图 x 轴的参数名称。
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        # `x_name` 的不同可能值。
        x_log=True,  # x axis is logarithmic.
        # x 轴为对数。
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        # 参数名称，其值对应于绘图中的不同线条。
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        # `line_arg` 的可能值。
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        # 线条的标签名称。
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        # 线条样式。
        ylabel='GB/s',  # Label name for the y-axis.
        # y 轴标签名称。
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        # 绘图名称。也用作保存绘图的文件名。
        args={},  # Values for function arguments not in `x_names` and `y_name`.
        # 不在 `x_names` 和 `y_name` 中的函数参数值。
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```

运行上述装饰函数 (decorated function)。输入 `print_data=True` 查看性能数据，输入 `show_plots=True` 绘制结果， 以及/或者输入 `save_path='/path/to/results/'` 将其与原始 CSV 数据一起保存到磁盘：

```python
benchmark.run(print_data=True, show_plots=True)
```

![图片](/img/docs/Tutorials/VectorAddition/01.png)

[Download Jupyter notebook: 01-vector-add.ipynb](https://triton-lang.org/main/_downloads/f191ee1e78dc52eb5f7cba88f71cef2f/01-vector-add.ipynb)

[Download Python source code: 01-vector-add.py](https://triton-lang.org/main/_downloads/62d97d49a32414049819dd8bb8378080/01-vector-add.py)

[Download zipped: 01-vector-add.zip](https://triton-lang.org/main/_downloads/4e511f795844d864249b83f016d8ce09/01-vector-add.zip)
