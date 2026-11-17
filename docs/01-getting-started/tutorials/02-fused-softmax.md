---
title: 融合 Softmax (Fused Softmax)
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/QEhTxGYyzqY)

在本教程中，您将编写一个融合的 softmax 操作，它在某一类特定的矩阵上——即那些每一行都能放入 GPU SRAM 的矩阵——比 PyTorch 的原生 op 显著更快。

在此过程中，您将了解到：

* 内核融合对于带宽受限操作的好处。

* Triton 中的归约算子。

## 动机

用于逐元素加法的自定义 GPU 内核具有教学价值，但在实践中并不能帮您走多远。让我们转而考虑一个简单的（数值稳定的）softmax 操作：

```python
import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a')


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    使用原生 PyTorch 计算 X 的逐行 softmax

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret
```

当直接用 PyTorch 实现时，计算 `y = naive_softmax(x)`（其中 $$ x \in \mathbb{R}^{M \times N} $$）需要从 DRAM 读取 $$ 5MN + 2M $$ 个元素，并写回 $$ 3MN + 2M $$ 个元素。这显然是浪费的；我们更希望有一个自定义的「融合」内核，它只读取 X 一次，并在芯片上完成所有必要的计算。这样做只需要读写 $$ MN $$ 字节，因此我们可以期望约 4 倍的理论加速（即 $$ \frac{8MN + 4M}{2MN} $$）。`torch.jit.script` 标志的目标是自动执行这种「内核融合」，但正如我们稍后会看到的，它仍远不够理想。

## 计算内核

我们的 softmax 内核的工作方式如下：每个程序加载输入矩阵 X 的一组行（按程序数量作为跨步），对其进行归一化，并将结果写回输出 Y。

请注意，Triton 的一个重要限制是每个块必须包含 2 的幂次数的元素，因此如果要处理任何可能的输入形状，我们就需要在内部对每一行进行「填充」，并恰当地保护内存操作：

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    # 本程序负责的起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # 跨步表示为了前进 1 行，需要把指针增加多少
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        # 块大小是大于 n_cols 的下一个 2 的幂，因此我们能把每一行放进单个块中
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # 将该行加载到 SRAM 中；由于 BLOCK_SIZE 可能大于 n_cols，需要使用掩码
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # 减去最大值以保证数值稳定性
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # 请注意，Triton 中的指数运算很快，但是近似的（可类比为 CUDA 中的 __expf）
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

我们可以创建一个辅助函数，为任意给定的输入张量把内核及其（元）参数排入队列。

```python
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # 每次循环迭代的块大小是大于 `x` 列数的最小 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    # 另一个可用的技巧是：增加每一行所分布的 warp 数量（`num_warps`），从而让编译器为每一行使用更多线程。
    # 您会在下一个教程中看到，如何以更自然的方式自动调优这个值，而不必自己拟定手动启发式规则。
    num_warps = 8

    # Number of software pipelining stages.
    # 软件流水线的级数。
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    # 分配输出
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    # 预编译内核以获取寄存器用量，并计算线程占用率。
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        # NUM_REGS 表示通用目的寄存器（regular purpose registers）的数量。在 CDNA 架构上，它是全部可用寄存器数的一半。
        # 但这并不总是成立。在多数情况下，所有寄存器都可以用作通用目的寄存器。
        # ISA 规范章节（CDNA3 见 3.6.4）
        # VGPR 从两个池中分配：常规 VGPR 与累加 VGPR（accumulation VGPRs）。累加 VGPR 用于矩阵 VALU 指令，也可以直接从内存加载。
        # 一个 wave 最多可以有 512 个 VGPR，两种类型各 256 个。当 wave 的 VGPR 总数少于 512 时，两种类型各自的数量是灵活的——
        # 并不要求两种类型数量相等。
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        # MAX_NUM_THREADS 表示每个多处理器上可常驻的线程的最大数量。
        # 将这个数除以 WARP_SIZE，就得到能够在单个 CU（多处理器）上并行执行的 wave 的最大数量。
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    # 创建若干持久化程序。
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y
```

## 单元测试

我们要确保在一个行数和列数都不规则的矩阵上测试内核。这样就能验证我们的填充机制确实有效。

```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

不出所料，结果完全一致。

## 基准测试

这里我们将以输入矩阵的列数为自变量对该操作进行基准测试——假设输入有 4096 行。然后把它与（1）`torch.softmax` 和（2）上面定义的 `naive_softmax` 的性能进行比较。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        # 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        # `x_name` 的不同可能取值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # 其值对应图表中不同线条的参数名
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        # `line_arg` 的可能取值
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        # 线条的标签名称
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        # 线条样式
        ylabel="GB/s",  # label name for the y-axis
        # y 轴的标签名称
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        # 图表名称，也用作保存图表时的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
        # `x_names` 和 `y_name` 中未包含的函数参数的取值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
```

![图片](/img/docs/Tutorials/FusedSoftmax/02.png)

在上面的图中，我们可以看到：

 - Triton 比 Torch JIT 快 4 倍。这印证了我们的猜测：Torch JIT 在这里没有做任何融合。
 - Triton 明显比 `torch.softmax` 更快——而且**更易读、易理解、易维护**。
   但请注意，PyTorch 的 `softmax` 操作更为通用，可用于任意形状的张量。

[Download Jupyter notebook: 02-fused-softmax.ipynb](https://triton-lang.org/main/_downloads/034d953b6214fedce6ea03803c712b89/02-fused-softmax.ipynb)

[Download Python source code: 02-fused-softmax.py](https://triton-lang.org/main/_downloads/d91442ac2982c4e0cc3ab0f43534afbc/02-fused-softmax.py)

[Download zipped: 02-fused-softmax.zip](https://triton-lang.org/main/_downloads/f66de4fbee2c4ba20b6f7f3ae99f7de3/02-fused-softmax.zip)
