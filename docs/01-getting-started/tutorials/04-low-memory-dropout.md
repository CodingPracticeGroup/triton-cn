---
title: 低内存 Dropout
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/mkRMwoRH87l)

在本教程中，您将编写一个内存高效的 Dropout 实现，其状态将由单个 int32 seed 组成。这与传统 Dropout 实现不同，传统实现通常由与输入 shape 相同的位掩码张量组成。


在这过程中，您将学习到以下内容：


* PyTorch 中原生实现 Dropout 的局限性。
* Triton 中的并行伪随机数生成。

## Baseline


Dropout 是在 [[SRIVASTAVA2014]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#srivastava2014) 中引入的一种技术，用于改善低数据条件下深度神经网络的性能（即正则化）。它接受一个向量作为输入，并生成相同 shape 的输出向量。输出中的每个标量都有概率 $$ p $$ 被设为零，否则直接从输入复制。这使得网络在仅有输入的 $$ 1 - p $$ 标量时也能表现良好。


在评估阶段，为了充分利用网络的能力，将 $$ p $$ 设为 0。但是简单地将 $$ p $$ 设为 0 会增加输出的范数，可能会人为地降低输出的 softmax temperature。为了防止这种情况发生，输出被缩放为 $$ \frac{1}{1 - p} $$，这使得无论 dropout 概率如何都能保持一致的范数。


首先看一下 baseline 的实现。


```python
import tabulate
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    # 输入指针
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    # 由 0 和 1 组成的掩码的指针
    output_ptr,  # pointer to the output
    # 输出指针
    n_elements,  # number of elements in the `x` tensor
    # `x` 张量的元素数量
    p,  # probability that an element of `x` is changed to zero
    # `x` 的某个元素被置为零的概率
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    # 下一行是上段描述的关键部分
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    # 写回输出
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# Input tensor
# 输入张量
x = torch.randn(size=(10, ), device=DEVICE)
# Dropout mask
# Dropout 掩码
p = 0.5
x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))
```


## 种子化 Dropout


上述 Dropout 实现效果良好，但处理起来会有些别扭。首先，我们需要存储 dropout 掩码以供反向传播使用。其次，在使用重计算/检查点 (recompute/checkpointing) 时，dropout 状态的管理会变得非常棘手（例如参见 https://pytorch.org/docs/stable/checkpoint.html 中关于 `preserve_rng_state` 的全部说明）。在本教程中，我们描述一种替代实现，它具有以下优点：


1. 更小的内存占用。
2. 较少的数据移动。
3. 简化了在多次调用内核函数时持久化随机性的管理。

Triton 中的伪随机数生成很简单！在本教程中，我们将使用 `triton.language.rand` 函数，给定一个种子和一组 `int32` 偏移量，它生成一个块的均匀分布的 `float32` 值，范围在 [0, 1) 内。但如果你需要，Triton 也提供其他[随机数生成策略](https://triton-lang.org/main/python-api/triton.language.html#random-number-generation)。


>注意
>Triton 的 PRNG 实现基于 Philox 算法（详见 [[SALMON2011]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#salmon2011)）。

现在将所有内容整合起来。


```python
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    # 计算由此实例处理的元素的内存偏移量
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    # 从 x 读取数据
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    # 随机修剪
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    # 写回
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
# 与基线相比 - dropout 掩码从未被实例化！
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))
```


大功告成！我们得到了一个 Triton 内核：只要种子相同，它应用的就是同一个 dropout 掩码！如果你想进一步探索伪随机性在 GPU 编程中的应用，建议你阅读 `python/triton/language/random.py`！


## 练习


1. 扩展内核以处理矩阵，并使用一个种子向量 — 每行一个种子。
2. 添加对 striding 的支持。
3. （挑战）实现稀疏 Johnson-Lindenstrauss 变换的内核，每次使用种子动态生成投影矩阵。

## 参考文献


* [[SALMON2011]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#id2) John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
* [[SRIVASTAVA2014]](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html#id1) Nitish Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014

[Download Jupyter notebook: 04-low-memory-dropout.ipynb](https://triton-lang.org/main/_downloads/bc847dec325798bdc436c4ef5ac8b78a/04-low-memory-dropout.ipynb)

[Download Python source code: 04-low-memory-dropout.py](https://triton-lang.org/main/_downloads/c9aed78977a4c05741d675a38dde3d7d/04-low-memory-dropout.py)

[Download zipped: 04-low-memory-dropout.zip](https://triton-lang.org/main/_downloads/9241eab99db7582ceb6cd81f77524214/04-low-memory-dropout.zip)
