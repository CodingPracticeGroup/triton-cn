---
title: triton_language.make_tensor_descriptor
---

```python
triton.language.make_tensor_descriptor(base, shape, strides, block_shape, padding_option, fp6_format)
```

创建一个张量描述符（tensor descriptor）对象。

**参数：**

* **base** - 张量的基础指针，必须是 16 字节对齐的。
* **shape** - 表示张量形状的非负整数列表。
* **strides** - 张量步幅列表。前面各维度必须是 16 字节步幅的整数倍，且最后一维必须是连续的。
* **block_shape** - 要从全局内存中加载/存储的块的形状。
* **fp6_format** - 对于打包的 FP6 TMA 存储，使用 "e2m3" 或 "e3m2"，并搭配 uint8 类型的基础指针。形状与偏移量按逻辑值计数。外层步幅按字节计数。每个源 uint8 保存一个 FP6 编码。

## 说明

在支持 TMA 的 NVIDIA GPU 上，这会生成一个 TMA 描述符对象，并且对该描述符的加载和存储将由 TMA 硬件提供支持。

目前仅支持 2-5 维张量。

## 示例

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))

# TMA descriptors require a global memory allocation
# TMA 描述符需要一次全局内存分配
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

M, N = 256, 256
x = torch.randn(M, N, device="cuda")
M_BLOCK, N_BLOCK = 32, 32
grid = (M / M_BLOCK, N / N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
