---
title: triton_language.inline_asm_elementwise
---

```python
triton.language.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)
```


在张量上执行内联汇编。本质上，这是一个映射，其中函数体是内联汇编。


输入张量 `args` 隐式地广播到相同的形状。


在 Gluon 中，`args` 还可以包含共享内存描述符或张量内存（tensor memory）描述符。每个描述符每次调用贡献 1 个 `i32` 地址，这与 `pack` 无关，且不参与广播。描述符操作数要求 `is_pure=False`；汇编被保守地视为会读写这些描述符的视图，并且访问必须保持在它们的逻辑元素范围之内。


`dtype` 可以是类型的元组，此时输出将是张量的元组。


每次内联汇编调用会一次处理一组元素。具体来说，一个块接收哪一组输入是未指定的。小于 4 字节的输入元素会被打包到 4 字节寄存器中。


此操作不支持空的 `dtype` —— 内联汇编必须返回至少一个张量，即使你不需要它。你可以通过返回 1 个任意类型的虚拟张量来解决这个问题；如果你不使用它，则不会产生任何成本。


使用 [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 汇编的示例：


```python
@triton.jit
def kernel(A, B, C, D, BLOCK: tl.constexpr):
    a = tl.load(A + tl.arange(0, BLOCK)) # uint8 tensor
    b = tl.load(B + tl.arange(0, BLOCK)) # float32 tensor


    # For each (a,b) in zip(a,b), perform the following:
    # 每一对 zip(a,p) 中的 (a,b), 执行以下操作
    # - Let ai be `a` converted to int32.
    # - 令 ai 转换为 int32 的`a`
    # - Let af be `a` converted to float.
    # - 令 af 为转换为 float 的 `a`
    # - Let m be the max of ai and b.
    # - 令 m 为 ai 和 b 的最大值
    # - Return ai and mi.
    # - 返回 ai 和 mi
    # Do the above 4 elements at a time.
    (c, d) = tl.inline_asm_elementwise(
        asm="""
        {
            // Unpack `a` into `ai`.    解包 `a` 到 `ai` 中
            .reg .b8 tmp<4>;
            mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
            cvt.u32.u8 $0, tmp0;
            cvt.u32.u8 $1, tmp1;
            cvt.u32.u8 $2, tmp2;
            cvt.u32.u8 $3, tmp3;
        }
        // Convert `ai` to float. 转换 `ai` 为 float
        cvt.rn.f32.s32 $4, $0;
        cvt.rn.f32.s32 $5, $1;
        cvt.rn.f32.s32 $6, $2;
        cvt.rn.f32.s32 $7, $3;
        // Take max of `ai` and `b`. 获取 `ai` 和 `b` 的最大值
        max.f32 $4, $4, $9;
        max.f32 $5, $5, $10;
        max.f32 $6, $6, $11;
        max.f32 $7, $7, $12;
        """,
        constraints=(
            # 8 output registers, namely 即 8 个输出寄存器
            #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
            #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
            "=r,=r,=r,=r,=r,=r,=r,=r,"
            # 5 input registers, namely 即 5 个输入寄存器
            #   $8=ai,
            #   $9=b0, $10=b1, $11=b2, $12=b3.
            # The four elements from `a` are all packed into one register.
            # 来自 `a` 的四个元素全都打包进一个寄存器中
            "r,r,r,r,r"),
        args=[a, b],
        dtype=(tl.int32, tl.float32),
        is_pure=True,
        pack=4,
    )
    tl.store(C + tl.arange(0, BLOCK), c)
    tl.store(D + tl.arange(0, BLOCK), d)
```


**参数：**

* **asm** - 要运行的汇编代码。必须与目标的汇编格式匹配。
* **constraints** - 采用 [LLVM 格式](https://llvm.org/docs/LangRef.html#inline-asm-constraint-string)的汇编约束条件。
* **args** - 输入张量，其值传递给汇编块。
* **dtype** - 返回的张量（一个或多个）的元素类型。
* **is_pure** - 如果为 true，编译器假定汇编块没有副作用。
* **pack** - 单个内联汇编实例处理的元素数量。

**返回：** 给定类型的 1 个张量，或由这些类型构成的 1 个张量元组。

