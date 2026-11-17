---
title: Libdevice (tl.extra.libdevice) 函数
---

[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/containers/RFagQOhvTsc)

Triton 可以调用外部库中的自定义函数。在这个例子中，我们将使用 `libdevice` 库在张量上应用 `asin` 函数。

关于所有可用 libdevice 函数的语义，请参阅 [CUDA libdevice-users-guide](https://docs.nvidia.com/cuda/libdevice-users-guide/index.html) 和/或 [HIP device-lib 源代码](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src)。

在 `libdevice.py` 中，我们试图将执行相同计算但针对不同数据类型的函数聚合在一起。例如，`__nv_asin` 和 `__nv_asinf` 都计算输入的反正弦的主值，但 `__nv_asin` 作用于 `double` 类型，而 `__nv_asinf` 作用于 `float` 类型。Triton 会根据输入和输出类型，自动选择要调用的正确底层设备函数。


## asin 内核

```python
import torch

import triton
import triton.language as tl
import inspect
import os
from triton.language.extra import libdevice

from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)
```


## 使用默认的 libdevice 库路径

可以使用 triton/language/math.py 中编码的默认 libdevice 库路径。


```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
output_triton = torch.zeros(size, device=DEVICE)
output_torch = torch.asin(x)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```


Out:

> tensor([0.4105, 0.5430, 0.0249,  ..., 0.0424, 0.5351, 0.8149], device='cuda:0')  tensor([0.4105, 0.5430, 0.0249,  ..., 0.0424, 0.5351, 0.8149], device='cuda:0')  The maximum difference between torch and triton is 2.384185791015625e-07

## 定制 libdevice 库路径


可以通过将 `libdevice` 库的路径传递给 `asin` 内核来定制 libdevice 库的路径。


```python
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


current_file = inspect.getfile(inspect.currentframe())
current_dir = Path(os.path.dirname(os.path.abspath(current_file)))

if is_cuda():
    libdir = current_dir.parent.parent / 'third_party/nvidia/backend/lib'
    extern_libs = {'libdevice': str(libdir / 'libdevice.10.bc')}
elif is_hip():
    libdir = current_dir.parent.parent / 'third_party/amd/backend/lib'
    extern_libs = {}
    libs = ["ocml", "ockl"]
    for lib in libs:
        extern_libs[lib] = str(libdir / f'{lib}.bc')
else:
    raise RuntimeError('unknown backend')

output_triton = torch.empty_like(x)
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```


Out:

> tensor([0.4105, 0.5430, 0.0249,  ..., 0.0424, 0.5351, 0.8149], device='cuda:0')  tensor([0.4105, 0.5430, 0.0249,  ..., 0.0424, 0.5351, 0.8149], device='cuda:0')  The maximum difference between torch and triton is 2.384185791015625e-07

[Download Jupyter notebook: 07-extern-functions.ipynb](https://triton-lang.org/main/_downloads/859d98d69fb02c33053d474c27761677/07-extern-functions.ipynb)

[Download Python source code: 07-extern-functions.py](https://triton-lang.org/main/_downloads/e496f88a5c4661dd03a2078bcc68f743/07-extern-functions.py)

[Download zipped: 07-extern-functions.zip](https://triton-lang.org/main/_downloads/3cf54e8aaddcfce69d180b77518fd544/07-extern-functions.zip)

