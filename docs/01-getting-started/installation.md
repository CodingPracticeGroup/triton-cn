---
title: 安装
---

已支持的平台/操作系统及硬件，请查看 [Github Compatibility 板块](https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)。


## Binary 版本


通过 pip 安装最新 Triton 稳定版：

```plain
pip install triton
```


针对 CPython 3.10-3.14 的 Binary wheels 现已可用。


## 源码安装


### Python Package


运行以下命令从源代码安装 Python 软件包：

```plain
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
# 构建时依赖
pip install -e .
```


注意：如果系统上没有安装 llvm，可以通过 setup.py 脚本下载官方 LLVM 静态库并自动链接。


如需使用自定义 LLVM 进行构建，请查看 Github [Building with a custom LLVM](https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm) 板块。


然后可通过运行测试来检验安装情况：

```plain
# One-time setup
# 一次性设置
make dev-install

# To run all tests (requires a GPU)
# 运行全部测试（需要 GPU）
make test

# Or, to run tests without a GPU
# 或者，在没有 GPU 的情况下运行测试
make test-nogpu
```
