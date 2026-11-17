# 上游同步记录

本项目的中文内容以英文上游 <https://triton-lang.org/main/index.html> 为准。为避免“最新版”这种无法复现的说法，这里记录每次同步实际依据的上游提交。

## 当前同步基线

| 项目 | 值 |
| --- | --- |
| 上游仓库 | <https://github.com/triton-lang/triton> |
| 同步到的提交 | `c590e26174098075360fbe67bf251f9c408c7d89`（main，2026-11-17） |
| 对应的最近发布版 | v3.9.0（2026-10-26） |
| 上一个基线 | 上游 main 在 2024-12-11 前后的快照（介于 v3.1.0 与 v3.2.0 之间；本仓库最后一次内容更新 commit `9a14798`） |

## 本次同步（分支 `sync-upstream-docs-2026-11`）覆盖范围

1. 快速开始与教程：安装页；教程 01–05 逐行跟随上游（含 `DEVICE` 取代硬编码 `device="cuda"`、上游新增的 `tl.assume`、`debug_barrier`、grid 写法等），02/03/06/08/09 整页对齐；新增 `10-block-scaled-matmul`、`11-programmatic-dependent-launch`；教程索引页的安装命令跟随上游。
2. 跟随上游改名（旧地址用 `@docusaurus/plugin-client-redirects` 保留）：
   - `07-libdevice-tl.extra.libdevice-function.md` → `07-extern-functions.md`
   - `08-group-gemm.md` → `08-grouped-gemm.md`
   
   注意：上游只改了文件名，该页 docstring 标题仍是 `Libdevice (tl.extra.libdevice) function`，中文页标题按此处理。
3. 编程指南：调试页新增指向 FpSan 的链接并修正 ROCm 文档链接；新增 FpSan 一章。
4. `triton-semantics`：补充除法/取模的提升例外、`float4e2m1x2` 与 `float8e8m0fnu` 的类型映射、链式比较、超出范围的浮点转整数、变量作用域。
5. `triton.language` API 页：按上游 `docs/python-api/*.rst` 的条目清单补齐缺页、刷新签名与文档文本（含由 `_add_*_docstr` 装饰器模板注入、以及 autodoc 从类定义生成这两类文本）；`triton.*`、`triton.testing` 与 `triton.language.extra.cuda` 同步；`02-triton.language/README.md` 索引按上游条目顺序重建。

## 本轮的取舍与已知偏差

- 签名写法：统一为 `triton.language.<名称>(参数=默认值…)`，**不写类型注解**，去掉 `self` 与下划线开头的私有参数。这与上游 Sphinx 渲染出的形式不完全一致（例如 `rand4x` 的 `n_rounds` 上游渲染为 `constexpr[10]`，我们写源码里的 `N_ROUNDS_DEFAULT`）；如要照线上站呈现，需要人工确定一套统一的写法后逐页改。
- `Out:` 基准数据表：凡本页代码已变化、旧数据不再由代码产生的，一律删除（01–06、08、09 等页），不编造新数据。
- 图片（`![图片](/img/docs/Tutorials/...)`）：保留旧版构建产物，需要在有 GPU 的环境重跑教程才能更新。
- 上游文档本身的不一致按原文保留，不代为订正：例如 `randn4x`/`randint4x` 的 `:param offsets:` 与签名参数名 `offset` 不符；若干陈旧注释与代码不符（如 06 页 `# Use device_descriptor for Hopper + warpspec.` 与实际条件相反）。
- `sort`：该快照里上游 `sort` 自身已无 docstring（文档文本在内部实现 `sort_impl` 上），中文页保留对 `sort` 参数仍然成立的概述，并去掉不再适用的「可作为 tensor 成员函数调用」一句。

## 暂未覆盖（已知缺口，留待后续分支）

- Gluon 全部文档：`gluon/index` 概览、`gluon/api/*`（NVIDIA Ampere/Hopper/Blackwell/Rubin、AMD CDNA3/4/5、RDNA3/4、runtime）以及 14 篇 Gluon 教程（`python/tutorials/gluon/*`）。
- `docs/meetups/*`（上游会议记录，非文档正文，本项目历史上未翻译）。
- MLIR Dialects 页面：上游由 TableGen 自动生成，本项目现有的 5 个静态页与生成结果会有差异；如需长期跟住上游，应改为脚本生成，而不是手工维护。
- 上游把 API 页完全交给 Sphinx autodoc 生成，本项目是逐页手工维护；后续可考虑写脚本从上游 docstring 生成英文底稿，再由人翻译，以减少每次同步的手工量。

## 同步一次上游的操作顺序

见 `TRANSLATION_GUIDE.md` 最后一条。核对差异时的两个检查点：`docs/index.rst` 的 toctree 对比本仓库 `docs/` 结构；`docs/python-api/*.rst` 的 autosummary 条目对比 `docs/02-python-api/` 下的页面。
