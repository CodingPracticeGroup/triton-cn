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

1. 被上游重写的已有页面：`06-fused-attention`、`08-grouped-gemm`、`09-persistent-matmul`、安装页、调试页、Triton 语义页、Python API 页。
2. 上游改名、本项目跟随改名的页面（旧地址通过 redirects 保留）：
   - `07-libdevice-tl.extra.libdevice-function.md` → `07-extern-functions.md`
   - `08-group-gemm.md` → `08-grouped-gemm.md`
3. 上游新增页面：教程 `10-block-scaled-matmul`、`11-programmatic-dependent-launch`；编程指南 FpSan；`triton.language.extra.cuda`。
4. `triton.language` API 页：按上游 autosummary 清单补齐新增条目、刷新签名与 docstring 已变化的条目。

## 暂未覆盖（已知缺口，留待后续分支）

- Gluon 全部文档：`gluon/index` 概览、`gluon/api/*`（NVIDIA Ampere/Hopper/Blackwell/Rubin、AMD CDNA3/4/5、RDNA3/4、runtime）以及 14 篇 Gluon 教程（`python/tutorials/gluon/*`）。
- 教程页中的图片（`![图片](/img/docs/Tutorials/...)`）与仍保留的 `Out:` 输出：均为英文站旧版构建产物。代码更新后，基准数据表已按约定删除；图片需要在有 GPU 的环境重跑教程才能更新，本次未更新。
- `docs/meetups/*`（上游会议记录，非文档正文，本项目历史上未翻译）。
- MLIR Dialects 页面：上游由 TableGen 自动生成，本项目现有的 5 个静态页与生成结果会有差异；如需长期跟住上游，应改为脚本生成，而不是手工维护。

## 同步一次上游的操作顺序

见 `TRANSLATION_GUIDE.md` 最后一条。核对差异时的两个检查点：`docs/index.rst` 的 toctree 对比本仓库 `docs/` 结构；`docs/python-api/*.rst` 的 autosummary 条目对比 `docs/02-python-api/` 下的页面。
