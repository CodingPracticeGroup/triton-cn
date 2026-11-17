# 翻译同步规范（Triton 中文文档）

本文件说明本项目与英文上游文档保持同步时遵循的约定。目的：让每个页面的更新可复现、术语一致、不因翻译引入新的事实。

## 上游基线

- 上游仓库：<https://github.com/triton-lang/triton>
- 文档源文件位置：`docs/`（说明文档）与 `python/tutorials/`（教程，站点页面由 Sphinx + sphinx-gallery 从这些 `.py` 生成）、`python/triton/language/**`（API 页正文取自函数 docstring）
- 英文站点（main 分支构建）：<https://triton-lang.org/main/index.html>
- 本项目对应同步的**固定上游提交**记录在 `UPSTREAM_SYNC.md`。任何一次同步都应以某个确定的提交为准，不用“最新版”这种模糊说法。

## 页面格式

- 每页开头保留 Docusaurus frontmatter，只写 `title`。标题格式：`中文标题 (English Title)`，与上游页面标题一致。
- 上游 docstring / markdown 单元格里的正文 → 译成中文段落，小节标题对应关系：上游 docstring 的 `Title\n=====` 与小节标题 → `##`。
- 代码块：代码本身与上游逐字一致（不改动 API 名、参数、数值、注释以外的内容）。
- 上游代码与 docstring 中的**英文注释**保留原文，并在其下一行补中文对照，形如：

  ```python
  # read  MN elements ; write M  elements
  # 读取 MN 个元素；写入 M 个元素
  ```

  已有中文对照的注释不重复添加。没有英文原文的注释不凭空增补。
- 本项目自行添加的内容（例如页首的 `[在线运行此教程](https://openbayes.com/console/hyperai-tutorials/...)` 链接、超神经相关文案）在同步时保留，不被上游内容覆盖，也不删除。
- 上游页面里的外部链接（论文、CUDA 文档、GitHub issue）原样保留。

## 术语

统一使用下列译法；一个概念只用一个词。

| 英文 | 中文 |
| --- | --- |
| kernel | 内核 |
| autotune / heuristic | 自动调优 / 启发式（规则） |
| reduction / scan | 归约 / 扫描 |
| mask | 掩码 |
| block pointer | 块指针 |
| tensor descriptor | 张量描述符（首次出现时给出英文） |
| shared memory / register | 共享内存 / 寄存器 |
| transpose | 转置 |
| eviction policy | 驱逐策略 |
| persistent kernel | 持久化内核 |
| pipelining | 流水线（化） |
| prologue / epilogue | 前导部分 / 结尾部分（首次出现给英文） |
| stage count | 流水线级数 |
| warp specialization | warp 特化 |
| leading dimension | 前导维度 |
| on host / on device | 主机端 / 设备端 |
| TMA descriptor | 张量描述符（与 tensor descriptor 同一事物，不另译） |
| subnormal number | 次正规数 |
| involution | 对合 |
| widening / narrowing | 加宽 / 窄化 |
| rotate | 循环移位 |

以下词保留英文，不翻译：`bitcast`、`trap`、`extern`、`payload`（译作「载荷」时仅在 FpSan 等正式章节首次出现时给英文）、`warp`、`grid`、`block`（指线程块语义时）、`tile`、`CTA`、`SM`、`PTX`、`TMA`、`MMA`、`WGMMA`、`tcgen05`、`Gluon`、`Proton`。首次出现时用括号给出中文说明即可。

## 翻译原则

- 保留原文的事实、数值、单位、条件、否定、例外与默认值；不补造性能数字、支持范围或使用建议。
- 原文说“可能/通常/仅支持”时不写成绝对结论；原文未提及的限制不自行补充。
- 代码、命令、路径、URL、错误信息、API 名称与参数名一律保持原样。
- 不确定或上游原文本身含糊之处，保持同样的不确定程度，并在提交说明里指出，不自行“补全”。

## 同步一次上游需要做的检查

1. 页面清单：上游 `docs/index.rst` 的 toctree 与本项目 `docs/` 结构对比，列出新增、删除、改名的页面。
2. 内容：对每个已存在的页面，与上游对应源文件的相应版本比对，只同步确有变化的部分。
3. API 页：以 `docs/python-api/*.rst` 的 autosummary 条目为清单，逐条核对 `python/triton/**` 中对应函数的签名与 docstring。
4. 改名页面：更新文件名的同时，在 `docusaurus.config.ts` 的 `redirects` 中补旧路径，保证线上已分发的 URL 不失效；同时修正站内引用。
5. 构建：`pnpm install` 后 `npx docusaurus build`，断链按 `onBrokenLinks: throw` 处理。

## 运行输出与图片（`Out:` 块、`![图片]` 引用）

本站页面的 `Out:` 文本块与 `![图片]` 引用，是从英文站旧版构建结果里复制过来的执行输出（基准数据表、性能图）。同步代码时的处理约定：

- `Out:` 数值表：如果本页代码已经改动、这段输出不再由页中代码产生，**删除**它（保留会给出与代码矛盾的结果）。
- 图片：无法在本地重跑（需要 GPU 与对应驱动），保留原图，同时在 `UPSTREAM_SYNC.md` 的已知缺口中登记「图片资产仍为旧版构建结果」。
- 不自己编造运行结果或基准数据，也不把旧数据改写成“示意”。
