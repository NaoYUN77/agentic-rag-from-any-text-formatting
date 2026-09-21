# 规划与实现对照表

> **为什么需要这份文档**
>
> `docs/format_routing_and_cleaning_plan.md` 写于**旧 LlamaIndex 链路时代**,  
> 描述的是"规划中的架构"。代码后来演进到 **Block-native**,但规划文档没有同步更新,  
> 导致两者出现系统性偏差。按规划文档理解当前架构会被误导。
>
> 本文逐条对照规划文档的每个设计点与代码实际状态,**规划文档保持原样作为设计史料**,  
> 认知以本文为准。
>
> 核对时间:2026-09-17　核对方式:读取 `pipeline/` 全部源码 + 实测产物  
> 涉及文档:`docs/format_routing_and_cleaning_plan.md`(1377 行)

---

## 1. 一句话结论

```text
规划文档的"分层思想"是对的,并且已经实现;
规划文档的"实现载体"多处过时 —— 尤其是"交给 LlamaIndex 切块"这一条。
```

规划的核心主张:

```text
多格式 → 各自 parser → 统一 DocumentArtifact/DocumentBlock
       → 质量门 → 清洗 → 切块 → dense/sparse 同 chunk_id
```

这条主线**代码里完整存在**。偏差集中在"谁来切块"和"目录怎么组织"。

---

## 2. 重大偏差(会导致理解错误,优先看)


### 偏差 A:切块责任方 —— 规划说"交给 LlamaIndex",实际是自己实现

**规划原文**(第 25 行,设计目标 7):

> ```text
> 7. 后续仍由 LlamaIndex 负责 chunking
> ```

**规划原文**(第 95 行,总体架构图):

> ```text
>                               LlamaIndex NodeParser
>                                          |
>                                          v
>                                   IndexReadyChunk
> ```

**规划原文**(第 1375 行,验收标准 8):

> ```text
> 8. 后续能把这些 block 继续交给 LlamaIndex NodeParser
> ```
>
>

**代码实际**:`ingest/chunker.py` 里有完整的 `BlockAwareHierarchicalChunkBuilder`(451 行),  
**自己实现了 Block → Piece → Chunk 的全部逻辑**。LlamaIndex 只被降级使用了  
一个组件 —— `SentenceSplitter`,且仅用于"长 text 块的内部句子级切分":

```python
# chunker.py:10-11   只 import 这两个
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

# chunker.py:141     唯一实际用途:句子切分
self._splitter = SentenceSplitter(chunk_size=chunk_tokens, chunk_overlap=0)
```

**验证**:`chunker.py` 全文不含 `embedding`、不含语义距离计算。  
"语义边界微调"**没有实现**(与 7.1 节自述一致)。

**为什么这个偏差重要**:它决定了 Chunk 的粒度、overlap 方式、section 边界  
**全部由本项目自己控制**,不受 LlamaIndex 的 NodeParser 行为约束。  
你如果要改切块策略,改的是 `chunker.py`,不是 LlamaIndex 配置。

**结论**:规划文档第 25 / 95 / 1375 行的 "LlamaIndex NodeParser" 应读作  
「Block-aware Hierarchical ChunkBuilder」。

---


### 偏差 B:目录规划 —— 描述的目录结构不存在

**规划原文**(第 1332-1351 行):

```text
pipeline/ingest/
├── models.py
├── router.py
├── source.py
├── quality.py
├── parsers/
│   ├── html_trafilatura.py      ← 实际叫 html.py
│   ├── pdf_mineru.py            ← 曾存在, 2026-09-20 已删除(见下)
│   ├── pdf_docling.py           ← 实际叫 docling_optional.py
│   ├── markdown.py              ✅ 存在
│   └── office.py                ← 实际叫 office_optional.py
└── cleaners/                    ← ❌ 整个目录不存在
    ├── html.py
    ├── pdf.py
    ├── markdown.py
    └── blocks.py
```

**代码实际**:

```text
pipeline/ingest/
├── models.py                    ✅ 存在
├── router.py                    ✅ 存在
├── source.py                    ✅ 存在
├── quality.py                   ✅ 存在
├── parser_registry.py           ➕ 计划外新增
├── pipeline.py                  ➕ 计划外新增(IngestPipeline 编排)
├── chunker.py                   ➕ 计划外新增(规划放在别处)
├── sentence_chunker.py          ➕ 计划外新增(单句窗口, 与 chunker 并存做 A/B)
├── cleaner.py                   ⚠️ 单个文件,不是 cleaners/ 目录
├── qdrant_indexer.py            ➕ 计划外新增
└── parsers/
    ├── __init__.py
    ├── html.py                  ✅ 从 DOM 直产 Block(不再经 Markdown)
    ├── markdown.py              ✅ 状态机
    ├── plain.py                 ✅ 纯文本(不再复用 Markdown 解析器)
    ├── pdf_pdfplumber.py        ➕ 计划外新增, PDF 默认(MIT)
    ├── pdf_common.py            ➕ 计划外新增(三个 PDF adapter 共享的解析核心)
    ├── pdf_pymupdf.py           ✅ 真实实现(AGPL 备选)
    ├── docling_optional.py      ⚠️ 仅接口壳,抛 NotImplementedError (15 行)
    └── office_optional.py       ⚠️ 仅接口壳,抛 NotImplementedError (15 行)
```

> 更新 (2026-09-20)：**`pdf_mineru.py` 已删除**。它是唯一经  
> `PDF → Markdown → Block` 中转的解析器，会丢页码/坐标/字号，  
> heading 层级被压平（issues/11），且依赖外部 CLI 与 20 页上限。

**关键变化:cleaner 没有按格式拆分。** 规划设想的是每个格式一个 cleaner,  
实际收敛成 `cleaner.py` 里的一个 `clean_blocks()` 函数 —— 因为 Block 化之后  
所有格式已是同一种表示,**清洗不再需要知道原始格式**。这是 Block 抽象的收益。

---

### 偏差 C:`block_ids` 与 `fragments` —— 规划文档自相矛盾

**规划第 202 行**(推荐 Chunk 结构示例):

```json
"block_ids": ["B4", "B5", "B6"]
```

**规划第 260 行**(Chunk 只保存检索视图示例):

```json
"fragments": [
  {"block_id": "b12", "start": 0, "end": 9},
  ...
]
```

**规划第 271 行**(自己纠正了):

> 不要只保存 `block_ids`。超长段落可能只被 Chunk 使用一部分,  
> overlap 也可能只复用某个 Block 的尾部。

**代码实际**:采用后者。`chunker.py:21-27` 的 `ChunkFragment`:

```python
@dataclass
class ChunkFragment:
    block_id: str
    start_char: int
    end_char: int
    block_type: str
    text: str
```

**字段名也不同**:规划写 `start`/`end`,代码是 `start_char`/`end_char`。  
后者更明确,是对的。

**注意**:规划文档把两套方案都留在正文里了,读到第 202 行会以为  
`block_ids` 是现行设计。**以第 271 行为准。**

---

## 3. 逐条核对(设计目标 8 条)

| # | 规划目标(第 18-27 行)                        | 状态           | 代码位置 / 说明                                                              |
| - | -------------------------------------- | ------------ | ---------------------------------------------------------------------- |
| 1 | 支持 URL、HTML、PDF、扫描件、Markdown、Office    | ⚠️ 部分        | URL/HTML/Markdown/PDF ✅;扫描件 OCR ❌;Office 仅接口壳                          |
| 2 | 不同格式进入不同专用解析器                          | ✅ 已实现        | `parser_registry.py` 注册 8 个 parser                                     |
| 3 | 最终统一成 DocumentArtifact / DocumentBlock | ✅ 已实现        | `models.py:38/88`                                                      |
| 4 | 保留代码块、数学公式和图片                          | ⚠️ 部分        | code ✅(有 HTML 补充机制);formula ✅(解析器有分支,语料未出现);image ⚠️ 仅 URI,caption 常为空 |
| 5 | 扫描件有 OCR 置信度和质量门控                      | ❌ 未实现        | `ocr_confidence` 字段存在但全为 None                                          |
| 6 | 低质量 OCR 不污染 sparse/BM25                | ⚠️ 机制在,数据未触发 | `quality.py:89` 决策链已实现;当前语料全 high                                      |
| 7 | **后续仍由 LlamaIndex 负责 chunking**        | ❌ **已偏离**    | 见偏差 A,自建 `BlockAwareHierarchicalChunkBuilder`                          |
| 8 | 每个结论能追溯来源、页码、section、parser            | ⚠️ 部分        | artifact / block / section ✅;**page 全为 None**、bbox 未使用                 |

### 非目标 4 条(第 29-36 行)

| 目标                               | 状态                            |
| -------------------------------- | ----------------------------- |
| 不训练自己的 OCR/公式识别模型                | ✅ 遵守                          |
| 不做任意 PDF 的完美版面还原                 | ✅ 遵守                          |
| 不把所有图片做多模态向量化                    | ✅ 遵守                          |
| 不直接替换现有 chunk/retrieval pipeline | ⚠️ 事实上替换了(Block-native 已成主链路) |

---


## 4. 逐条核对(路由矩阵,规划第 885-895 行)

| 规划输入         | 规划首选                        | 代码实际                                    | 状态                  |
| ------------ | --------------------------- | --------------------------------------- | ------------------- |
| URL 博客/文章    | Trafilatura                 | `html.py` 双路,**按 heading 数量择优**         | ✅ 且更强               |
| URL 代码文档     | Trafilatura + BeautifulSoup | 同上 + `_extract_code_blocks` 补充          | ✅ 已实现               |
| 本地 Markdown  | Markdown passthrough        | `markdown_parser`                       | ✅                   |
| 中文 PDF / 扫描件 | MinerU                      | ~~`pdf_mineru` → `pdf_loader`~~ **已移除** | ⚠️ **无 OCR 链路**（见下） |
| 英文通用 PDF     | Docling                     | **未接入**(`docling_optional` 抛异常)         | ❌                   |
| 英文论文/公式      | Docling                     | **未接入**                                 | ❌                   |
| 简单文本层 PDF    | PyMuPDF                     | `pdf_pymupdf`                           | ✅                   |
| Office       | MarkItDown                  | **未接入**(`office_optional` 抛异常)          | ❌                   |

> **移除 `pdf_mineru` 的代价（2026-09-20）**
>
> 上表"中文 PDF / 扫描件"一行的首选路径已删除，因此：
>
> ```text
> 现在只剩两个 PDF parser，都是"文本层"解析：
>   pdf_pdfplumber (MIT, 默认)   需要 PDF 内含可提取文本
>   pdf_pymupdf    (AGPL, 回退)  同上
>
> 两者都没有 OCR —— 扫描件 / 纯图片 PDF 现在无法解析。
> ```
>
> 即：**这是一次能力回退，换取架构一致性与结构信息完整性。**  
> 如果后续需要扫描件支持，正确的补法不是把 MinerU 请回来，而是  
> 接一个**直产 Block 且带 OCR** 的 adapter（能给出页码/bbox/置信度），  
> 让它在 `pdf_pdfplumber` 之后作为回退。

**额外发现:规划未提及的"双路择优"机制。**

规划只写了"HTML → Trafilatura,备选 Readability",  
实际代码 `html.py:84-102` 是**两条路都跑,再比 heading 数量择优**:

```python
if readability_headings > trafilatura_headings and len(readability_markdown) >= minimum_chars:
    markdown = readability_markdown
    parser = "readability_structured"
else:
    markdown = trafilatura_markdown
    parser = "trafilatura"
```

实测 10 篇语料:`readability_structured` 7 篇、`trafilatura` 3 篇。  
**这是规划文档没有的设计改进。**

---

## 5. 逐条核对(Chunk Builder 规则,规划第 470-482 行)

| # | 规划规则                         | 实现 | 位置                                     |
| - | ---------------------------- | -- | -------------------------------------- |
| 1 | section 作为 Parent 边界         | ✅  | `chunker.py:206` `_group_blocks()`     |
| 2 | heading 进入 Leaf 开头或 metadata | ✅  | `chunker.py:242-249`                   |
| 3 | paragraph/list 可合并或切分        | ✅  | `chunker.py:149-175`                   |
| 4 | code/formula/table/image 原子  | ✅  | `chunker.py:16` `ATOMIC_TYPES`         |
| 5 | 接近 token 上限时优先在文本 Block 内切   | ✅  | `chunker.py:406-417`                   |
| 6 | overlap 使用尾块或尾句回退            | ✅  | `chunker.py:303-366` `_tail_overlap()` |
| 7 | 不把 Parent 和 Leaf 混成同一个检索点    | ✅  | Parent 不进 Qdrant                       |
| 8 | Parent 可只存摘要、标题路径、子节点关系      | ✅  | `ParentNode` 含 `child_chunk_ids`       |
| 9 | Leaf 是 dense/sparse 唯一检索单位   | ✅  | `qdrant_indexer` 只写 chunk              |

**第 5 条需补充说明:规划写的"语义断点"没有实现。**

规划第 369-370 行:

> ```text
> 4. 如果当前 Block 是可分割文本:
>        在边界窗口内寻找语义断点
> ```

实际 `_split_text_block()` 用的是 `SentenceSplitter`(纯句子边界),  
**`window_tokens` 参数接收了但没有参与语义计算**。这与 7.1 节自述一致:

> ```text
> window_tokens 没有真正参与 embedding 语义边界微调
> ```

---

## 6. 逐条核对(核心不变量,项目总览第 349-437 行)

这些必须保持,核对了代码与实际产物:

| #   | 不变量                             | 状态 | 实测证据                                           |
| --- | ------------------------------- | -- | ---------------------------------------------- |
| 5.1 | Chunk 是 Block 的索引视图             | ✅  | Block 存结构,Chunk 存 fragments                    |
| 5.2 | 一个 Block 可进入多个 Chunk            | ✅  | 实测 **78 个 block** 被切进多个 chunk                  |
| 5.3 | 一个 Chunk 可引用多个 Block fragment   | ✅  | `c0001` 引用 4 个 fragment,顺序保持                   |
| 5.4 | Fragment 坐标用字符 offset           | ✅  | `fragment.text == block.text[start:end]` 实测全成立 |
| 5.5 | section 是边界,heading 不重复进 sparse | ✅  | `SPARSE_TYPES` 不含 heading                      |
| 5.6 | 原子 Block 不拆                     | ✅  | `oversized_atomic_block` 标记                    |
| 5.7 | 普通 Chunk 不超 token 上限            | ✅  | 有单测覆盖                                          |
| 5.8 | dense 和 sparse 同一个 point_id     | ✅  | `point_id_for(chunk_id)` 纯函数                   |

**全部 8 条不变量成立。** 这是项目当前最扎实的部分。

---

## 7. 规划未实现 vs 已超额实现

### 7.1 规划里写了但没做

```text
Docling adapter                 → docling_optional.py 仅抛 NotImplementedError
Office / MarkItDown adapter     → office_optional.py 仅抛 NotImplementedError
扫描件 OCR 链路                  → ocr_confidence 字段全 None
图片 asset 提取                  → assets 字段恒为空列表
公式识别                         → 解析器有 formula 分支,语料中未出现实例
语义边界微调                     → window_tokens 未参与计算
page / bbox 覆盖                 → 当前 URL 语料全为 None
```

### 7.2 规划里没写但已经做了

```text
双路 HTML 择优(readability vs trafilatura)   → html.py:84-102
HTML 代码块补充(BeautifulSoup 兜底)          → html.py:111-123
Byte-level 稳定 point_id(blake2b)            → qdrant_indexer.py:154
Qdrant local rebuild + 旧 collection 清理     → 有单测覆盖
BM25 完整实现(vocab/postings/idf)            → sparse_pipeline_demo.py
RRF 融合 + Reranker                          → hybrid_retriever.py / reranker.py
FastAPI 服务 + Web 页面                       → rag_server.py
多格式摄取 CLI                                → format_router_demo.py
端到端闭环验证脚本                            → verify_phase0_loop.py
跨语言检索对照实验                            → eval_zh_crosslingual.py
```

### 7.3 规划里写了、实际状态不同的关键参数

| 项              | 规划           | 实际          | 位置                      |
| -------------- | ------------ | ----------- | ----------------------- |
| chunk_tokens   | 800(第 368 行) | 800         | `chunker.py:127`        |
| overlap_tokens | 未明确          | 400         | `chunker.py:128`        |
| window_tokens  | 400          | 400(但未参与语义) | `chunker.py:129`        |
| RRF k          | 未明确          | 60          | `rag_server.py` `rrf_k` |
| candidate_k    | 未明确          | 20          | `hybrid_retriever.py`   |
| final top_k    | 未明确          | 5           | `hybrid_retriever.py`   |
| BM25 k1 / b    | 未明确          | 1.2 / 0.75  | `bm25_stats.json`       |

---


## 8. 阅读指引:哪些文档可信,哪些要打折

| 文档                                       | 可信度         | 说明                                                                 |
| ---------------------------------------- | ----------- | ------------------------------------------------------------------ |
| `project_overview_for_agents.md`         | ✅ 高         | 2026-09-17 更新,已含 Phase 0 实况                                        |
| `block_structure_and_blockification.md`  | ✅ 高         | 实测产物核对                                                             |
| `ingest_pipeline_progress.md`            | ✅ 高         | 进度文档,持续更新                                                          |
| `block_aware_chunking_implementation.md` | ✅ 高         | 描述当前实现                                                             |
| `phase0_loop_verification.md`            | ✅ 高         | 实测结论                                                               |
| `zh_query_crosslingual_eval.md`          | ✅ 高         | 实测结论                                                               |
| `format_routing_and_cleaning_plan.md`    | ⚠️ **设计史料** | 第 1/2/3 章思想仍有效;凡涉及 LlamaIndex chunking、目录结构、block_ids 的部分已过时,以本文为准 |
| `rag_chunking_concepts.md` 等概念文档         | ✅ 高         | 通用原理,不随实现变化                                                        |
| `issues/`                                | ✅ 高         | 问题追踪                                                               |

**阅读 `format_routing_and_cleaning_plan.md` 时的替换规则**:

```text
看到 "LlamaIndex NodeParser"          → 读作 BlockAwareHierarchicalChunkBuilder
看到 "Block Cleaner(第 76 行流程)"     → 读作 cleaner.py 的 clean_blocks()
看到 "cleaners/ 目录"                 → 实际不存在,只有 cleaner.py
看到 "block_ids"                      → 读作 fragments(block_id, start_char, end_char)
看到 "window_tokens 语义微调"          → 实际未实现
看到 "RawDocumentArtifact"            → 实际叫 DocumentArtifact
```

---

## 9. 给规划文档的修订建议(待执行)

规划文档作为"设计思想的完整表达"有保留价值,建议**不做整篇重写**,而是:

```text
方案一(推荐): 在文首加"状态声明"块
    标明本文件写于旧链路时代,实现已演进,并链接本文
    优点: 保留设计推理的完整脉络,同时消除误导
    代价: 需改动 1 处

方案二: 在两条重大偏差处就地插注
    在 第 25 行、第 95 行、第 202 行、第 1332 行 各加一段
    "【已过时】实际实现见 xxx"
    优点: 读到哪就知道哪过时
    代价: 需改动 4 处

方案三: 整篇重写
    不推荐 —— 会丢失"为什么这样设计"的推理过程,
    而那部分对理解架构意图仍然有价值
```

**建议采用方案一 + 方案二的组合**:文首声明总体状态,关键位置就地标注。

---

## 附:核对方法与复现命令

所有结论来自以下核对,可复现:

```powershell
cd pipeline

# 目录结构
ls ingest/ ingest/parsers/

# parser 实现程度(找 NotImplementedError)
Select-String -Path ingest/parsers/*.py -Pattern "NotImplementedError"

# chunker 是否用 LlamaIndex NodeParser
Select-String -Path ingest/chunker.py -Pattern "llama_index"
Select-String -Path ingest/chunker.py -Pattern "embedding"    # 应无结果

# 默认参数
Select-String -Path ingest/chunker.py -Pattern "chunk_tokens|overlap_tokens|window_tokens"

# 不变量实测(需在 pipeline/ 下用 .venv_rag 运行)
#   fragment.text == block.text[start:end]
#   block_id/order 一致性
#   同一 block 是否跨多个 chunk
# 见 docs/block_structure_and_blockification.md 第 7 节

# 测试
.\.venv_rag\Scripts\python.exe -m unittest discover -s tests -v
```
