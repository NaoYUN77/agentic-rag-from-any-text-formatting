# Agent 项目总览: Agentic RAG from Any Text Formatting

更新时间: 2026-09-17

## 1. 一句话说明项目在做什么

这个项目要把不同来源、不同格式的文档：

```text
URL / HTML / PDF / 扫描件 / Markdown / Office
```

转换成可追溯、可清洗、可质量评估、可切块的统一文档结构，再构建：

```text
Dense retrieval
Sparse retrieval
Hybrid + RRF
Rerank
Generation + citations
FastAPI service
```

目标是形成一条完整的 Agentic RAG 文档处理与检索链路。

当前项目不是单纯的学习笔记，也不是只做切块演示。

它同时包含：

```text
可运行代码
实验记录
工程文档
问题追踪
在线 FastAPI 服务
本地 Qdrant 索引
```

## 2. 两个必须区分的链路

项目中同时存在两套链路，其他 Agent 接手时必须先分清。

### 2.1 Phase 0 多格式摄取与切块链路

```text
Source
-> Source Loader
-> Format Router
-> Parser
-> DocumentArtifact / DocumentBlock
-> Raw Parse Quality Gate
-> Block Cleaner
-> Index Quality Gate
-> Block-aware Hierarchical ChunkBuilder
-> ParentNode / IndexReadyChunk
-> artifact.json / parents.json / chunks.jsonl
-> qdrant_indexer
-> Dense / Sparse Qdrant collections
```

特点：

```text
偏新的架构
按 DocumentBlock 组织
有 fragments 来源追踪
dense_text / sparse_text / full_text 分离
可以独立重建 Qdrant
```

### 2.2 在线检索与生成链路

```text
FastAPI
-> query
-> Dense top-k
-> Sparse top-k
-> RRF
-> Reranker
-> final top-k
-> Qwen generation
-> answer + citations
```

当前 FastAPI 可以通过环境变量切换 Qdrant collection。

因此它可以读取：

```text
旧的 LlamaIndex chunks
```

也可以读取：

```text
Phase 0 IndexReadyChunk
```

这两条链路不要混为一谈。

## 3. 当前总体架构

```text
                    Source
               URL / File / PDF
                        |
                        v
                 Format Router
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
      HTML           Markdown          PDF
        |               |               |
 Trafilatura      Markdown Parser    MinerU
 Readability                         PyMuPDF
 BeautifulSoup
        |               |               |
        +---------------+---------------+
                        |
                        v
              DocumentArtifact
              DocumentBlock[]
                        |
                        v
              Raw Quality Gate
                        |
                        v
                 Block Cleaner
                        |
                        v
              Index Quality Gate
                        |
                        v
       Block-aware Hierarchical ChunkBuilder
                        |
         +--------------+--------------+
         |                             |
         v                             v
    ParentNode                  IndexReadyChunk
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
              Dense Embedding                    Sparse Vector
                    |                                   |
                    +-----------------+-----------------+
                                      |
                                      v
                                   Qdrant
                                      |
                                      v
                          Dense / Sparse retrieval
                                      |
                                      v
                                    RRF
                                      |
                                      v
                                  Reranker
                                      |
                                      v
                             Generation + citations
                                      |
                                      v
                                   FastAPI
```

## 4. 核心数据模型

主要位于：

```text
pipeline/ingest/models.py
pipeline/ingest/chunker.py
```

### 4.1 SourcePayload

表示已经获取到的原始来源：

```text
source_type
uri
final_uri
mime_type
content_type
data
encoding
headers
size_bytes
sha256
local_path
warnings
```

### 4.2 RouteDecision

表示格式路由结果：

```text
source_type
mime_type
parser
confidence
reason
fallbacks
mode
```

### 4.3 DocumentBlock

统一 Block 模型：

```text
block_id
type
text
markdown
parent_id
section_path
order
page
bbox
code_language
latex
image_uri
caption
ocr_confidence
quality_score
quality_flags
metadata
```

当前主要 Block 类型：

```text
heading
text
list
code
formula
table
image
```

### 4.4 DocumentArtifact

一篇文档的统一结果：

```text
schema_version
artifact_id
source_uri
source_type
mime_type
parser
parser_version
title
authors
published_at
language
blocks
assets
quality
index_decision
route
metadata
warnings
```

### 4.5 ChunkFragment

Chunk 的来源追踪单位：

```text
block_id
start_char
end_char
block_type
text
```

坐标语义：

```text
[start_char, end_char)
```

基准文本：

```text
CleanDocumentArtifact 中该 Block 的规范化 block.text
```

字符单位：

```text
Unicode code point
Python str 下标
```

关系：

```python
fragment.text == block.text[fragment.start_char:fragment.end_char]
```

### 4.6 ParentNode

章节级父节点：

```text
parent_id
artifact_id
scope_title
section_path
child_chunk_ids
full_text
token_count
page_start
page_end
metadata
```

### 4.7 IndexReadyChunk

真正的检索索引单位：

```text
chunk_id
artifact_id
parent_id
section_block_id
section_path
fragments
full_text
dense_text
sparse_text
token_count
char_count
page_start
page_end
overlap_from_previous
quality
index_decision
metadata
```

## 5. 必须遵守的核心不变量

其他 Agent 修改代码时，不要破坏这些约束。

### 5.1 Chunk 是 Block 的索引视图

```text
Block 负责文档结构
Chunk 负责检索组织
```

不要直接把 Block 当成 Chunk，也不要把 Chunk 当作原文档结构。

### 5.2 一个 Block 可以进入多个 Chunk

常见原因：

```text
长 Block 内部切分
overlap
重新切块
```

### 5.3 一个 Chunk 可以引用多个 Block fragment

fragments 必须保留顺序。

### 5.4 Fragment 坐标使用字符 offset

不能把 token 下标填进：

```text
start_char
end_char
```

Token 只用于：

```text
token_count
预算
统计
```

### 5.5 section 和 heading

默认策略：

```text
section 是结构边界
heading 跟随后续正文
```

heading 默认不重复进入 `sparse_text`。

### 5.6 原子 Block

当前默认原子类型：

```text
code
formula
table
image
```

超长原子 Block 不拆，单独形成 Chunk，并允许超过 token 上限。

### 5.7 普通 Chunk 不超 token 上限

普通 text/list Chunk 通过：

```text
token packing
overlap trim
```

保持在 `chunk_tokens` 以内。

### 5.8 dense 和 sparse 使用同一个身份

Qdrant 中：

```text
dense point id == sparse point id
```

原始 `chunk_id` 保存在 payload。

## 6. 已实现内容

### 6.1 Source 与路由

- 本地文件加载
- URL 下载
- 协议检查
- 内网地址拒绝
- 文件大小限制
- 请求超时
- MIME / 扩展名 / magic bytes 路由

### 6.2 Parser

已实现：

```text
HTML
Markdown
plain text
PDF
```

HTML：

```text
Trafilatura
Readability
BeautifulSoup code supplement
```

当前 HTML Parser 会比较 Trafilatura 和 Readability 的 heading 完整性。

如果 Readability 保留的 heading 更多，且正文长度合理：

```text
采用 readability_structured
```

Markdown：

```text
heading
paragraph
list
code
formula
table
front matter
source line
```

PDF：

```text
MinerU 主 parser
PyMuPDF fallback
```

只有接口外壳：

```text
Docling
Office
```

### 6.3 质量门控与清洗

已实现：

```text
Raw Parse Quality Gate
Block Cleaner
Index Quality Gate
```

质量状态：

```text
high
medium
low
reject
```

索引决策：

```text
high   -> dense + sparse
medium -> dense + sparse, sparse_weight=0.6
low    -> dense only, review_required
reject -> 不索引
```

### 6.4 Block-aware Chunking

已实现：

- `_Piece` 中间结构
- 原子 Block 保护
- heading 跟随正文
- 长 text/list 内部切分
- `chunk_tokens - overlap_tokens` 文本步长
- overlap 回退
- overlap 内部字符截取
- 普通 Chunk token 上限
- 超大原子 Block 标记
- fragments 来源追踪
- ParentNode
- IndexReadyChunk
- full_text / dense_text / sparse_text

### 6.5 JSON 导出

`format_router_demo.py` 可以输出：

```text
artifact.json
document.md
parents.json
chunks.jsonl
```

### 6.6 Qdrant 入库

`pipeline/ingest/qdrant_indexer.py` 已经实现：

```text
读取 artifact.json + chunks.jsonl
-> dense_text embedding
-> sparse_text BM25 vector
-> dense collection
-> sparse collection
-> vocab.json
-> bm25_stats.json
-> index_manifest.json
```

支持：

```text
多个 export-dir
统一重建
稳定 point id
payload 保存 chunk_id / fragments / quality / index_decision
Qdrant local rebuild 清理旧 collection 目录
```

### 6.7 在线检索

已实现：

```text
Dense retrieval
Sparse retrieval
Hybrid + RRF
Rerank
Generation + citations
FastAPI
Web page
```

当前默认链路参数：

```text
candidate_k        = 20
rrf_k              = 60
rerank_candidate_k = 20
final_top_k        = 5
context_k          = 5
```

### 6.8 当前索引能力

可以重建统一 Phase 0 collection。

当前演示语料：

```text
Contextual Retrieval in AI Systems
Building Effective AI Agents
```

当前合并索引结果：

```text
chunks            37
dense points      37
sparse points     36
parents           4
```

有一个 Chunk 没有可用的 sparse 文本，因此只进入 dense。

### 6.9 测试

当前测试：

```text
10 tests
OK
```

覆盖：

- PDF magic 路由
- HTML code block
- Markdown heading / formula
- 文档质量拒绝
- fragments 保留
- chunk token 上限
- overlap
- dense/sparse index decision
- 稳定 point id
- sparse_weight
- Qdrant rebuild 清理旧点

### 6.10 CI

GitHub Actions 会自动运行 Phase 0 测试。

CI 依赖包含：

```text
llama-index-core
trafilatura
beautifulsoup4
markdownify
lxml
readability-lxml
qdrant-client
jieba
```

## 7. 当前未实现内容

这一节是其他 Agent 最需要先看的。

### 7.1 Chunking

未实现：

```text
window_tokens 没有真正参与 embedding 语义边界微调
chunk 没有独立的最小 token 目标
完整的层级 Parent tree 尚未实现
Parent full_text 直接拼接 child chunks，overlap 可能重复
```

### 7.2 Retrieval

未实现：

```text
Parent expansion
hierarchical retrieval
child 命中后自动扩展 parent context
字段化 BM25F
```

当前 sparse 默认只使用 `sparse_text`。

### 7.3 Evaluation

未实现：

```text
固定 qrels
Recall@k
Precision@k
MRR
nDCG
parser quality 评估
chunk quality 评估
dense / sparse / RRF / rerank 分阶段评估
generation faithfulness 评估
```

当前调参仍缺少正式评估集。

### 7.4 Parser

未实现：

```text
Playwright 动态网页渲染
Docling 实际 adapter
Office / MarkItDown 实际 adapter
复杂 PDF 版面还原
MathML / LaTeX 完整性评估
真实 PDF page / bbox 全覆盖
```

### 7.5 OCR 与多模态

未实现：

```text
独立 OCR
OCR confidence 接入质量门控
图片 asset 提取
图片 caption
图片 OCR
多模态 embedding
图片节点检索
```

### 7.6 Index 管理

未实现：

```text
增量追加而不全量 rebuild
多文档 index manifest 版本管理
collection schema version
文档删除 / 更新传播
embedding model 版本迁移
```

当前最安全的做法：

```text
给定多个 export-dir
一次重建统一的 dense / sparse collection
```

### 7.7 Contextual Retrieval

当前项目还没有实现 Anthropic 风格的：

```text
Contextual Embeddings
Contextual Sparse
```

也就是说，目前没有：

```text
整篇 document + 当前 chunk
-> LLM 生成 chunk-specific context prefix
```

现在只是把这篇文章当作普通文档处理。

## 8. 计划新增内容

### 8.1 高优先级

```text
1. 固定评估集和 qrels
2. Recall@k / MRR / nDCG 评估脚本
3. 多文档增量索引
4. Parent expansion
5. 字段化 BM25F
```

### 8.2 中优先级

```text
1. embedding 语义边界微调
2. Contextual Embeddings 实验
3. Contextual Sparse 实验
4. Office parser
5. Docling parser
6. Playwright 页面渲染
```

### 8.3 长线

```text
1. 图片 asset 与 OCR
2. 多模态 embedding
3. PDF bbox 级引用
4. collection schema migration
5. 多租户和权限过滤
```

## 9. 代码地图

```text
docs/
    工程知识、架构说明、复习路线、进度和 finding

issues/
    已知问题、根因和解决状态

pipeline/
    所有可运行实现

pipeline/ingest/
    Phase 0 核心包
```

关键文件：

```text
pipeline/ingest/models.py
    统一数据模型

pipeline/ingest/source.py
    本地文件和 URL 加载

pipeline/ingest/router.py
    格式路由

pipeline/ingest/parsers/
    HTML / Markdown / PDF / optional adapters

pipeline/ingest/cleaner.py
    Block 清洗

pipeline/ingest/quality.py
    Raw Quality Gate 和 Index Decision

pipeline/ingest/chunker.py
    Block-aware Hierarchical ChunkBuilder

pipeline/ingest/pipeline.py
    IngestPipeline 编排

pipeline/ingest/qdrant_indexer.py
    Phase 0 JSON -> Qdrant

pipeline/format_router_demo.py
    摄取 CLI

pipeline/hybrid_retriever.py
    Dense + Sparse + RRF

pipeline/rag_server.py
    FastAPI 和检索接口

pipeline/generation.py
    上下文构造和 Qwen 回答

pipeline/reranker.py
    DashScope reranker

pipeline/sparse_pipeline_demo.py
    当前 Sparse 通道实现

pipeline/tests/
    单元和集成测试
```

旧脚本仍在 `pipeline/` 根目录，主要作为实验和兼容入口。

## 10. 运行环境

```text
Python 3.12
虚拟环境 pipeline/.venv_rag
Qdrant local
DASHSCOPE_API_KEY
```

运行测试：

```powershell
cd pipeline
.\.venv_rag\Scripts\python.exe -m unittest discover -s tests -v
```

摄取 URL：

```powershell
.\.venv_rag\Scripts\python.exe format_router_demo.py `
    --url https://example.com/article `
    --out-dir experiments/example
```

写入 Qdrant：

```powershell
.\.venv_rag\Scripts\python.exe -m ingest.qdrant_indexer `
    --export-dir experiments/example `
    --qdrant-path qdrant_data `
    --dense-collection phase0_dense `
    --sparse-collection phase0_sparse `
    --artifact-dir index_artifacts/phase0_combined `
    --embedding-backend qwen `
    --rebuild
```

启动 FastAPI：

```powershell
$env:RAG_DENSE_COLLECTION = "phase0_dense"
$env:RAG_SPARSE_COLLECTION = "phase0_sparse"
$env:RAG_SPARSE_ARTIFACT_DIR = "index_artifacts/phase0_combined"

.\.venv_rag\Scripts\python.exe -m uvicorn rag_server:app `
    --host 127.0.0.1 --port 8000
```

## 11. 工程约束

### 11.1 不要提交

```text
API key
第三方完整语料
qdrant_data
index_artifacts
backups
日志
实验原始大文件
虚拟环境
```

### 11.2 Qdrant local 锁

运行索引器前必须停止 FastAPI。

否则可能出现：

```text
qdrant_data 文件锁
```

### 11.3 Rebuild

当前重建只对输入 export-dir 集合生效。

如果要保证文档集合一致：

```text
列出全部需要的 export-dir
一次执行 rebuild
```

不要只重建一个文档后继续复用旧 collection 的 sparse vocab。

### 11.4 修改 Parser 后

至少验证：

```text
单元测试
真实 URL
heading
code block
table
section_path
quality score
```

### 11.5 修改 Chunker 后

至少验证：

```text
普通 chunk 不超 token 上限
原子 block 不拆
heading 跟随正文
fragment 字符范围正确
overlap 在 Block 内部可截取
```

### 11.6 修改 Retrieval 后

至少验证：

```text
dense
sparse
hybrid
rerank
answer
同一 chunk_id
sparse vocab 是否匹配
```

## 12. 当前推荐下一步

优先级最高的不是继续换模型，而是建立评估闭环。

推荐顺序：

```text
1. 生成 qrels
2. 测 Dense Recall@k
3. 测 Sparse Recall@k
4. 测 union Recall@k
5. 测 RRF Recall@k / MRR
6. 测 Rerank 前后 nDCG
7. 再调 candidate_k / rrf_k / rerank_candidate_k
8. 最后评估 Parent expansion 和 Contextual Retrieval
```

## 13. 新 Agent 阅读顺序

最短路径：

```text
1. 本文
2. docs/block_aware_chunking_implementation.md
3. docs/top_k_and_candidate_depth.md
4. docs/retrieval_scoring_and_rrf.md
5. docs/ingest_pipeline_progress.md
6. pipeline/tests/test_format_router.py
7. pipeline/tests/test_qdrant_indexer.py
8. pipeline/ingest/chunker.py
9. pipeline/ingest/qdrant_indexer.py
10. pipeline/hybrid_retriever.py
```

## 14. 最终速记

```text
这个项目目前已经做到：
多格式文档 -> Block -> Chunk -> Dense/Sparse -> Qdrant -> Hybrid/RRF/Rerank -> Answer -> FastAPI

这个项目目前还缺：
正式评估集、增量索引、Parent expansion、字段化 BM25F、OCR/多模态、动态网页、Office/Docling

其他 Agent 修改时必须保住：
fragments 字符溯源
Block/Chunk 分离
dense/sparse 同 chunk_id
普通 chunk token 上限
可重建索引
```
