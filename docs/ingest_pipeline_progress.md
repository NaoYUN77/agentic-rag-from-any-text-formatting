# 多格式摄取与 Block-aware Chunking 进度

更新时间：2026-09-16

## 1. 当前结论

项目正在从“解析后的纯文本/Markdown 直接交给 LlamaIndex 切分”，升级为：

```text
URL / PDF / 扫描件 / HTML / Markdown / Office
-> Format Router
-> Parser
-> RawDocumentArtifact + DocumentBlock
-> Raw Parse Quality Gate
-> Block Cleaner
-> Block-aware Hierarchical ChunkBuilder
-> Index Quality Gate
-> ParentNode + IndexReadyChunk
-> Dense / Sparse / RRF / Rerank
-> Vector DB
```

当前准确状态：

- `Block -> Chunk` 已经实现，不再只是设计稿。
- `IndexReadyChunk` 已经可以通过 `ingest.qdrant_indexer` 写入 Qdrant dense/sparse。
- dense 和 sparse 使用同一个稳定 point id，原始 `chunk_id` 保存在 payload。
- 在线服务默认仍使用旧的 `redhat / redhat_sparse`，可以通过环境变量切换到 Phase 0 collection。
- 真正的 embedding 语义边界微调还没有接入 ChunkBuilder。
- 当前 ChunkBuilder 是“结构感知 + token 打包 + overlap”，不是完整的语义窗口切块器。

## 2. 已实现的摄取能力

### Source 与路由

- 支持本地文件加载。
- 支持 URL HTTP 加载。
- URL 加载包含协议限制、内网地址拒绝、大小限制和超时限制。
- 支持扩展名、MIME、magic bytes 联合判断格式。
- 路由结果包含主 parser、fallback parser、置信度和判定原因。

### Parser

- Markdown / plain text -> `DocumentBlock`
- HTML -> Trafilatura 正文提取
- HTML 代码块补强 -> BeautifulSoup
- PDF -> MinerU
- PDF fallback -> PyMuPDF
- Office、Docling 留有可选 adapter 接口
- 统一输出 `DocumentArtifact` 和 `DocumentBlock`

### Cleaner 与质量门控

- Raw Parse Quality Gate，用于决定是否换 parser 或重新解析。
- Block Cleaner，用于清理 block。
- Index Quality Gate，用于决定索引策略：
  - `high`：dense + sparse
  - `medium`：dense + sparse，`sparse_weight=0.6`
  - `low`：仅 dense，并标记 review
  - `reject`：不进入索引

### Block-aware ChunkBuilder

当前实现位于：

`pipeline/ingest/chunker.py`

核心规则：

- `section_path` 作为硬边界。
- `heading` 跟随后续正文，不单独生成最终检索 chunk。
- `code`、`formula`、`table`、`image` 作为原子 Block。
- 长 `text` Block 可以在内部继续切分。
- 每个 chunk 保存 `fragments`：
  - `block_id`
  - `start_char`
  - `end_char`
  - `block_type`
  - `text`
- 一个 Block 可以出现在多个 chunk 中。
- 一个 chunk 可以引用多个 Block 或 Block 的局部范围。
- Chunk 同时输出：
  - `full_text`
  - `dense_text`
  - `sparse_text`
  - `token_count`
  - `char_count`
  - `page_start`
  - `page_end`
  - `quality`
  - `index_decision`
- `sparse_text` 默认不重复塞入 heading，避免提高 `df` 并降低关键词区分度。

## 3. Block 与 Chunk 的关系

```text
Block 是文档结构单位。
Chunk 是面向检索的索引视图。
```

```text
Document
  -> Block Tree
      -> Chunk View
          -> fragments
          -> full_text
          -> dense_text
          -> sparse_text
          -> index decision
```

Chunk 不是 Block 的替代品，也不应该复制 Block 的全部结构信息。

## 4. 最新修复

本轮检查发现两个实际问题：

1. 旧 overlap 只回退完整 Block 片段。
   当正文片段接近 `800 token` 时，`overlap_tokens=400` 常常直接退化成 `0`。
2. overlap 回填后继续追加 Block，可能让普通 chunk 超过 `800 token`。

已修复：

- 长正文按 `chunk_tokens - overlap_tokens` 切成可重组片段。
- overlap 支持落在 Block 内部，不再只回退完整片段。
- 在追加新 Block 前重新裁减 overlap。
- 普通 chunk 强制不超过 `chunk_tokens`。
- 超大原子 Block 仍允许单独超限，并标记 `oversized_atomic_block`。

对应代码：

- `pipeline/ingest/chunker.py`
- `pipeline/tests/test_format_router.py`

## 5. 已验证数据

### 单元测试

执行目录：`pipeline`

```powershell
.\.venv_rag\Scripts\python.exe -m unittest discover -s tests -v
```

结果：

```text
Ran 9 tests
OK
```

覆盖内容：

- PDF magic 路由
- HTML code block
- Markdown heading / formula
- 空文档质量拒绝
- code block 与 fragments 保留
- 长度上限与 overlap
- IndexDecision 对 dense/sparse 分支的控制
- dense/sparse 共用稳定 point id
- sparse_weight 真正乘入 BM25 document vector

### 合成长文本测试

参数：

```text
chunk_tokens=100
overlap_tokens=40
```

结果：

```text
没有 chunk 超过 100 token
内部边界 overlap 实际为 40
所有 chunk 都有 fragments
fragments 可以精确还原 Block 的字符范围
```

### 真实 Markdown CLI 测试

输入：

`docs/format_routing_and_cleaning_plan.md`

结果：

```text
parser: markdown
quality: high 0.9804
blocks: 245
parents: 1
chunks: 49
over limit: 0
fragments ok: true
```

该文档没有产生 overlap，是因为它主要由短 section 和 heading 硬边界组成。overlap 能力由合成长文本测试单独验证。

### Phase 0 JSON -> Qdrant 端到端验证

```text
dense points: 49
sparse points: 49
sparse vocabulary: 842
manifest: ok
```


### 早期样本验证

以下数据来自修复前的早期验证，仅作为 parser 和 pipeline 可行性记录：

```text
Red Hat Markdown:
blocks 226
quality high 0.9894
parents 8
chunks 62

Anthropic URL:
parser trafilatura
blocks 52
quality high 1.0
parents 1
chunks 9
```

## 6. 当前未完成事项

### Chunking

- `window_tokens` 目前只是保留参数，没有真正参与语义边界调整。
- 还没有把 embedding 语义距离接入 ChunkBuilder。
- `ParentNode` 当前按较粗的 `section_path[0]` 分组，不是完整层级树。
- `ParentNode.full_text` 直接拼接 child chunk，overlap 内容可能重复。
- 还没有评估不同 chunk 参数在真实 qrels 上的 recall、precision 和 MRR。

### 检索链路

- Phase 0 JSON 已经可以写入 Qdrant dense/sparse collection。
- dense 和 sparse 共用稳定 point id，并通过 payload 保留原始 `chunk_id`。
- 在线 RAG 服务默认仍使用旧的 LlamaIndex chunks，需要通过环境变量切换 collection。
- Parent expansion 和 hierarchical retrieval 尚未实现。
- 字段化 BM25F 尚未实现。

### 解析能力

- Playwright 动态网页渲染尚未接入。
- Docling 英文 PDF / 数学公式 adapter 尚未接入。
- Office / MarkItDown adapter 尚未接入。
- 图片 asset 提取尚未完整实现。
- 图片 OCR、caption、多模态 embedding 尚未实现。
- OCR confidence 尚未真正接入质量门控。
- 真实 PDF page / bbox 信息仍有缺失。

### 评估

- 需要重建固定评估集和 qrels。
- 需要补充 recall@k、MRR、nDCG。
- 需要拆开评估：
  - parser quality
  - chunk quality
  - dense recall
  - sparse recall
  - RRF quality
  - rerank gain
  - generation faithfulness

## 7. 下一步建议

优先级从高到低：

1. 用新 ChunkBuilder 重跑当前全部语料，比较新旧 chunk 分布。
2. 用 Phase 0 索引启动服务，验证 Dense、Sparse、RRF 和 Rerank。
3. 实现 Parent expansion，检索 leaf，返回 parent context。
4. 重建 qrels，对 dense、sparse、RRF、rerank 分阶段评测。
5. 再接入 embedding 语义边界微调。
6. 最后扩展动态 URL、Office、图片和 OCR 能力。

## 8. 关键文件

- `pipeline/ingest/models.py`
- `pipeline/ingest/router.py`
- `pipeline/ingest/source.py`
- `pipeline/ingest/cleaner.py`
- `pipeline/ingest/quality.py`
- `pipeline/ingest/chunker.py`
- `pipeline/ingest/pipeline.py`
- `pipeline/ingest/qdrant_indexer.py`
- `pipeline/tests/test_qdrant_indexer.py`
- `pipeline/format_router_demo.py`
- `pipeline/tests/test_format_router.py`
- `docs/format_routing_and_cleaning_plan.md`

## 9. Git 状态

- `v0.0.1` 位于 `main`，已经提交并推送到 GitHub。
- Phase 0 摄取链路记录在 `feature/phase0-ingest-structure` 分支，作为 `v0.0.2` 发布准备。
- 不应提交 API Key、第三方语料、Qdrant 数据、index artifacts、backups 和日志。
