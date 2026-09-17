# 实验 12: Anthropic Contextual Retrieval 走通 Phase 0 全链路

## 1. 实验条件

```text
日期             2026-09-17
URL              https://www.anthropic.com/engineering/contextual-retrieval
chunk_tokens     800
overlap_tokens   400
window_tokens    400
embedding        qwen3-vl-embedding, 1024 维
sparse           jieba + BM25
vector DB        Qdrant local
collections      anthropic_dense / anthropic_sparse
rerank           gte-rerank-v2
原始导出         pipeline/experiments/anthropic_contextual_retrieval/
索引产物         pipeline/index_artifacts/anthropic_contextual_retrieval/
```

## 2. Parser 发现

第一次使用 Trafilatura 输出 Markdown：

```text
parser      trafilatura
blocks      47
chunks      7
quality     high 0.9809
```

问题是文章 heading 全部丢失，所有 chunk 的 section 都退化为 footer 中的：

```text
Get the developer newsletter
```

检查原始 HTML 和 Readability 输出后确认：

```text
Trafilatura          正文完整，但 heading 丢失
Readability DOM      保留 h1/h2/h3/h4
markdownify          可以把 Readability HTML 转回 Markdown heading
```

因此 HTML Parser 改为结构化选择：

```text
Trafilatura Markdown
vs
Readability HTML -> markdownify Markdown

如果 Readability 保留的 heading 更多
且正文长度不低于 Trafilatura 结果的 50%
则采用 readability_structured
```

修复后的结果：

```text
parser      readability_structured
blocks      78
parents     2
chunks      17
quality     high 0.9875
```

## 3. Qdrant 入库结果

```text
dense points        17
sparse points       17
sparse vocabulary   693
point id            同一个 chunk_id 在 dense/sparse 中共用稳定 point id
```

payload 包含：

```text
chunk_id
artifact_id
parent_id
section_path
fragments
quality
index_decision
```

## 4. API 验证

使用旧 FastAPI `rag_server.py`，通过环境变量切换到：

```text
RAG_DENSE_COLLECTION=anthropic_dense
RAG_SPARSE_COLLECTION=anthropic_sparse
RAG_SPARSE_ARTIFACT_DIR=index_artifacts/anthropic_contextual_retrieval
```

### Dense

```text
query       How much does contextual retrieval reduce failed retrievals?
top1 chunk  doc_5c3e74c436ef_c0010
top1 score  0.785798
elapsed     1836 ms
```

### Sparse

```text
query       How much does contextual retrieval reduce failed retrievals?
top1 chunk  doc_5c3e74c436ef_c0001
top1 score  6.718024
elapsed     1 ms
```

### Hybrid + Rerank

```text
query       Why does contextual retrieval reduce failed retrievals?
top chunks  doc_5c3e74c436ef_c0001
            doc_5c3e74c436ef_c0010
            doc_5c3e74c436ef_c0016
rerank      gte-rerank-v2, applied
generation  约 5.2 秒
```

答案能够引用文章中的核心结论：

```text
Contextual Embeddings + Contextual BM25
top-20 retrieval failure rate 降低 49%
组合 reranking 后降低 67%
```

## 5. 发现与修正

### 5.1 HTML heading 丢失

```text
只检查 Trafilatura 是否为空，不足以决定是否使用 fallback。
应该比较提取结果的结构完整性。
```

### 5.2 Qdrant local rebuild 残留

在 Windows 本地 Qdrant 中：

```text
delete_collection()
-> 重新 create_collection()
-> 旧 SQLite 数据仍可能残留
```

原因是 collection 目录未被彻底清理。

修复：

```text
1. 关闭 Qdrant client
2. 删除目标 collection
3. del client + gc.collect()
4. 严格校验 collection 名
5. 删除 qdrant_data/collection/<collection> 目录
6. 重新创建 collection 并写入
```

增加回归测试：

```text
test_rebuild_removes_stale_collection_points
```

## 6. 当前结论

```text
Anthropic Contextual Retrieval 文章已经走通:
URL
-> structured HTML parser
-> DocumentBlock
-> Block-aware chunks
-> Qwen dense
-> BM25 sparse
-> Qdrant
-> 旧 FastAPI
-> Hybrid / RRF / Rerank / Answer
```

## 7. 适用边界

这次结果只代表：

```text
单篇文章
17 个 chunk
Qwen 1024 维 embedding
jieba + BM25
Qdrant local
无正式 qrels 评测
```

因此不能据此证明：

```text
readability_structured 对所有网站都优于 Trafilatura
BM25 或 hybrid 一定优于 dense
chunk 参数 800/400 一定最优
```

## 8. 待验证

```text
增加更多新闻、博客和文档站点的 HTML 对照
建立固定 qrels
评估 section 元数据正确率
评估不同 chunk 参数下的 recall@k / MRR / nDCG
比较 Contextual Embeddings 是否应作为额外表示加入本项目
```
