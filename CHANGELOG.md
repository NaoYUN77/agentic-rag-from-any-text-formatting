# Changelog

## Unreleased

### Added

- 新增 `pipeline/ingest/qdrant_indexer.py`，支持把 Phase 0 的 `artifact.json + chunks.jsonl` 写入 Qdrant。
- dense 使用 `dense_text` 生成 embedding，sparse 使用 `sparse_text` 生成 jieba + BM25 sparse vector。
- dense 和 sparse 使用同一个稳定 point id，并把原始 `chunk_id`、`fragments`、质量和索引决策写入 payload。
- 新增 `pipeline/tests/test_qdrant_indexer.py`。

### Changed

- `rag_server.py` 支持通过 `RAG_QDRANT_PATH`、`RAG_DENSE_COLLECTION`、`RAG_SPARSE_COLLECTION`、`RAG_SPARSE_ARTIFACT_DIR` 切换索引。

### Verified

- 9 个单元与集成测试通过。
- Phase 0 JSON 已能在临时 Qdrant 中同时创建 dense 和 sparse collection。
- low quality 的 dense-only chunk 不会写入 sparse collection。

## v0.0.2 - 2026-09-16

### Added

- 新增多格式摄取 Phase 0：
  - 本地文件与 URL Source Loader
  - 扩展名、MIME、magic bytes 格式路由
  - HTML / Markdown / plain text / PDF parser adapters
  - `DocumentArtifact` / `DocumentBlock` 统一模型
  - Raw Parse Quality Gate 与 Index Decision
  - Block Cleaner
  - `BlockAwareHierarchicalChunkBuilder`
  - `ParentNode` / `IndexReadyChunk` / `fragments`
- 新增 `format_router_demo.py` CLI，可导出 artifact、Markdown、parent 和 chunk。
- 新增 `pipeline/tests/test_format_router.py`。
- 新增多格式摄取方案与进度文档。
- 新增 `docs/README.md` 和 `pipeline/README.md` 导航。
- 新增 GitHub Actions，自动运行 Phase 0 单元测试。

### Changed

- 长文本按 `chunk_tokens - overlap_tokens` 生成可重组片段。
- overlap 支持落在 Block 内部。
- 追加新 Block 前重新裁减 overlap，普通 chunk 不再超过 token 上限。
- 更新 `pipeline/requirements.txt`，加入 HTML 抽取与清洗依赖。

### Verified

- 6 个单元测试通过。
- 合成长文本验证 `chunk_tokens=100`、`overlap_tokens=40`：
  - 无 token 超限
  - 内部边界 overlap 为 40
  - fragments 字符范围可还原
- 真实 Markdown CLI 验证：
  - 235 blocks
  - 48 chunks
  - 最大 482 token
  - 超限 0

### Not Yet

- `window_tokens` 尚未接入 embedding 语义边界微调。
- `IndexReadyChunk` 尚未接入 Qdrant dense/sparse 入库。
- Parent expansion、字段化 BM25F 和固定 qrels 评估尚未完成。
- 动态网页、Office、图片 OCR、OCR confidence 和多模态能力尚未闭环。

## v0.0.1

- 初始 RAG 工程实践版本。
- PDF / Markdown 摄取、Dense / Sparse、RRF、Rerank、Generation、FastAPI Web。
