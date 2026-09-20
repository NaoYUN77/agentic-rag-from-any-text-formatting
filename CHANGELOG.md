# Changelog

## Unreleased

### Added

- 新增 `pipeline/ingest/qdrant_indexer.py`，支持把 Phase 0 的 `artifact.json + chunks.jsonl` 写入 Qdrant。
- dense 使用 `full_text` 生成 embedding，sparse 使用 `sparse_text` 生成 jieba + BM25 sparse vector。
- dense 和 sparse 使用同一个稳定 point id，并把原始 `chunk_id`、`fragments`、质量和索引决策写入 payload。
- 新增 `pipeline/tests/test_qdrant_indexer.py`。
- 新增 Anthropic Contextual Retrieval Phase 0 全链路 finding。
- 新增 `pipeline/scan_chunk_tokens.py`：`chunk_tokens` 灵敏度扫描（纯本地，不写索引）。
- 新增 `pipeline/sweep_chunk_params.py`：端到端扫描（重建语料 → 建索引 → 跑评估），每个变体独立路径、不删既有目录。
- 新增 `pipeline/sweep_ranking.py`：排序参数扫描（复用已有索引，不重建），支持 `--stages`。
- 新增 `pipeline/tests/test_pdf_section_path.py`、`test_pdf_outline_calibration.py`、`test_heading_stack.py`、`test_artifact_id.py`。
- `rebuild_phase0_index.py` 新增 `--chunk-tokens` / `--overlap-tokens`（此前硬编码 800/400），并把参数写进 manifest。
- `pdf_common.py` 新增 `_OutlineMatcher`（精确优先 + 字符二元组 Jaccard 模糊兜底，阈值 0.85）与 `_calibrate_font_levels`（用 outline 实测层级校正字号排名阶梯）。
- `ingest/models.py` 新增 `stable_artifact_id()`。

### Changed

- **`artifact_id` 改为确定性派生**（`stable_artifact_id`：`final_uri` → `uri` → 内容 `sha256`，取前 12 位十六进制）。原实现 4 个 parser 都用 `uuid.uuid4()`，导致每次重建 `chunk_id` 全变，无法做增量更新与逐 chunk 对比。保持 `doc_<12hex>` 形态不变。
- **修复 `html.py` / `markdown.py` 的 `heading_stack` bug**：弹栈条件由 `while len(stack) >= level` 改为 `while stack and stack[-1][0] >= level`（栈改存 `(level, title)`）。原写法拿"栈深度"与"heading 层级"两种量纲相比，使**同级标题被 append 成前一个同级标题的子节点**。实测 `section_path` 在 80/117 chunks (68%) 上变化，parents 16 → 54。**注意：chunk 全文未变，检索指标逐位相同** —— 该修复的价值在 metadata / 引用 / Parent expansion，不在这套检索指标上。
- `pdf_common.py` 修复 `_heading_size_levels` 的 `min(idx+1, 6)` 上限（会把第 7 档字号压平到同一级），并在 `build_artifact` 中改用 `_calibrate_font_levels` 的结果。
- `eval/run_eval.py` 新增「阶段失败告警」：rerank 单条失败原被 try/except 吞掉、按 0 分计入，会把"配额耗尽/鉴权失败"显示成"模型效果差"。现在报告与控制台都会打印失败率与首个错误。
- `eval/run_eval.py` docstring 用法修正：`--chunks` 必须传每篇文档各自的 `chunks.jsonl`，不能传 `index_artifacts/<x>/chunks.jsonl`（该文件不含 `fragments`）。
- HTML Parser 会比较 Trafilatura 与 Readability Markdown 的 heading 完整性，优先保留结构更完整的正文。
- `rag_server.py` 支持通过 `RAG_QDRANT_PATH`、`RAG_DENSE_COLLECTION`、`RAG_SPARSE_COLLECTION`、`RAG_SPARSE_ARTIFACT_DIR` 切换索引。
- FastAPI 检索和引用响应增加 `chunk_id`、`artifact_id`。
- Qdrant local rebuild 会在关闭客户端后清理目标 collection 目录，避免旧 artifact 的 point 残留。
- **heading 从 chunk 正文剥离，只留 metadata**（`BlockAwareHierarchicalChunkBuilder.strip_headings`，默认 `True`）：`_render_block` 对 heading 渲染成 `""`，标题清单存入 `chunk.metadata.headings`，祖先链已在 `section_path`。纯 heading 的 chunk（如 PDF 封面标题）被 `_make_chunk` 显式丢弃（`full_text` 为空）。dense embedding 现在用 heading-free 的 `full_text`。`sparse_text` 本就不含 heading，所以稀疏检索与 gold 反查**零影响**。
- **删除冗余的 `dense_text` 字段**：实测与 `full_text` 100% 相同。索引器 `qdrant_indexer._chunk_payload` 改为直接用 `full_text` 生成 dense embedding；payload 新增 `headings` 字段（供生成阶段补背景，旧 chunks.jsonl 无此字段时优雅降级）。
- **生成阶段用 metadata 补背景**（`generation.build_context` 新增 `include_background`，默认 `True`）：每段正文前加一行 `背景: 文档《x》；章节路径: …；本段涵盖小节: …；出处: url`，用 `section_path` / `headings` / `url` 把被剥离的标题结构还原给 LLM。可关闭做 A/B。
- `rebuild_phase0_index.py` 新增 `--keep-headings`（反义 `--strip-headings` 默认值）与 `--dry-run`（只写 chunks.jsonl、不写 Qdrant）。
- **向量 / 重排序模型升级**：embedding 由 `qwen3-vl-embedding`（多模态接口）换成 `qwen3.7-text-embedding`（通用文本向量接口，默认 1024 维，201 语种，128K token）；rerank 由 `gte-rerank-v2` 换成 `qwen3.7-text-rerank`。
  - 新增 `semantic_chunker_demo.QwenTextEmbeddings`：走 `/services/embeddings/text-embedding/text-embedding`，请求体 `input.texts`、响应 `text_index`，并支持 `text_type=query/document` 非对称检索（入库 document、检索 query）。**注意这是与多模态接口不同的 endpoint，不能只改模型名。**
  - `rag_server.py` 的 `/api/info` 改为动态上报实际 embedding 模型名（不再硬编码）。

### Removed

- **移除 `pdf_mineru` 解析器**（`pipeline/ingest/parsers/pdf_mineru.py`），PDF 路由的
  fallback 由 `[pdf_mineru, pdf_pymupdf]` 改为 `[pdf_pymupdf]`。它是唯一经
  `PDF → Markdown → Block` 中转的解析器：
  - Markdown 表达不了页码/坐标/字号，实测其产物 `font_size` 全为 `null`；
  - heading 层级被压平（只剩 lv1/lv2，见 `issues/11`）；
  - 依赖外部 CLI `mineru-open-api`，且 `flash-extract` 有 20 页上限。

  代价：**PDF 侧不再有任何 OCR 链路**，扫描件/纯图片 PDF 无法解析
  （`pdf_pdfplumber` 与 `pdf_pymupdf` 都只处理文本层）。
  至此架构中不再存在经 Markdown 中转的解析器，`parse_markdown_text` 只服务
  真正的 `.md` 输入。新增不变量测试 `test_no_markdown_transit_parser_registered`。

### Verified

- 10 个单元与集成测试通过。
- Anthropic Contextual Retrieval 文章：78 blocks、17 chunks、17 dense points、17 sparse points。
- 旧 FastAPI 已通过 Dense、Sparse、Hybrid、Rerank 和 Answer 接口验证。
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
