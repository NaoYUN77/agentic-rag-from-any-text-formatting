# Changelog

## Unreleased

### Added

- **评估器新增「分差与排序可靠性」一节**（`eval/run_eval.py::gap_analysis`）：
  issue 03 观察到「两个块分差 0.0001、基本并列」，怀疑分差小 == 排序不可靠。
  按 `top1 - top2` 的 dense 分差中位数分两组比较命中率，把猜测变成数字。
  实测（39 条 / 可回答 34 条）：分差小组（≤0.0429，n=17）hit@5 **0.882**，
  分差大组（≥0.0470，n=17）**1.000**，差 0.118（约 2 条）——
  **方向与猜测一致，但每组仅 17 条，不足以定论**。
  ⚠️ 必须用 **hit@5** 而不是 MRR：分差小天然让 MRR 偏低（top-1 与 top-2 谁在前
  对 MRR 影响大），拿 MRR 分析等于**循环论证**；回归测试把这点钉住了。
  新增 `GapAnalysisTests`（5 例）。
- **质量门新增「结构保真度」维度**（`ingest/quality.py::_structural_fidelity`）：
  此前质量门只看 token 数、长度分布、重复率、乱码率 —— 对「结构丢了」完全无感。
  issues 11/12 的三次回归（heading 层级被压平、`section_path` 大面积为空、
  PDF 页码完全丢失）当时**一个都没被发现** —— 一本讲配置的手册丢了全部配置示例，
  质量分仍是 0.99。新增三个判据（都取「有前提才检查」，避免假警报）：
  `pdf_without_page_numbers`（仅 PDF）、`flat_heading_levels`（标题 ≥5 且层级 ≤1）、
  `section_path_all_empty`（标题 ≥3 且正文覆盖率 = 0）。
  每个 flag 只扣 0.06 分 —— 目的是**可见** + 压到 medium 提示人工看一眼，
  不是把文档一棒打死（结构坏了内容往往还在，仍值得进索引）；并写进 `reasons`。
  实测零误报：真实 PDF `score=0.9902 flags=[]`，7 篇 URL 语料全部 ok。
  新增 `tests/test_quality.py`（8 例）。
- **VoyageAI rerank 客户端**（`reranker.VoyageReranker`，模型 `rerank-2.5-lite`）：
  百炼的 rerank 免费额度已耗尽（403 `FreeTierOnly`），改走 VoyageAI。
  ⚠️ **两家协议不同，不能只换 URL** —— Voyage 的请求体是**扁平**的
  （`query`/`documents`/`top_k` 在顶层），结果是 `data[]`；
  DashScope 则是 `input.query` + `parameters.top_n`，结果在 `output.results[]`。
  新增 `build_reranker(name)` 工厂（`voyage` / `dashscope` / `none`）；
  `run_eval.py` 加 `--reranker`（默认 voyage，可用 `RAG_RERANKER` 覆盖），
  `rag_server.py` 同步改走工厂。
  自带 **429 退避重试**（Retry-After 优先，其次指数退避封顶 60s）——
  实测该账号**未绑支付方式**时限速只有 **3 RPM / 10K TPM**，
  一次评估要打 39 次，不重试几乎全被丢弃（实测 37/39 失败）。
  官方限速表：绑卡后（Tier 1）是 **2000 RPM / 4M TPM**（差 667 倍），
  且「Even with a payment method entered, the free tokens will still apply」。
  **另加主动限速** `min_interval`（`VOYAGE_RERANK_MIN_INTERVAL`）：
  限速按分钟计，撞 429 再退避等于白打一次请求、还要等一整轮窗口，
  已知档位时直接按间隔发更省。
  ⚠️ **TPM 的 token 口径很反直觉**（官方 FAQ）：`query_token × 文档数 + 所有文档 token 和`
  —— 20 个候选时约 `15 × 20 + 20 × 500 ≈ 10,300` token，**光 query 就被乘了 20 倍**。
  未绑卡档位 10K TPM → 每分钟只能 **0.97 次** → 间隔须 ≥ **62 秒**，
  39 条评估因此有 **~40 分钟的物理下限**。
  ⚠️ 别把 `min_interval` 设小了"省时间"：设 41s（每分钟 1.46 次）会让**每次调用都超限**，
  重试不断、总时长反而涨到 **1 小时以上**（实测踩过）。
  提速只有两条路：绑卡（4M TPM），或把 `candidate_k` 降到 10（约 20 分钟）。
  重试次数默认保持 **4**（退避是复利式的，7 次最坏等 182s×39 条）；
  批量场景应靠 `min_interval` 提前限速，而不是撞了再等。
  新增 `tests/test_reranker.py`（16 例，含主动限速的间隔断言）。
- **评估器新增「拒答能力（负样本）」一节**（`eval/run_eval.py`）：
  qrels 里有 5 条 `unanswerable`，注释写着「考系统会不会硬答」，但评估器只算检索
  指标、而检索**永远返回 top-k** —— 这个点一直没有对应的数字。
  新增 `abstention_analysis()`，用 **top-1 的 dense 余弦相似度**定拒答阈值
  （有界、跨查询可比；RRF 分数是排名派生的，不可比），
  推荐阈值取「**误拒为 0** 的前提下抓住最多负样本」。
  实测：可回答 `min=0.528 p50=0.672`，负样本 `min=0.382 p50=0.487`，
  **阈值 0.5285 可拒答 3/5 且误拒 0/34**；两类分布有重叠，无法 100% 分开。
- 新增 `pipeline/ingest/qdrant_indexer.py`，支持把 Phase 0 的 `artifact.json + chunks.jsonl` 写入 Qdrant。
- dense 使用 `full_text` 生成 embedding，sparse 使用 `sparse_text` 生成 jieba + BM25 sparse vector。
- dense 和 sparse 使用同一个稳定 point id，并把原始 `chunk_id`、`fragments`、质量和索引决策写入 payload。
- 新增 `pipeline/tests/test_qdrant_indexer.py`。
- 新增 Anthropic Contextual Retrieval Phase 0 全链路 finding。
- 新增 `pipeline/scan_chunk_tokens.py`：`chunk_tokens` 灵敏度扫描（纯本地，不写索引）。
- 新增 `pipeline/sweep_chunk_params.py`：端到端扫描（重建语料 → 建索引 → 跑评估），每个变体独立路径、不删既有目录。
- 新增 `pipeline/sweep_ranking.py`：排序参数扫描（复用已有索引，不重建），支持 `--stages`。
- 新增 `pipeline/tests/test_pdf_section_path.py`、`test_pdf_outline_calibration.py`、`test_heading_stack.py`、`test_artifact_id.py`。
- `rebuild_phase0_index.py` 新增 `--chunk-tokens`（此前硬编码 800），并把参数写进 manifest。
- `pdf_common.py` 新增 `_OutlineMatcher`（精确优先 + 字符二元组 Jaccard 模糊兜底，阈值 0.85）与 `_calibrate_font_levels`（用 outline 实测层级校正字号排名阶梯）。
- `ingest/models.py` 新增 `stable_artifact_id()`。

### Changed

- **清洗阶段丢弃 PDF 目录（TOC）块**（`ingest/cleaner.py`）：新增 `is_toc_block()`，
  判据是「多级编号 + 标题 + 结尾页码」，实测该 PDF 的目录块命中率 0.79（31/39 行），
  其余 125 个 block 全部 < 0.2 —— 零误报。
  目录是纯导航内容，留在索引里只会制造假命中（查「KEEPALIVED 概述」命中目录条目
  而非正文）；实测该块 5300 字符、会参与召回。
  ⚠️ 两个防误判要点：① 要求 `(\.[0-9]+)+` 至少一层，否则会把有序列表
  （`1. 第一步`）误判成目录；② 丢内容必须留痕 —— `clean_artifact()` 会往
  `artifact.warnings` 写 `dropped_toc_blocks: <id>`。
  **URL 语料不受影响**（chunk_id 序列逐位一致）—— 只作用于 PDF。
  新增 `tests/test_cleaner.py`（7 例）。见 `issues/12`。
- **parent 分组改用完整 `section_path`**（`chunker._scope()`）：原实现只取
  `section_path[0]`，实测在两类文档上退化 —— ① 顶层只有一个标题时（如
  `openai_scaling_storage` 的 9 个章节全挂在同一个 lv1 下）第一层恒为同一个值；
  ② 完全无 heading 的文档 `section_path` 全为空。两者都导致 parent 退化成"整篇一个组"。
  改为用完整路径后 **parents 54 → 103**，单 parent 最大 chunk 数 **11 → 4**，
  parent token p90 **1773 → 745**，>1500 token 的 parent **7 个(13%) → 2 个(2%)**。
  标题块需把自己补到路径末尾（标题的 `section_path` 是祖先链、不含自己），
  否则会与自己的正文分家。**检索指标完全不变**（chunks 113→113、总 token 34,662→34,662、
  `full_text`/`sparse_text` 逐字相同、全文集合 SHA256 相同）—— 因为 chunk 边界由
  "遇 heading 就收口"决定，而新分组恰好与 heading 边界重合，所以是**纯 metadata 改进**。
  详见 `docs/parent_grouping_scope_fix.md`、`issues/10`。
- **新增 `parent_tokens` 参数**（默认 `1600` = 2 × `chunk_tokens`；`0` 表示不限制）：
  完全无 heading 的文档没有结构信号，整篇会落进一个分组 → parent ≈ 整篇文档。
  超限的分组在**跑完之后**按 chunk 顺序拆成多个 parent。实测 parent token max
  **2378 → 1543**、>1600 token 的 parent **2 → 0**、parents 103 → 105，
  而 **chunks / 总 token 完全不变**（全文集合 SHA256 相同）。
  有结构的分组不受影响（p90 只有 745 token）—— 属**纯安全阀**，不是常规切分手段。
  ⚠️ 这是 `parent expansion` 上线前的前提：没有它，`final_top_k=5` 全展开最坏
  约 11,890 token context；有它则 ≤ 8,000。
  新增 `--parent-tokens` CLI 与 manifest 字段。
  新索引：`index_artifacts/phase0_capped`，collections `phase0_capped_dense` / `_sparse`；
  `rag_server.py` 默认 collection 随之更新。
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

- **废弃整条「语义切片 / 滑动窗口」链路**（2026-09-21）。此前 overlap 与滑动窗口都属于
  "固定切片 + 语义切片"这条探索路线，现决定整体放弃，content 先做纯固定大小切块，
  语义方向留待后续重新讨论。删除的文件：

  | 类 | 文件 | 说明 |
  |---|---|---|
  | LlamaIndex 固定+语义切片 | `llamaindex_fixed_semantic_splitter.py`、`llamaindex_fixed_semantic_demo.py`、`rebuild_llamaindex_corpus.py`、`ingest_markdown_corpus.py` | 产出 `redhat` collection 的那条链路 |
  | 滑动窗口 demo | `llamaindex_window_demo.py`、`sentence_window_demo.py`、`window_sweep_demo.py`、`hybrid_chunk_demo.py` | 边界窗口 / 句窗语义微调实验 |
  | 旧端到端管线 | `rag_pipeline.py`、`preprocess.py` | MinerU 清洗 + 语义微调，已被 Phase 0 取代 |
  | 单句窗口切块器 | `ingest/sentence_chunker.py`（+ `--chunker sentence` 选项、`window_sentences` 参数） | 生产里最后一个"滑动窗口"机制 |

  - `rag_server.py` 默认 collection 由 `redhat` / `redhat_sparse` 改为
    `phase0_noov_dense` / `phase0_noov_sparse`，`RAG_SPARSE_ARTIFACT_DIR` 默认指向
    `index_artifacts/phase0_noov`；启动错误提示改为指向 `rebuild_phase0_index.py`。
  - README 的快速开始、文件树、用法示例不再引用已删脚本。
  - `ingest/pipeline.py` 的 `ingest()` 移除 `chunker` / `window_sentences` 参数。
  - `tests/test_format_router.py` 删除 `SentenceSplitTests` 与
    `SentenceWindowChunkBuilderTests`（共 9 例）。
- **把嵌入模型封装从 demo 文件抽成正式模块** `pipeline/embeddings.py`
  （`build_embeddings` / `QwenTextEmbeddings` / `QwenVLEmbeddings` / `HashingEmbeddings` /
  `CN_SENTENCE_SPLIT_REGEX`）。此前这些**生产基础设施**住在 `semantic_chunker_demo.py`
  这个演示脚本里，`rag_server.py` / `rebuild_phase0_index.py` / `eval/run_eval.py` /
  `ingest/qdrant_indexer.py` 反过来依赖一个 "demo" 文件，命名与实际角色不符。
  抽取后 `semantic_chunker_demo.py` 本身被删除，8 个文件的 import 改为 `from embeddings import ...`。
- **取消 chunk 的 `overlap`**（`BlockAwareHierarchicalChunkBuilder`）：移除 `overlap_tokens`
  参数、`IndexReadyChunk.overlap_from_previous` 字段、以及 `_tail_overlap()` / `_slice_tail()`
  两个方法。收口逻辑简化为纯粹的"造块 + 清空累积"。
  - **原因（实测）**：overlap 分布是**双峰**的，且与 heading 精确互补 —— 有 heading 的文档
    几乎不触发（章节本身即语义边界），无 heading 的文档则占到该文 token 的 **43%~47%**。
    全局 14/117 chunks（12%）带 overlap，占 5,407/40,069 token（**13.5%**）。
    重叠文本会同时进两个 chunk 的向量 → 同一内容可被双命中，挤占 `candidate_k`。
  - **收益侧**：overlap 真正起作用的只是"接住跨块指代"，实测它携带的 2~6 句里只有第 1 句在起作用。
  - **影响**：chunks 117 → **113**，总 token 40,069 → **34,662（−13.5%）**。
    减少量 5,407 与独立测得的 overlap token 数**逐位相同**；各文档的片段覆盖字符数不变
    （无内容丢失）。检索指标 **hit@5 全阶段持平**（dense 0.941 / sparse 0.765 / rrf 0.971），
    rrf 的 recall@20 与 MRR@20 也完全一致 → **overlap 对检索质量没有可测收益，纯属成本**。
  - `rebuild_phase0_index.py` 的 `--overlap-tokens` 已移除；manifest 仍写 `"overlap_tokens": 0`
    以保持产物可追溯。
- **删除滑动窗口参数 `window_tokens`**：它从未参与任何计算（只被接收并保存）。属于"语义切片"
  的规划范围（用滑动窗口算 embedding 语义距离、在距离峰值处切分），该方向暂缓，故连同参数一并
  删除，避免留下"看起来能用、实际是死参数"的坑。同步移除 `format_router_demo.py` 的
  `--overlap-tokens` / `--window-tokens`，`scan_chunk_tokens.py` / `sweep_chunk_params.py`
  的 overlap 统计与 `--overlap-ratio`。
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

### Fixed

- **评估器 `--stages` 只请求下游阶段时静默给 0 分**（`eval/run_eval.py`）：
  `union` / `rrf` / `rerank` 都拿 dense + sparse 当输入，但 dense/sparse 只在
  **自己出现在 `--stages` 里**时才被计算。于是 `--stages rrf` 会拿两个空列表做融合，
  **指标全 0 却不报错** —— 看起来像「模型效果差」。实测：单独跑 rrf 得 0.000，
  跑全链路得 0.971。
  新增 `required_stages()`：请求下游阶段时自动带上 dense + sparse；
  只测 dense 时不会白跑 sparse（省一半检索开销）。
  回归测试 `StageDependencyTests`（4 例）。

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
