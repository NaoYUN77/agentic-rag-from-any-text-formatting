# v0.0.2 — Phase 0 多格式摄取与 Block-aware 分块

## 概述

把文档摄取链路从「LlamaIndex 分块 + 单一格式」升级为**面向任意文本格式**的统一管线：
多格式路由 → 统一中间表示（`DocumentArtifact` / `DocumentBlock`）→ 两级质量门控 →
自建 Block-aware 分块 → Dense/Sparse 双通道入库。

同时补齐**端到端闭环验证**与**中文 Query 跨语言检索实验**，并把架构现状与规划文档的
差异逐条对照写进文档，避免后来者按过时设计理解代码。

**范围**：`origin/main` (`699ad06`, v0.0.1) → `feature/phase0-ingest-structure` (`4f49d20`)
**规模**：12 个提交，48 个文件，+10605 / −666

---

## 主要改动

### 1. 统一中间表示（架构基石）

新增 `pipeline/ingest/` 子系统，Parser 的唯一职责是把任意格式翻译成 `DocumentBlock[]`：

| 文件 | 职责 |
|---|---|
| `ingest/models.py` | `DocumentArtifact` / `DocumentBlock` / `QualityReport` / `IndexDecision` |
| `ingest/source.py` | Source Loader：本地读取 / URL 下载 |
| `ingest/router.py` | 格式路由：PDF magic bytes → html → markdown → plain text → office |
| `ingest/parsers/markdown.py` | Markdown → Block 状态机（heading_stack 维护 `section_path`） |
| `ingest/parsers/html.py` | HTML 双路择优（trafilatura + readability） |
| `ingest/parsers/pdf_*.py` | MinerU / PyMuPDF 两条 PDF 通路 |
| `ingest/cleaner.py` | 按 block 类型归一化、去样板、去重、重排 `order` |
| `ingest/quality.py` | Raw Parse Quality Gate + Index Quality Gate |
| `ingest/chunker.py` | **自建** `BlockAwareHierarchicalChunkBuilder`（451 行） |
| `ingest/qdrant_indexer.py` | IndexReadyChunk → Dense / Sparse / Qdrant |

**7 种 block 类型**：`text` / `heading` / `list` / `code` / `formula` / `table` / `image`

### 2. Block-aware 分层分块（自建，非 LlamaIndex）

- `ATOMIC_TYPES = {code, formula, table, image}` —— 原子块不切分
- `MERGEABLE_TYPES = {heading, text, list}` —— 可合并成 chunk
- `ParentNode`（章节级上下文）/ `IndexReadyChunk`（检索单元）分离
- **fragments 字符级溯源**：`fragment.text == block.text[start_char:end_char]`
- overlap 通过二分查找做尾部切片，不切断句子

> 规划文档原写「后续仍由 LlamaIndex 负责 chunking」，实际已由自建 ChunkBuilder 承担，
> LlamaIndex 仅降级使用 `SentenceSplitter` 一个组件。

### 3. Dense / Sparse 双通道 + 稳定映射

- Dense：`qwen3-vl-embedding`（1024 维）
- Sparse：`jieba` + BM25 + Qdrant sparse vector
- **dense / sparse 共用 `point_id`**（`blake2b(chunk_id)` 取低 63 位），payload 保留 `chunk_id`

### 4. 端到端闭环验证

新增 `pipeline/verify_phase0_loop.py`，验证五条通路，结果 **PASS**。

### 5. 中文 Query 跨语言检索实验

新增 `pipeline/eval_zh_crosslingual.py`，对照 A/B/C 三方案：

| 方案 | top1 命中 | top5 命中 | sparse 空转 |
|---|---|---|---|
| A 现状（dense+sparse 都用中文） | 8/10 | 10/10 | 4/10 |
| B 分通道（sparse 用英文译文） | 10/10 | 10/10 | 0/10 |
| C 全翻译 | 10/10 | 10/10 | 0/10 |

**结论**：B 与 C 全指标打平，说明翻译 dense query 零额外收益。
但 top5 在 A/B 都是 10/10，dense 单独已能兜住，最终**决策为不做 query 翻译**，
接受 sparse 在中文 query 下退化（生产链路零改动）。

### 6. 文档与问题清单

新增/改写：

- `docs/block_structure_and_blockification.md` —— Block 结构与 block 化全过程
- `docs/plan_vs_implementation.md` —— 规划 vs 实现逐条对照（3 处重大偏差、8 条核心不变量）
- `docs/project_overview_for_agents.md` —— 架构图重画 + 三层分工
- `docs/phase0_loop_verification.md` —— 闭环验证结论与踩坑
- `docs/zh_query_crosslingual_eval.md` —— 跨语言实验对照
- `issues/08_html_code_supplement_placement.md` —— HTML 补充代码块位置错乱（P2 潜伏）
- `README.md` —— 功能介绍与架构图更新为实际实现

---

## 当前索引状态（实测）

```text
语料             10 篇（anthropic_* 7 篇 + openai_* 3 篇）
Blocks           644（text 466 / heading 95 / image 32 / list 30 / code 21）
Chunks           129  IndexReadyChunk
Parents          17   ParentNode
Dense points     129
Sparse points    123
BM25 vocab       4267
bm25             k1=1.2  b=0.75
```

数据来源：`pipeline/index_artifacts/phase0_combined/index_manifest.json`

---

## 测试

```powershell
cd pipeline
.\.venv_rag\Scripts\python.exe -m unittest discover -s tests -v
```

10 个单元测试全部通过（`test_format_router` 6 个 + `test_qdrant_indexer` 4 个）。

---

## 已知未闭环项

- [ ] **语义边界**：`window_tokens=400` 已接收但未参与计算，`_split_text_block()`
      走 `SentenceSplitter` 的句子边界，无 embedding 距离判定
- [ ] **Parent expansion**：检索 leaf 后返回 parent context 未实现
- [ ] **字段化 BM25F**：heading 权重与正文权重未分离
- [ ] **固定评估集**：qrels / recall@k / MRR / nDCG 尚未建立
      （旧 `pre_llamaindex/mixed_qrels.jsonl` 的 `gold` 是裸 `point_id`，语料一换即失效）
- [ ] **扩展 Parser**：Office / 图片 OCR / OCR confidence 尚未接入
      （`docling_optional.py` / `office_optional.py` 仅抛 `NotImplementedError`）

## 已知限制

- 中文 query 下 sparse 通道基本失效（BM25 词表全英文），hybrid 退化为纯 dense
- parent 分组退化：`_scope()` 取 `section_path[0]`，10 篇文档几乎都是 2 个 parent
- heading 层级抽取不准：readability 剥掉网站导航的 h1/h2，正文从 h3 开始，
  导致 71 个 block 挂在同一个 lv3 标题下

---

## 审查建议

1. **优先看** `pipeline/ingest/chunker.py` —— 分块是核心逻辑，451 行
2. **对照看** `docs/plan_vs_implementation.md` —— 快速理解规划与实现的分歧点
3. **验证数据** `pipeline/index_artifacts/phase0_combined/index_manifest.json`
4. **跑测试** 确认 10 个测试通过
5. 已知未闭环项建议作为后续 PR 处理，不阻塞本次合并
