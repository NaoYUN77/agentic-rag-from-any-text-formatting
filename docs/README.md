# 文档导航

`docs/` 保存持续迭代的工程认知、架构设计和学习笔记。历史实验输出放在 `pipeline/experiments/`，不要把一次性结果和当前结论混在一起。

## 1. 架构与当前进度

| 文档 | 定位 |
|---|---|
| `project_overview_for_agents.md` | **新 Agent 接手总览**：项目目标、架构、已实现、未实现、待新增、约束和代码地图 |
| `format_routing_and_cleaning_plan.md` | 多格式摄取、格式路由、质量门控、Block 与 Chunk 的架构方案 |
| `ingest_pipeline_progress.md` | 当前 Phase 0 实现状态、验证数据、未完成项和下一步 |
| `block_aware_chunking_implementation.md` | 当前 Block -> Piece -> Chunk、overlap、fragments 和 Qdrant 入库实现说明 |

## 2. Chunking

| 文档 | 定位 |
|---|---|
| `rag_chunking_concepts.md` | 通用分块原理、相似度与距离、断点判定、窗口机制 |
| `chunking_strategy_guide.md` | 分块策略选型、成本与实验结论 |
| `semantic_chunking_notes.md` | LangChain 语义切分专栏 |
| `llamaindex_chunking_notes.md` | LlamaIndex Node Parser 与窗口机制专栏 |

## 3. Retrieval

| 文档 | 定位 |
|---|---|
| `sparse_vs_dense_retrieval.md` | 稀疏与稠密召回的共同概念 |
| `sparse_retrieval_concepts.md` | 词项、词表、倒排索引、稀疏向量与 BM25 |
| `retrieval_scoring_and_rrf.md` | 混合检索、多路召回、候选深度、RRF、加权融合、重排与评估 |
| `top_k_and_candidate_depth.md` | Top-K、candidate_k、RRF k、rerank、context_k 与 Recall@k |

## 4. 学习与复习

| 文档 | 定位 |
|---|---|
| `RAG_learn.md` | RAG 入门资源和阅读入口 |
| `RAG_review_route.md` | 复习路线、自测题和阶段检查 |

## 5. 阅读建议

```text
第一次阅读:
  RAG_learn.md
  -> rag_chunking_concepts.md
  -> sparse_vs_dense_retrieval.md

理解当前实现:
  format_routing_and_cleaning_plan.md
  -> ingest_pipeline_progress.md
  -> pipeline/README.md

准备继续开发:
  ingest_pipeline_progress.md 的第 6、7 节
  -> pipeline/tests/test_format_router.py
  -> pipeline/ingest/chunker.py
```

## 6. 文档边界

- `docs/`: 当前认知和持续更新的工程说明。
- `pipeline/experiments/findings/`: 单次实验档案，原则上不覆盖历史结论。
- `issues/`: 已定位缺陷、根因和解决状态。
- 每个工程结论都应注明前提条件、数据来源和复现命令。
