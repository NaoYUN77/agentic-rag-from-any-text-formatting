# Agentic RAG from Any Text Formatting

> 面向多格式文档的 RAG 工程实践: 文档摄取、Block-aware 分块、Dense/Sparse 混合召回、RRF、Rerank、答案生成与引用。
>
> 仓库: <https://github.com/NaoYUN77/agentic-rag-from-any-text-formatting>
>
> 当前稳定版: `v0.0.1`。开发分支: `feature/phase0-ingest-structure`，对应 `v0.0.2` 的格式路由与 Block-aware chunking。
>
> 导航: [文档索引](docs/README.md) | [实现说明](pipeline/README.md) | [变更记录](CHANGELOG.md)

## 功能概览

已实现:

```text
[摄取层] 多格式路由
  PDF (magic bytes) / URL / HTML / Markdown / plain text
  格式判定: MIME + 文件头 magic bytes, 不依赖扩展名

[摄取层] 统一中间表示 (架构基石)
  DocumentArtifact / DocumentBlock[]
  7 种 block 类型: text / heading / list / code / formula / table / image
  block 字段: block_id / order / section_path / parent_id / quality_score

[摄取层] HTML 双路择优 (非串行回退)
  同时跑 trafilatura + readability, 按标题数择优
  parser 命名: readability_structured (择优胜出) / readability_fallback (trafilatura 全败)
  代码块补强: 从原始 HTML 提取 <pre><code> 补齐丢失代码

[摄取层] 两级质量门控
  Raw Parse Quality Gate -> 决定是否换 parser
  Index Quality Gate     -> 决定 dense / sparse 准入
  分级: high >= 0.85 / medium >= 0.60 / low >= 0.35 / else reject

[分块层] Block-aware Hierarchical ChunkBuilder (自建, 非 LlamaIndex)
  ATOMIC_TYPES 原子块不切: code / formula / table / image
  MERGEABLE_TYPES 可合并: heading / text / list
  ParentNode (单 parent 多 child) / IndexReadyChunk
  fragments 字符级溯源: fragment.text == block.text[start:end]
  overlap 回填: 二分查找尾部切片, 不切断句子

[索引层] Dense + Sparse 双通道
  Dense: qwen3-vl-embedding (1024 维)
  Sparse: jieba + BM25 + Qdrant sparse vector
  稳定映射: dense / sparse 共用 point_id (blake2b(chunk_id))

[检索层] Hybrid 检索
  Dense + Sparse 并行召回 -> RRF 融合 (在 candidate_k 深度) -> gte-rerank-v2

[生成层] 答案 + 引用
  retrieval context -> qwen-turbo / qwen-plus -> [C1] 引用标注

[服务层] FastAPI + Web 检索界面
[工程层] 实验输出、工程记录、问题清单与复现命令
```

规划中:

```text
完整格式路由器: Office / 图片 / 更细的 PDF 画像
Docling 英文/数学 PDF 适配
中文/英文 PDF 分流
代码块保真
数学公式 LaTeX 保真
OCR 置信度评分与低质量内容降权
```

## 架构

```text
URL / PDF / Scan / HTML / Markdown / Office
                    |
                    v
              Source Loader
        (local file / HTTP fetch)
                    |
                    v
              Format Router
        MIME + magic bytes 判定
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     HTML        PDF/Scan      Office
        |           |           |
        |     +-----+-----+     |
        |     v           v     v
        |  MinerU    Docling  MarkItDown      [规划中]
        v  / pdf    / docling
  parse_html()
  trafilatura + readability
   双路并行 -> 按标题数择优
        |
        v
  DocumentArtifact / DocumentBlock[]           [统一中间表示]
        |   7 种 block: text/heading/list/code/formula/table/image
        |
        v
  Raw Parse Quality Gate
        |   quality < 阈值 -> 换 parser 重试
        v
  clean_blocks()  (按类型归一化 / 丢样板与重复 / 重排 order)
        |
        v
  Index Quality Gate  ->  dense / sparse 准入
        |
        v
  BlockAwareHierarchicalChunkBuilder           [自建, 非 LlamaIndex]
        |  ATOMIC 不切 / MERGEABLE 合并 / fragments 字符溯源
        |
        +---------------+---------------+
        |                               |
        v                               v
   ParentNode                    IndexReadyChunk
   (context 聚合)                 (检索单元)
        |                               |
        |   [Parent expansion 未实现]     |
        |                               |
        +---------------+---------------+
                        |
                        v
                +-------+-------+
                |               |
                v               v
       Dense Embedding     Sparse / BM25
    qwen3-vl-embedding     jieba + BM25
                |               |
                +-------+-------+
                        |
                        v
                  Qdrant Index
          dense / sparse 共用 point_id
                        |
                        v
                    RRF -> Rerank
              (candidate_k 深度) (gte-rerank-v2)
                        |
                        v
              Answer + Citations -> FastAPI / Web
```

> Phase 0 的 `ParentNode + IndexReadyChunk` 已可通过 `ingest.qdrant_indexer` 写入独立 Qdrant collection（`phase0_dense` / `phase0_sparse`）。在线服务默认仍走旧的 LlamaIndex `FixedSemanticNodeParser` 索引，可通过环境变量切换到 Phase 0 索引。
>
> 详见 [架构与实现差异对照](docs/plan_vs_implementation.md) 与 [Block 结构与 block 化](docs/block_structure_and_blockification.md)。


## 当前 Phase 0 索引状态

```text
语料             10 篇 (Red Hat / NGINX / Anthropic / OpenAI 等)
Blocks           644 个 (text 466 / heading 95 / image 32 / list 30 / code 21)
Chunks           129 个 IndexReadyChunk
Parents          17 个 ParentNode
Dense points     129
Sparse points    123
BM25 vocab       4267

Chunker          BlockAwareHierarchicalChunkBuilder
chunk_tokens     800
overlap_tokens   400
window_tokens    400  (注: 尚未接入 embedding 语义边界)
candidate_k      20
rrf_k            60
final_top_k      5
BM25             k1=1.2, b=0.75
```

> `window_tokens` 目前对语义边界**无实际作用**——`_split_text_block()` 走的是 `SentenceSplitter` 的句子边界，没有 embedding 距离判定。这是 Phase 0 的已知未闭环项。

公开仓库不包含第三方 PDF 和完整 Markdown 语料。语料、Qdrant 数据、索引产物和 API Key 均被 `.gitignore` 排除。

## 快速开始

```powershell
git clone https://github.com/NaoYUN77/agentic-rag-from-any-text-formatting.git
cd agentic-rag-from-any-text-formatting/pipeline

python -m venv .venv_rag
.\.venv_rag\Scripts\python.exe -m pip install -r requirements.txt

$env:DASHSCOPE_API_KEY = "sk-..."

# 将合法语料放入 pipeline/corpus/ 后重建索引
.\.venv_rag\Scripts\python.exe rebuild_llamaindex_corpus.py `
    --chunk-tokens 800 `
    --overlap-tokens 400 `
    --window-tokens 400

# 启动服务
.\.venv_rag\Scripts\python.exe -m uvicorn rag_server:app `
    --host 127.0.0.1 --port 8000
```

页面: <http://127.0.0.1:8000/>

格式路由预览:

```powershell
.\.venv_rag\Scripts\python.exe format_router_demo.py `
    --url https://example.com/article `
    --out-dir experiments/router_demo

.\.venv_rag\Scripts\python.exe format_router_demo.py `
    --file corpus/my_document.md `
    --out-dir experiments/router_demo
```

Phase 0 JSON 入库:

```powershell
.\.venv_rag\Scripts\python.exe -m ingest.qdrant_indexer `
    --export-dir experiments/router_demo `
    --dense-collection phase0_dense `
    --sparse-collection phase0_sparse `
    --artifact-dir index_artifacts/phase0 `
    --rebuild
```

## 仓库结构

```text
CHANGELOG.md  版本变更和阶段成果
.github/      GitHub Actions 与仓库自动化
docs/         工程知识、架构和复习路线
docs/README.md 文档导航入口
issues/       当前缺陷、根因和解决状态
pipeline/     文档处理、检索、生成、服务和实验数据
pipeline/README.md 实现层导航与运行入口
```

---

# 工作区详细导航

> 这个工作区记录从分块到检索的学习与工程实践。
> 核心原则: 每个结论都能追到数据, 每个数据都能重新跑出来。

## 目录结构

```text
python_dev_agent_demo\
├── README.md            本文件(入口导航)
│
├── docs/                ① 工程经验  —— 人写的总结, 会迭代
├── issues/              ② 问题清单  —— 当前实现的缺陷 + 状态追踪
└── pipeline/            ③ 工程实现  —— 代码 + 环境 + 数据
```

**三类东西严格分开:**

```text
docs/        知识沉淀    "我学到了什么"
issues/      缺陷追踪    "现在有什么毛病"
pipeline/    实现与数据  "代码在哪、环境在哪、数据在哪"
```

---

## 一、三分钟看懂

```text
                         ┌──────────────────────┐
                         │   README.md (本文)    │
                         └──────────┬───────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  docs/        │         │  issues/        │         │  pipeline/      │
│  工程经验      │         │  问题清单        │         │  工程实现        │
│               │         │                 │         │                 │
│ "该选哪个"     │         │ "哪里不对"       │         │ "怎么跑"         │
│ "为什么"       │         │ "修到哪一步"      │         │ "数据在哪"       │
└───────────────┘         └─────────────────┘         └─────────────────┘
```

---

## 二、docs/ — 工程经验

| 文件 | 规模 | 定位 | 覆盖 | 数据来源 |
|---|---|---|---|---|
| `chunking_strategy_guide.md` | 357 行 | **选型指南** | 五种策略横向对比、成本、决策树 | `pipeline/experiments/04 05 08 09` |
| `rag_chunking_concepts.md` | 996 行 | **通用原理** | 10 章: 前置工作 / 五步流程 / 相似度vs距离 / 断点判定 / 窗口角色 / 工程结论 / 策略全景 / 相关策略 / 框架对照 | `pipeline/experiments/01 03 05 08 09` |
| `semantic_chunking_notes.md` | 440 行 | **LangChain 专栏** | TextSplitter / Document 结构 / SemanticChunker 参数 / 框架特有的坑 | `pipeline/experiments/01 02 03` |
| `llamaindex_chunking_notes.md` | 548 行 | **LlamaIndex 专栏** | Node Parser 全景 / metadata / SentenceWindow / 800+400 自定义 NodeParser | `pipeline/experiments/07 23 24 25` |
| `sparse_vs_dense_retrieval.md` | 1137 行 | **共用检索概念** | 稀疏/稠密两条召回范式 / 词表 / 倒排索引 / embedding / ANN / 失效模式 | 公式推导 + `pipeline/experiments/14 15 16` |
| `sparse_retrieval_concepts.md` | 1160 行 | **稀疏检索通用原理** | 词项 / 词表 / 倒排索引 / 稀疏向量 / tf / df / IDF / BM25 | 通用概念 + 手算例子 |
| `retrieval_scoring_and_rrf.md` | 1123 行 | **混合检索与 RRF** | 多路召回 / 候选池 / 分数不可比 / RRF / 加权 RRF / Rerank / 评估 | 概念推导 + 当前实现映射 |
| `top_k_and_candidate_depth.md` | 概念文档 | **Top-K 与候选深度** | per-channel top_k / candidate_k / rrf_k / rerank_candidate_k / final_top_k / context_k / Recall@k | 概念 + 当前实现映射 |
| `RAG_learn.md` | 454 行 | **入门索引** | 学习资源清单(10 个, 含 URL), 并指向工作区内部阅读路线 | 无 |
| `RAG_review_route.md` | 668 行 | **复习路线** | Chunk/OpenAI 基线 / 今日状态 / 复习顺序 / 自测题 | `pipeline/experiments/01 ~ 25` |
| `project_overview_for_agents.md` | 项目总览 | **Agent 接手入口** | 架构 / 已实现 / 未实现 / 待新增 / 不变量 / 代码地图 / 运行方式 | 当前代码 + 文档汇总 |
| `format_routing_and_cleaning_plan.md` | 规划文档 | **格式路由计划** | URL/HTML/PDF/扫描件/代码/公式/图片/OCR 质量门控 | 方案设计 |
| `ingest_pipeline_progress.md` | 289 行 | **当前进度** | Phase 0 实现、验证数据、风险和下一步 | 代码 + 测试 + CLI 实测 |
| `block_aware_chunking_implementation.md` | 实现说明 | **当前切块实现** | Block -> Piece -> Chunk、overlap、fragments、Dense/Sparse Qdrant | 当前代码 + 9 个测试 + E2E |
| `block_structure_and_blockification.md` | 概念文档 | **Block 结构与 block 化** | 数据结构 / 7 种 block / heading 状态机 / section_path / 清洁与门控 / 实测数据 / 工程结论 | 源码 + 真实产物 |
| `plan_vs_implementation.md` | 对齐文档 | **规划 vs 实现差异对照** | 3 处重大偏差 + 8 条设计目标逐条核对 + 8 条核心不变量 + 修订建议 | 规划文档 + 当前代码 |

**推荐阅读顺序:**

```text
A. 新 Agent 接手 -> project_overview_for_agents.md
0. 明天复习        -> RAG_review_route.md
1. 刚入门          -> RAG_learn.md
2. 想懂分块        -> rag_chunking_concepts.md 的第三/四/五/七章
3. 想选分块方案    -> chunking_strategy_guide.md 的"选型决策"
4. 想懂共用检索概念 -> sparse_vs_dense_retrieval.md
5. 想懂稀疏内部    -> sparse_retrieval_concepts.md
6. 想懂混合检索与RRF -> retrieval_scoring_and_rrf.md
7. 想用框架        -> semantic_chunking_notes.md 或 llamaindex_chunking_notes.md
8. 规划多格式摄取  -> format_routing_and_cleaning_plan.md
9. 查看当前进度    -> ingest_pipeline_progress.md
10. 想看切块实现   -> block_aware_chunking_implementation.md
11. 搞懂 Top-K       -> top_k_and_candidate_depth.md
12. 搞懂 Block 结构  -> block_structure_and_blockification.md
13. 规划与实现对齐   -> plan_vs_implementation.md
```

---

## 三、issues/ — 问题清单

| 文件 | 层级 | 问题 | 优先级 |
|---|---|---|---|
| `README.md` | — | 总览 + 四层框架 + 优先级 | — |
| `01_data_layer.md` | 数据层 | 目录/版权/正文没分类, Markdown 标记污染 | **P0** |
| `02_constraint_layer.md` | 约束层 | 块粒度不均(431~2138 字) | **P0** |
| `03_representation_layer.md` | 表示层 | 嵌入有损压缩, 排序不准 | P2 |
| `04_channel_layer.md` | 通道层 | 只有稠密一路, 精确匹配弱 | **P1** |
| `05_metadata.md` | 元数据层 | section / page 质量差 | P1 |

**四层框架**(每个问题分属一层, 改一个不会自动修好另一个):

```text
① 数据层   解析保留了结构, 分块当纯文本处理    -> 结构降级为噪声
② 约束层   长度约束单边 + 语义断点分布不均匀    -> 边界漂移, 粒度不一
③ 表示层   嵌入是有损压缩 + 块内多信息点        -> 分辨率不足, 排序不准
④ 通道层   只有稠密一路                        -> 字面匹配能力缺失
```

---

## 四、pipeline/ — 工程实现

```text
pipeline\
├── .venv_rag\              虚拟环境(Python 3.12 + 全部依赖)
├── .dashscope_key          百炼 API Key 文件(占位符, 未使用)
│
├── rag_pipeline.py         ★ 完整流程: 清洗 → 分块 → 向量化 → 入库 → 检索
├── sparse_pipeline_demo.py  ★ 独立 sparse: PDF/MD → 清洗 → jieba → BM25 → top-k
├── hybrid_retriever.py      ★ dense + sparse + RRF
├── generation.py            ★ retrieval context → Qwen answer + citations
├── reranker.py              ★ RRF 候选 → gte-rerank-v2
├── ingest_markdown_corpus.py ★ 增量 Markdown → dense/sparse Qdrant
├── rebuild_llamaindex_corpus.py ★ LlamaIndex 全量重建 dense/sparse
├── format_router_demo.py    ★ URL/文件 -> Block JSON CLI
├── ingest/                  ★ 格式路由器与 parser adapters
│   └── chunker.py           ★ Block-aware Hierarchical ChunkBuilder
├── semantic_chunker_demo.py  语义分块主程序
├── hybrid_chunk_demo.py      四种分块策略对比
├── window_sweep_demo.py      边界窗口宽度 sweep
├── embedding_mechanism_compare.py  单句 vs 窗口嵌入的 AUC 对比
├── embedding_config_compare.py     三种嵌入配置的成本+效果
├── sentence_window_demo.py   句子窗口检索对照
├── llamaindex_window_demo.py LlamaIndex 窗口机制
├── llamaindex_fixed_semantic_splitter.py  LlamaIndex 自定义 NodeParser
├── llamaindex_fixed_semantic_demo.py      800/400 语义切分预览
├── pdf_loader.py             MinerU 调用 + Markdown 清洗
├── check_dashscope_key.py    API Key 诊断
│
├── corpus\                  原始素材
│   ├── redhat_p1-20.md      Red Hat 手册前 20 页(274 句 / 17123 字)
│   ├── extracted.md         同上(另一次运行)
│   └── nginx_guide.md       NGINX 完全指南 200 页(169995 字)
│
├── experiments\             实验数据 + 结论档案
│   ├── README.md            数据索引 + 复现命令
│   ├── 01 ~ 25 *.txt        原始运行输出
│   └── findings\            单次实验的结论
│       ├── README.md
│       ├── 01_breakpoint_methods.md
│       ├── 02_window_width.md
│       ├── 03_embedding_mechanism.md
│       ├── 04_jieba_sparse_pipeline.md
│       ├── 05_sparse_qdrant_storage.md
│       ├── 06_hybrid_rrf.md
│       ├── 07_generation.md
│       ├── 08_generation_model_latency.md
│       ├── 09_rerank_effect.md
│       ├── 10_nginx_ingest.md
│       └── 11_llamaindex_rebuild.md
│
├── backups\                索引与 sidecar 备份
├── index_artifacts\        Phase 0 索引产物 (vocab / BM25 / postings)
└── qdrant_data\            Phase 0: 129 dense + 123 sparse
```

### 环境

```text
Python      3.12.7
虚拟环境     pipeline\.venv_rag\
关键依赖     langchain-experimental / llama-index-core / qdrant-client
             numpy / requests / chromadb
嵌入模型     阿里云百炼 qwen3-vl-embedding (1024 维)
向量库       Qdrant 本地模式(无需 Docker)
```

---

## 五、快速开始

**所有命令都要在 `pipeline\` 目录下执行。**

```powershell
cd E:\Code\python_dev_agent_demo\pipeline
$env:DASHSCOPE_API_KEY = "sk-..."     # 百炼 API Key

# ---- 完整链路: 清洗 → 分块 → 向量化 → 入库 → 检索 ----
.\.venv_rag\Scripts\python.exe rag_pipeline.py `
    --md corpus/redhat_p1-20.md --max-chars 800 --window-chars 400 --rebuild

# ---- 只检索(不重建索引) ----
.\.venv_rag\Scripts\python.exe rag_pipeline.py --only-search --query "你的问题"

# ---- 用 PDF 作为语料(需要 MinerU) ----
.\.venv_rag\Scripts\python.exe rag_pipeline.py --pdf 文件.pdf --pdf-pages 1-20 --rebuild
```

**其它脚本:**

```powershell
# 语义分块(不联网的验证版)
.\.venv_rag\Scripts\python.exe semantic_chunker_demo.py --backend local

# 语义分块(真实模型)
.\.venv_rag\Scripts\python.exe semantic_chunker_demo.py

# 五种策略对比
.\.venv_rag\Scripts\python.exe hybrid_chunk_demo.py --max-chars 800 --window 4
.\.venv_rag\Scripts\python.exe window_sweep_demo.py

# 嵌入机制对比
.\.venv_rag\Scripts\python.exe embedding_mechanism_compare.py
.\.venv_rag\Scripts\python.exe embedding_config_compare.py

# 独立 sparse 链路(不需要 API Key, 不依赖 Qdrant)
$env:SPARSE_QUERY = "你自己的查询"
.\.venv_rag\Scripts\python.exe sparse_pipeline_demo.py `
    --md corpus/redhat_p1-20.md `
    --query $env:SPARSE_QUERY `
    --out-dir experiments/14_sparse_pipeline_artifacts

# 复用 dense collection 的 chunk, 写入独立 sparse collection
.\.venv_rag\Scripts\python.exe sparse_pipeline_demo.py `
    --qdrant `
    --qdrant-path qdrant_data `
    --collection redhat `
    --query $env:SPARSE_QUERY `
    --write-sparse `
    --rebuild-sparse `
    --sparse-collection redhat_sparse
```

---

## 六、当前进度

> Phase 0 多格式摄取、Block-aware chunking 和 JSON -> Qdrant 入库已接通。
> 在线服务默认仍使用旧的 LlamaIndex chunks，可通过环境变量切换到 Phase 0 索引。

### 已完成

```text
[x] 多格式路由       URL / HTML / Markdown / PDF / plain text
[x] 统一文档模型     DocumentArtifact / DocumentBlock / fragments
[x] 质量门控         Raw Parse Quality Gate + Index Decision
[x] Block 清理       heading / code / formula / table / image 感知
[x] Block-aware 切块 ParentNode / IndexReadyChunk / overlap / token 上限
[x] Phase 0 入库     IndexReadyChunk -> Dense / Sparse / Qdrant
[x] 稳定映射         dense / sparse 共用 point id，payload 保留 chunk_id
[x] 在线检索链路     旧 LlamaIndex chunks -> Dense / Sparse / Qdrant / RRF / Rerank
[x] 生成与引用       retrieval context -> Qwen answer + [C1] citations
[x] 实验归档         原始输出 + findings + 复现命令
```

### Phase 0 未闭环

```text
[ ] 语义边界         window_tokens 还没有接入 embedding 边界微调
[ ] Parent expansion 检索 leaf 后返回 parent context
[ ] 字段化 BM25F     heading 权重和正文权重还没有分离
[ ] 固定评估集       qrels / recall@k / MRR / nDCG 尚未建立
[ ] 扩展 Parser      动态网页 / Office / 图片 OCR / OCR confidence
```

### 推荐下一步

```text
1. 用新 ChunkBuilder 重跑现有全部语料并建立 Phase 0 索引。
2. 通过环境变量让在线服务查询 phase0_dense / phase0_sparse。
3. 重建 qrels，分阶段评估 dense、sparse、RRF 和 rerank。
4. 实现 Parent expansion，最后再接入 embedding 语义边界微调。
```

---

## 七、约定

```text
1. 三类东西不混
   docs/ 知识   issues/ 缺陷   pipeline/ 实现

2. 结论必须带【前提条件】
   同一份数据在不同参数下结论可能相反, 不写前提等于没结论

3. 区分【代理指标】和【真实指标】
   代理: 切点距离 / 块大小分布 / 嵌入成本    (便宜, 随时可算)
   真实: recall@k / MRR / Precision@k       (需要评估集)

4. 区分【实验档案】和【当前认知】
   findings/ 写完不改(历史), docs/ 持续迭代(现状)

5. 每个数字都能追到文件
   找不到来源的数字就不该写进文档
```
