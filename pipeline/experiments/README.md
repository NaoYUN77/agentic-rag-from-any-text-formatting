# 实验数据索引

本目录存放**真实运行产生的输出**，用于支撑 `rag_chunking_concepts.md` 等文档里的结论。

> **`findings/` 子目录**存放每次实验的**结论**（含前提条件与适用边界）。
> 本 README 是**数据索引**（哪个文件是什么、怎么复现），findings 是**结论档案**。
> 两者配套：数据可复现，结论可追溯。

**所有数据都来自实际调用**，不是估算：

```text
嵌入模型    阿里云百炼 qwen3-vl-embedding (1024 维)
分块实现    LangChain SemanticChunker / LlamaIndex NodeParser / 自写脚本
语料        见 corpus/ 目录
```

---

## 文件清单

| 文件 | 框架 | **嵌入对象** | metadata | 关键数据 |
|---|---|---|---|---|
| `01_semantic_chunking_qwen.txt` | **LangChain** SemanticChunker | **窗口**(前1+本+后1) | 无 | 距离序列、四种断点方法、buffer_size 影响 |
| `02_semantic_chunking_local.txt` | LangChain SemanticChunker | 窗口 | 无 | 证明哈希函数无语义能力 |
| `03_pdf_pipeline.txt` | LangChain SemanticChunker | 窗口 | 无 | 274 句语料下的分块结果 |
| `04_chunking_strategies.txt` | **自写脚本** | **单句** | 无 | 切点质量 A/C/D 对比 |
| `05_window_sweep.txt` | **自写脚本** | **单句** | 无 | 窗口 0~600 的块数与切点质量 |
| `06_sentence_window_retrieval.txt` | 自写脚本 | 单句(索引)+窗口(对照) | 手写 dict | 4 个查询全部单句胜出 |
| `07_llamaindex_window.txt` | **LlamaIndex** SentenceWindowNodeParser | 单句 | **真 Node** | node 结构 + 检索 + MetadataReplacement |
| `14_sparse_pipeline.txt` | **独立 sparse 脚本** | **jieba 词项** | chunk/section | BM25 词表、倒排索引、sparse vector、top-k |
| `15_sparse_qdrant_pipeline.txt` | **独立 sparse 脚本** | **复用 Qdrant chunk** | 原 payload | 22 个 chunk -> `redhat_sparse` |
| `15_sparse_qdrant_verify.txt` | **Qdrant sparse 查询** | sparse vector | 原 payload | 直接从 `redhat_sparse` 检索的 top-k |
| `16_hybrid_rrf.txt` | **HybridRetriever** | dense + sparse | 原 payload | dense / sparse / RRF 三种排名对照 |
| `17_generation.txt` | **QwenChatGenerator** | hybrid top-5 | 引用来源 | answer + citations + 生成耗时 |
| `18_generation_turbo.txt` | **QwenChatGenerator** | hybrid top-5 | 引用来源 | qwen-turbo 快速模式 |
| `19_rerank_compare.txt` | **DashScopeReranker** | RRF top-20 | 8 条 smoke qrels | RRF vs RRF + rerank |
| `20_nginx_ingest_verify.txt` | **增量入库** | NGINX PDF + Red Hat | 总 297 points | NGINX 与 Red Hat 查询验证 |
| `21_nginx_ingest_stats.txt` | **增量入库** | 275 NGINX + 22 Red Hat | 语料统计 | chunk 数与长度分布 |
| `22_mixed_qrels_compare.txt` | **Rerank smoke** | 旧 297-point 混合索引 | 13 条 qrels | RRF vs rerank(旧 point_id) |
| `23_llamaindex_fixed_semantic.txt` | **LlamaIndex NodeParser** | 800 token + 400 overlap | TextNode metadata | Red Hat 预览与 token 分布 |
| `24_llamaindex_rebuild.txt` | **LlamaIndex 重建** | 800/400 | 失败样本 | 超长句导致 7875 token |
| `25_llamaindex_rebuild.txt` | **LlamaIndex 重建** | 800/400 | 354 points | Red Hat 42 + NGINX 312 |

### ⚠️ 两种嵌入机制的距离尺度不可比

```text
窗口嵌入(01/02/03)    距离范围 0.0342 ~ 0.1700     阈值 0.1690
单句嵌入(04/05/06)    距离范围 0.3600 ~ 0.5200     全局均值 0.3612
```

**同一个语料、同一个模型, 换一种嵌入对象, 距离尺度就换了一套 —— 差了约一倍。**

原因是窗口嵌入让相邻向量更相似(相邻窗口共享 2/3 的内容), 距离被整体压小。

**所以 05 里的"相对全局 1.00x"是以【单句嵌入】为基准的, 不能和 01 的 0.1690 直接比较。**

要用哪种机制做切点判定, 就一直用同一种 —— **不要混用两个实验的阈值。**

---

## 复现命令

> **在 `pipeline\` 目录下执行。** 脚本用的是相对路径(`corpus/`、`experiments/`),
> 换目录会找不到文件。

```powershell
cd E:\Code\python_dev_agent_demo\pipeline
$env:DASHSCOPE_API_KEY = "sk-..."

# 01 语义分块(真实模型)
python semantic_chunker_demo.py | Out-File experiments/repro_01.txt

# 02 语义分块(本地哈希, 无需网络)
python semantic_chunker_demo.py --backend local | Out-File experiments/repro_02.txt

# 03 PDF 全链路
python semantic_chunker_demo.py --pdf 文件.pdf --pdf-pages 1-20 | Out-File experiments/repro_03.txt

# 04 四种分块策略对比
python hybrid_chunk_demo.py --max-chars 800 --window 4 | Out-File experiments/repro_04.txt

# 05 边界窗口 sweep
python window_sweep_demo.py | Out-File experiments/repro_05.txt

# 06 句子窗口检索对照
python sentence_window_demo.py | Out-File experiments/repro_06.txt

# 07 LlamaIndex 窗口机制
python llamaindex_window_demo.py | Out-File experiments/repro_07.txt

# 14 独立 sparse 完整链路
$env:SPARSE_QUERY = "你自己的查询"
python sparse_pipeline_demo.py `
    --md corpus/redhat_p1-20.md `
    --query $env:SPARSE_QUERY `
    --out-dir experiments/14_sparse_pipeline_artifacts `
    | Out-File experiments/14_sparse_pipeline.txt

# 15 复用 dense chunk 并写入 Qdrant sparse collection
python sparse_pipeline_demo.py `
    --qdrant `
    --qdrant-path qdrant_data `
    --collection redhat `
    --query $env:SPARSE_QUERY `
    --write-sparse `
    --rebuild-sparse `
    --sparse-collection redhat_sparse `
    --out-dir experiments/15_sparse_qdrant_artifacts `
    | Out-File experiments/15_sparse_qdrant_pipeline.txt
```

---

## 关键数据速览

### 语义速检（01, 02 对比）

同一组对照句，两种嵌入函数给出的相似度：

```text
对照                              真实模型   本地哈希
──────────────────────────────────────────────────
汽车 / automobile     跨语言同义    0.782     0.000
如何部署系统 / 安装步骤如下  同义改写  0.642     0.000
退货流程 / 退换货怎么走   同义改写    0.853     0.000
如何部署系统 / 今天天气不错 不相关   0.359     0.000
```

**哈希函数不是"不太精确"，是完全没有语义能力。**

### 四种断点判定方法（01，10 句语料）

```text
percentile            阈值 0.1690   ->  2 块   切在正确位置
standard_deviation    阈值 0.2320   ->  1 块   一刀没切
interquartile         阈值 0.1678   ->  2 块   切在正确位置
gradient              阈值 0.0969   ->  2 块   切在假断点上
```

### 块大小失控（03, 274 句语料）

```text
amount=50  -> 137 个 chunk   字数 min=6     max=2901
amount=80  ->  56 个 chunk   字数 min=19    max=2949
amount=90  ->  29 个 chunk   字数 min=36    max=3304
amount=95  ->  15 个 chunk   字数 min=50    max=4426
amount=99  ->   4 个 chunk   字数 min=3264  max=6015
```

**同一套参数下，块从 6 字到 6015 字都有。**

### 分块策略对比（04, 274 句语料）

```text
全局相邻距离均值: 0.3612    ← 基准线

A. 纯固定长度        块数 21   min 438  均值 815  切点距离 0.3628  (1.00x)
C. 长度优先+语义微调   块数 22   min 357  均值 778  切点距离 0.4796  (1.33x)
D. 语义优先+长度约束   块数 28   min 131  均值 612  切点距离 0.4821  (1.33x)
```

**A 的切点距离几乎等于全局均值 —— 说明纯固定长度的切点在语义上是随机的。**

### 边界窗口 sweep（05, max_chars=800）

```text
窗口   块数   字数 min/均值/max        切点均值距离   相对全局
─────────────────────────────────────────────────────────────
   0     21     438 /   815 /  2138      0.3628        1.00x
 100     22     214 /   778 /  2138      0.4319        1.20x
 200     23     357 /   744 /  2138      0.4495        1.24x
 300     23     357 /   744 /  2138      0.4624        1.28x
 400     24     431 /   713 /  2138      0.4881        1.35x
 600     28     171 /   612 /  2138      0.5221        1.45x
```

**窗口越大切点质量越高，但块数增加、块大小开始不稳。**

### 成本对比（05 的延伸计算）

```text
方案                          嵌入输入量    相对基准
──────────────────────────────────────────────────
纯语义分块(buffer_size=1)      49214 字      2.87x
全句嵌入 + 边界微调             17123 字      1.00x
只嵌边界窗口内的句子            13016 字      0.76x
```

**主要成本差在"嵌窗口"和"嵌单句"，不在"嵌全部"和"嵌边界"。**

### 单句嵌入 vs 窗口嵌入检索（06）

```text
查询                              单句嵌入   窗口嵌入
────────────────────────────────────────────────
显存不够用怎么办                    命中 S3    偏到 S4
长文档应该怎么切分                  命中 S8    偏到 S9
怎么判断系统答得好不好              命中 S23   偏到 S24
为什么换了模型要重建索引             命中 S14   偏到 S15
```

**4:0，单句嵌入全部命中，窗口嵌入全部偏了一格。**

### 独立 jieba + BM25 稀疏链路（14）

```text
清洗后 Markdown    17365 字
chunk 数           62
词表大小           1025
平均 token/chunk   69.081
非零 sparse 权重   2852
稀疏密度           0.044878
BM25                k1=1.2, b=0.75
```

**这条链路不调用 embedding，也不依赖 Qdrant。** 它先验证了:

```text
chunk -> jieba 词项 -> term id -> 倒排索引 -> BM25 top-k
```

结构化产物在 `14_sparse_pipeline_artifacts/`:

```text
chunks.jsonl        每个 chunk 的文本、section、tokens、sparse indices/values
vocab.json          term -> id 词表
postings.json       term -> chunk id 倒排索引
summary.json        索引规模与稀疏密度
```

结论与边界见 `findings/04_jieba_sparse_pipeline.md`。

### 复用 dense chunk 并写入 Qdrant sparse collection（15）

```text
来源 collection      redhat
来源 chunk           22
目标 collection      redhat_sparse
文本规模             13208 字
词表大小             838
平均 token/chunk     169.73
非零 sparse 权重     2120
稀疏密度             0.114992
```

这条链路没有重新解析 PDF、没有重新分块，而是直接读取现有 dense collection
的 payload，把同一批 chunk 写成 Qdrant sparse vectors。验证文件确认
`redhat_sparse` 可以被 Qdrant 直接检索。

结论与边界见 `findings/05_sparse_qdrant_storage.md`。

### Dense + Sparse + RRF 融合（16）

```text
dense top-5    4, 0, 5, 12, 15
sparse top-5   0, 1, 4, 3, 8
hybrid top-5   0, 4, 3, 1, 15

每路候选数     20
RRF k          60
```

融合实现位于 `hybrid_retriever.py`，页面截图见:

```text
16_hybrid_ui.png
16_hybrid_ui_mobile.png
```

结论与边界见 `findings/06_hybrid_rrf.md`。

### 检索增强生成（17）

```text
检索             Hybrid + RRF
上下文数         5
上下文上限       6000 字
生成模型         qwen-plus
answer 长度      705 字
citations        5
generation_ms    9868
```

接口为 `POST /api/answer`，返回答案、引用来源和检索证据。
页面支持“问答生成 / 仅检索”两种任务模式。

结论与边界见 `findings/07_generation.md`。

快速模型对比:

```text
qwen-plus   generation_ms = 9868
qwen-turbo  generation_ms = 3093
```

结论与边界见 `findings/08_generation_model_latency.md`。

### RRF + Rerank smoke（19）

```text
reranker      gte-rerank-v2
查询数        8
Recall@5      0.808 -> 0.777
Recall@10     0.902 -> 0.933
MRR@20        0.875 -> 0.938
rerank 延迟   约 1.1 ~ 1.4 秒/查询
```

结论与边界见 `findings/09_rerank_effect.md`。

### 增量加入 NGINX PDF 语料（20/21）

```text
PDF 页数       200
新增 chunks    275
总 chunks      297
dense points   297
sparse points  297
NGINX 字数     169995
```

结论与边界见 `findings/10_nginx_ingest.md`。

### LlamaIndex 固定长度 + 语义微调（23）

```text
chunk_size       800 token
chunk_overlap    400 token
window_size      400 token
nodes            42
tokens min/avg/max  378 / 590.0 / 763
```

实现位于:

```text
llamaindex_fixed_semantic_splitter.py
llamaindex_fixed_semantic_demo.py
```

### LlamaIndex 全量重建（24/25）

```text
Red Hat chunks   42
NGINX chunks     312
总 chunks        354
dense points     354
sparse points    354
```

结论与边界见 `findings/11_llamaindex_rebuild.md`。

---

## 数据的使用边界

**这些数据是真实的、可复现的**（同一输入两次运行输出逐行一致），但要注意：

```text
✅ 可以信的结论
   - 真实模型 vs 哈希函数的语义差异
   - 纯固定长度的切点在语义上随机（切点距离 ≈ 全局均值，这是数学必然）
   - 语义微调能把切点质量提升 1.2~1.45 倍

❓ 证据不足的结论
   - 不同策略之间的小差异（如 0.4796 vs 0.4821）
   - 单份语料上的参数最优值

📌 尚未验证的
   - 这些指标是【代理指标】，衡量"切点是否落在语义跳变处"
   - 不代表检索效果就好。要判断好坏，需要建立【评估集】跑 recall@k / MRR
   - 目前还没有评估集，所有关于"哪种分块更好"的结论都只是间接推断
```

**语料规模**：`03/04/05` 用的是同一份文档（Red Hat 手册前 20 页，274 句 / 17123 字）。**单份文档的结论不应直接推广。**



