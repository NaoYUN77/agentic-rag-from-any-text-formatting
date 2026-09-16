# RAG 复习路线: Chunk -> Dense/Sparse -> RRF -> Generation

> 版本: 2026-09-15
>
> 用途: 明天按顺序复习今天已经生成的理论文档、实验数据、实现代码和关键结论。
> 不要从 README 或代码目录随机翻; 按下面的依赖顺序走。

---

## 今日状态快照

### 今天解决了什么

```text
0. 今天主线是 chunk 策略探索: 做过多轮效果与成本对比
1. 把 dense / sparse 的共用概念独立整理
2. 写通独立 sparse 链路: PDF/Markdown -> 清洗 -> jieba -> BM25 -> top-k
3. 从现有 dense collection 复用同一批 chunk, 写入 redhat_sparse
4. 实现 dense + sparse 双路召回
5. 实现 RRF(k=60) 融合
6. 把检索结果接入 FastAPI 和网页
7. 写通基础 generation: hybrid top-k -> context -> Qwen -> answer + citations
8. 增加 qwen-turbo / qwen-plus 生成模型切换
9. 增量加入 200 页 NGINX PDF, 总 corpus 从 22 扩展到 297 chunks
10. 接入 gte-rerank-v2, 完成 RRF / rerank smoke 对比
11. 实现 LlamaIndex 自定义 NodeParser: 800 token + 400 overlap + 语义边界微调
```

### Chunking 主线结论

今天的主线不是直接堆 retrieval 组件, 而是先比较 chunk 策略的效果和成本。

核心结论:

```text
纯固定长度
    成本最低、实现最简单
    切点在语义上接近随机

纯语义分块
    边界更合理
    但嵌入成本更高, 块大小容易失控

固定长度 + 重叠 + 语义微调
    成本可控
    保留固定长度的工程稳定性
    在边界附近用语义距离修正切点
    是当前最值得保留的工程折中
```

OpenAI 思路的价值主要在这里:

```text
1. 固定 chunk_size, 让成本、块数、上下文长度可预测
2. 使用 chunk_overlap, 降低边界句被切断的风险
3. 优先保留标题和段落结构
4. 在需要时再加语义微调, 而不是一开始就上纯语义分块
```

这不是说固定长度一定优于其它策略, 而是说它更适合作为**稳定基线**:

```text
先用固定长度 + overlap 建立可复现基线
再用语义微调改善边界
最后用评估集验证是否真的带来收益
```

### 当前系统真实状态

```text
redhat
    354 个 dense points
    Red Hat 42 + NGINX 312
    qwen3-vl-embedding, 1024 维

redhat_sparse
    354 个 sparse points
    jieba + BM25(k1=1.2, b=0.75)

HybridRetriever
    dense top-20 + sparse top-20 -> RRF -> top-5

Generation
    默认 qwen-turbo
    可选 qwen-plus
    answer + [C1] citations

Web
    http://127.0.0.1:8000/
    支持 hybrid / dense / sparse
    支持 answer / search
```

### 基础生成已跑通, 生产级 generation 未完成

```text
已经跑通的基础版
    /api/answer
    context builder
    hybrid top-k 作为证据
    qwen-turbo / qwen-plus
    answer + citations
    页面展示答案和来源

还未达到生产级
    没有流式输出
    没有逐句引用一致性校验
    没有 answer / citation 评估集
    rerank 第一版已接入 gte-rerank-v2
    context_k 和 context 长度还没有调优
    没有超时重试、降级、缓存和并发控制
```

### 今日验证数据

| 项目 | 实测数据 |
|---|---|
| 总 chunk 数 | 354 |
| Red Hat / NGINX chunks | 42 / 312 |
| dense points | 354 |
| sparse points | 354 |
| sparse 词表 | 838 |
| 平均 token/chunk | 169.73 |
| 非零 sparse 权重 | 2120 |
| dense top-5 | 4, 0, 5, 12, 15 |
| sparse top-5 | 0, 1, 4, 3, 8 |
| hybrid top-5 | 0, 4, 3, 1, 15 |
| qwen-plus 生成 | 约 9.9s ~ 15.9s |
| qwen-turbo 生成 | 约 2.9s ~ 3.1s |
| RRF Recall@5 / Recall@10 / MRR@20 | 0.808 / 0.902 / 0.875 |
| Rerank Recall@5 / Recall@10 / MRR@20 | 0.777 / 0.933 / 0.938 |

### 仍未解决的问题

```text
1. 没有检索评估集: Recall@k / MRR / nDCG
2. 没有答案评估集: faithfulness / answer relevance / citation accuracy
3. dense 和 sparse 仍是两个 collection, 尚未合并 named vectors
4. RRF 的两路权重和 candidate_k 还没调优
5. rerank 只有 8 条 smoke 验证, 还需要更大评估集
6. 生成没有流式输出
7. 引用只做到上下文编号, 还没有逐句溯源校验
8. NGINX PDF 的代码块被误判为章节标题, section metadata 噪声严重
9. LlamaIndex 新切分器只完成 Red Hat 预览, 现有在线索引仍是旧切分
10. 现有 qrels 的 point_id 已被 LlamaIndex 重建失效, 需要重新标注
```

### 明天推荐起点

```text
优先级 P0  建立评估集
优先级 P1  对 dense / sparse / hybrid 做同集对比
优先级 P1  调优 rerank candidate_k 和 reranker 模型
优先级 P1  用 LlamaIndex 新切分器重建 Red Hat + NGINX 并重新评估
优先级 P2  做流式 generation
优先级 P2  做引用一致性校验
```

最推荐的第一个动作:

```text
先准备一组 query, 标出正确 chunk 和标准答案。
没有这组数据, 后续 RRF 权重、rerank 和 generation 都只能凭感觉。
```

---

## 一、先明确复习目标

复习结束后, 你应该能独立回答:

```text
1. 稀疏召回和稠密召回分别解决什么问题?
2. 稀疏词表从哪里来? 为什么不是“BM25 筛出的高频词”?
3. BM25 如何变成 Qdrant sparse vector? 查询 top-k 怎么算?
4. dense top-k 和 sparse top-k 为什么不能用同一套分数直接相加?
5. RRF 的 rank_i(d)、k、candidate_k 分别是什么?
6. dense 和 sparse 如何共享同一批 chunk、point_id 和 payload?
7. chunk 策略如何改变 BM25 的 df、IDF、文档长度和最终排序?
8. 检索完成后, generation 阶段如何把 chunk 变成答案和引用?
9. qwen-plus 和 qwen-turbo 的延迟差异来自哪里?
```

---

## 二、复习顺序总览

```text
第 0 步  Chunk 策略与 OpenAI 基线
第 1 步  共用概念: dense vs sparse
第 2 步  打分与融合: BM25 + RRF
第 3 步  独立 sparse 完整链路
第 4 步  复用 dense chunk 并写入 Qdrant sparse
第 5 步  Dense + Sparse + RRF
第 6 步  Generation + Citations
第 7 步  模型延迟与当前边界
第 8 步  代码和实验对账
第 9 步  自测题
```

实际复习时, 先完成下面的第 0 步, 再进入 dense/sparse。

建议时间:

```text
快速复习          60 分钟
完整复习         120 分钟
带代码实验       180 分钟
```

---

## 第 0 步: Chunk 策略与 OpenAI 基线 (约 25 分钟)

阅读:

```text
../docs/chunking_strategy_guide.md
../docs/rag_chunking_concepts.md
../docs/semantic_chunking_notes.md
```

重点对照:

```text
纯固定长度
纯语义分块
长度优先 + 语义微调
语义优先 + 长度约束
句子窗口
```

重点实验:

```text
../pipeline/experiments/04_chunking_strategies.txt
../pipeline/experiments/05_window_sweep.txt
../pipeline/experiments/08_embedding_mechanism_compare.txt
../pipeline/experiments/09_embedding_config_compare.txt
```

对应结论:

```text
../pipeline/experiments/findings/01_breakpoint_methods.md
../pipeline/experiments/findings/02_window_width.md
../pipeline/experiments/findings/03_embedding_mechanism.md
```

复习时重点回答:

```text
固定长度为什么适合做稳定基线?
overlap 解决什么问题? 代价是什么?
语义边界为什么不能只看固定长度?
长度约束和语义切分如何组合?
为什么最终还要用检索评估集验证?
```

---

## 第 1 步: 共用概念 (约 20 分钟)

阅读:

```text
../docs/sparse_vs_dense_retrieval.md
../docs/sparse_retrieval_concepts.md
```

重点章节:

```text
一、一句话区分
二、概念从哪里来
三、稀疏召回: 核心是词项匹配
四、稠密召回: 核心是语义向量
五、两者的核心差异
六、同一查询在两条通道里会怎么走
```

复习时必须分清:

```text
稀疏向量
    维度 = 词表大小
    非零值很少
    一个维度通常对应一个词项

稠密向量
    维度 = embedding 输出维度
    大部分维度都有值
    一个维度不对应具体词
```

关键记忆:

```text
Dense  = 语义相关
Sparse = 字面相关
二者互补, 不是替代
```

---

## 第 2 步: BM25 与 RRF (约 20 分钟)

阅读:

```text
../docs/retrieval_scoring_and_rrf.md
```

优先复习:

```text
三、TF-IDF
四、BM25
五、共用概念: 稀疏召回与稠密召回
六、混合检索
七、RRF: Reciprocal Rank Fusion
```

必须手写或复述:

```text
RRF(d) = Σ_i 1 / (k + rank_i(d))
```

分清:

```text
rank_i(d)
    文档 d 在第 i 条通道中的名次

candidate_k
    每路先取多少候选

rrf_k
    RRF 平滑常数, 当前为 60

final top_k
    融合后最终返回多少条
```

自测:

```text
如果 d 在 dense 排名 2、sparse 排名 1, RRF 如何算?
如果 d 只出现在 sparse 排名 5, dense 没召回, 还有贡献吗?
为什么 RRF 不直接相加 cosine 和 BM25 分数?
```

---

## 第 3 步: 独立 sparse 完整链路 (约 20 分钟)

阅读实验结论:

```text
../pipeline/experiments/findings/04_jieba_sparse_pipeline.md
```

对照原始输出:

```text
../pipeline/experiments/14_sparse_pipeline.txt
```

查看结构化产物:

```text
../pipeline/experiments/14_sparse_pipeline_artifacts/
```

复习链路:

```text
Markdown/PDF
    -> 清洗
    -> 段落级实验分块
    -> jieba
    -> 技术 token 保护
    -> term id
    -> 倒排索引
    -> BM25
    -> top-k
```

重点结论:

```text
词表 = 分析器输出的所有词项
词表不是 BM25 挑出来的高频词
BM25 不调用 embedding 模型
查询先查 posting list, 再从候选 chunk 计算分数
```

---

## 第 4 步: 复用 dense chunk, 写入 sparse collection (约 20 分钟)

阅读实验结论:

```text
../pipeline/experiments/findings/05_sparse_qdrant_storage.md
```

对照输出:

```text
../pipeline/experiments/15_sparse_qdrant_pipeline.txt
../pipeline/experiments/15_sparse_qdrant_verify.txt
```

复习当前存储结构:

```text
redhat
    dense vector
    payload: text/file_name/page/section/chunk_index

redhat_sparse
    sparse vector
    相同 point_id
    相同 payload
```

重点结论:

```text
chunk 可以共用
不需要为了 sparse 重新解析 PDF 或重新分块
Qdrant sparse vector 存储 indices + values
term -> id 和 IDF 仍然由应用侧维护
```

---

## 第 5 步: Dense + Sparse + RRF (约 25 分钟)

阅读实验结论:

```text
../pipeline/experiments/findings/06_hybrid_rrf.md
```

对照输出:

```text
../pipeline/experiments/16_hybrid_rrf.txt
```

复习实现:

```text
../pipeline/hybrid_retriever.py
```

重点看:

```text
dense_search()
sparse_query_vector()
sparse_search()
rrf_fuse()
search()
```

实测结果:

```text
dense top-5    4, 0, 5, 12, 15
sparse top-5   0, 1, 4, 3, 8
hybrid top-5   0, 4, 3, 1, 15
```

重点理解:

```text
point_id=0: dense #2 + sparse #1 -> RRF #1
point_id=4: dense #1 + sparse #3 -> RRF #2
point_id=1: dense #9 + sparse #2 -> RRF #4
```

这说明 RRF 奖励两路都靠前的候选, 同时保留单路强项。

---

## 第 6 步: Generation + Citations (约 25 分钟)

阅读实验结论:

```text
../pipeline/experiments/findings/07_generation.md
```

对照输出:

```text
../pipeline/experiments/17_generation.txt
```

复习实现:

```text
../pipeline/generation.py
../pipeline/rag_server.py
../pipeline/static/index.html
```

复习链路:

```text
hybrid top-k
    -> context builder
    -> [C1] [C2] ... 编号
    -> Qwen chat
    -> answer
    -> citations
```

当前提示词约束:

```text
只能基于上下文回答
上下文不足时明确说无法确定
关键结论必须使用 [C1] 等引用
```

重点区分:

```text
chunk 是证据
answer 是生成结果
citation 是答案和证据之间的映射
```

---

## 第 7 步: 模型延迟与当前边界 (约 10 分钟)

阅读实验结论:

```text
../pipeline/experiments/findings/08_generation_model_latency.md
```

对照输出:

```text
../pipeline/experiments/18_generation_turbo.txt
```

实测:

```text
qwen-plus    generation_ms ~ 9868 ~ 15884
qwen-turbo   generation_ms ~ 2908 ~ 3093
```

当前默认:

```text
qwen-turbo   页面默认快速模式
qwen-plus    页面可选质量模式
```

还没解决:

```text
流式输出
rerank
答案质量评估
引用准确率评估
context_k 的调优
```

---

## 第 8 步: 代码和实验对账

按这个顺序读代码:

```text
1. ../pipeline/sparse_pipeline_demo.py
2. ../pipeline/hybrid_retriever.py
3. ../pipeline/generation.py
4. ../pipeline/rag_server.py
5. ../pipeline/static/index.html
```

对应实验:

```text
14  sparse 独立链路
15  复用 chunk 并写入 redhat_sparse
16  dense + sparse + RRF
17  生成答案和引用
18  qwen-turbo 快速生成
```

对应 findings:

```text
04  jieba sparse 独立链路
05  Qdrant sparse 存储
06  Hybrid + RRF
07  Generation + Citations
08  Qwen 模型延迟
```

---

## 第 9 步: 明天自测题

```text
[ ] 为什么词表不是高频词列表?
[ ] BM25 的 df、IDF、文档长度如何影响分数?
[ ] dense 和 sparse 的 top-k 为什么必须分开算?
[ ] RRF 为什么只看 rank, 不看原始分数?
[ ] 为什么 dense 和 sparse 能共用 chunk 和 point_id?
[ ] payload 和 vector 的区别是什么?
[ ] chunk 切分如何反过来影响 BM25?
[ ] 为什么 generation 不能直接使用 chunk 原文作为答案?
[ ] 生成答案中的 [C1] 如何映射回来源?
[ ] qwen-turbo 更快, 为什么还要保留 qwen-plus?
```

如果以上都能解释清楚, 今天的学习链路基本闭环。

---

## 明天的起点

建议从这两个方向继续:

```text
方向 A: 生产级 generation
    流式输出
    更好的 context 组装
    引用一致性校验

方向 B: 检索质量验证
    建立评估集
    Recall@k / MRR / nDCG
    dense / sparse / hybrid 三种模式对比
```

最推荐的顺序:

```text
先做检索评估集 -> 再调 RRF/context_k -> 再做 rerank -> 最后继续优化 generation
```

否则生成模型的回答“看起来不错”也无法证明是检索和融合真的变好了。

---

## 一句话复习结论

```text
Chunk 负责提供证据,
Dense 负责语义召回,
Sparse/BM25 负责字面召回,
RRF 负责合并两路排名,
Generation 负责把证据变成答案和引用。
```
