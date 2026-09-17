# Top-K、候选深度与 Recall@k

## 1. 先给一句话定义

在检索系统中，`K` 是一次排序或截断时保留的前 K 个结果。

```text
Top-K = 从排名结果中取前 K 条
```

例如：

```text
Dense 检索得到 1000 个候选
按相似度从高到低排序
只取前 20 条
```

这里的 20 就是这次检索的 `top_k`。

但真实 RAG 系统里不止一个 Top-K：

```text
每路召回的 Top-K
RRF 融合后的 Top-K
送入 Reranker 的 Top-K
返回给用户的 Top-K
送入 LLM 的上下文 Top-K
```

它们和 `candidate_k`、`rrf_k` 不是同一个概念。

## 2. 完整检索链路

一个常见的混合检索链路：

```text
query
  |
  +----------------------+----------------------+
  |                                             |
  v                                             v
Dense 召回 top_k_dense                    Sparse 召回 top_k_sparse
  |                                             |
  +----------------------+----------------------+
                         |
                         v
                   RRF 排名融合
                         |
                         v
                  fused top_k_rrf
                         |
                         v
                    Reranker
                         |
                         v
                  final top_k
                         |
                         v
                  context_k -> LLM
```

每一步都可以有自己的 K：

```text
top_k_dense
    Dense 通道保留多少候选

top_k_sparse
    Sparse 通道保留多少候选

candidate_k
    通常表示每路召回的候选深度

top_k_rrf
    RRF 融合后保留多少候选

rerank_candidate_k
    送多少条进入 Reranker

final_top_k
    最终返回给用户多少条

context_k
    最终把多少条塞进 LLM 上下文
```

## 3. 为什么需要 Top-K

检索系统通常不能把所有文档都送进后续步骤。

原因包括：

```text
候选数量太大
Reranker 成本随候选数增长
LLM 上下文窗口有限
响应延迟有限
低排名结果噪声高
```

Top-K 的作用是：

```text
把高成本步骤限制在少量候选上
```

例如：

```text
100 万文档
-> Dense 召回 top-50
-> Sparse 召回 top-50
-> RRF 融合
-> Reranker 只处理 top-20
-> 最终返回 top-5
-> LLM 只读取 top-5
```

## 4. Top-K 不是单一参数

### 4.1 每路召回的 Top-K

Dense 和 Sparse 是两个独立的召回器。

```text
Dense top_k_dense
    按向量相似度排序后取前 K

Sparse top_k_sparse
    按 BM25 分数排序后取前 K
```

两者应该分别控制候选深度。

原因：

```text
Dense score 和 BM25 score 不在同一个尺度
不能直接比较
也不能简单相加
```

Dense 示例：

```text
doc_A  0.82
doc_B  0.79
doc_C  0.74
```

Sparse 示例：

```text
doc_C  12.7
doc_A  10.1
doc_E   8.4
```

这里的 `0.82` 和 `12.7` 没有可比性。

### 4.2 candidate_k

`candidate_k` 通常表示：

```text
每一路进入融合前保留的候选深度
```

例如：

```text
candidate_k = 20
```

含义可以是：

```text
Dense 取 top-20
Sparse 取 top-20
RRF 最多看到两路的 40 个排名项
```

`candidate_k` 越大：

```text
召回率可能提高
后续融合的候选更多
成本和延迟增加
```

`candidate_k` 越小：

```text
速度更快
但可能提前丢掉正确文档
```

核心原则：

```text
candidate_k 通常应该大于或等于 final_top_k
```

### 4.3 RRF 的 k

RRF 中也有一个 `k`，但它不是候选数量。

RRF 公式：

```text
RRF_score(d) =
    sum_over_i(
        1 / (rrf_k + rank_i(d))
    )
```

其中：

```text
rank_i(d)
    文档 d 在第 i 路召回中的名次

rrf_k
    控制排名差值的平滑程度
```

典型值：

```text
rrf_k = 60
```

`rrf_k` 越大：

```text
不同名次之间的差距被压平
排名更平滑
```

`rrf_k` 越小：

```text
靠前名次的优势更明显
```

不要把：

```text
rrf_k
```

误认为是：

```text
top_k
candidate_k
```

它们是不同参数。

### 4.4 rerank_candidate_k

Reranker 通常比向量检索更贵。

所以不会对所有文档做 rerank，而是只处理少量候选：

```text
rerank_candidate_k = 20
```

含义：

```text
把 RRF 后的前 20 条送入 Reranker
```

Reranker 输出新的排序：

```text
candidate 20 条
-> cross-encoder / reranker 逐条打分
-> 重新排序
-> final top-5
```

如果正确文档没有进入 `rerank_candidate_k`：

```text
Reranker 无法把它救回来
```

所以：

```text
召回不足不能靠 Reranker 修复
```

### 4.5 final_top_k

最终返回给用户或业务系统的条数：

```text
final_top_k = 5
```

这是 UI 或 API 看到的最终结果数。

它通常是：

```text
candidate_k
>= rerank_candidate_k
>= final_top_k
```

### 4.6 context_k

`context_k` 是送进 LLM 的文档数量。

它不一定等于 `final_top_k`。

原因：

```text
返回给用户 10 条
但 LLM 上下文窗口只够 5 条
```

或者：

```text
最终展示 5 条
但生成模型需要更多上下文
```

所以：

```text
final_top_k      展示层
context_k        生成层
```

二者可以相等，也可以不同。

## 5. Dense Top-K 怎么计算

Dense 检索步骤：

```text
query
-> query embedding
-> 与文档向量计算相似度
-> 按相似度降序排序
-> 取 top_k_dense
```

常见相似度：

```text
cosine similarity
dot product
inner product
```

对于 cosine：

```text
相似度越高越相似
```

典型流程：

```text
1000 个文档
-> 计算或 ANN 搜索
-> 得到相似度排名
-> 取 top-20
```

Dense 擅长：

```text
同义表达
语义改写
自然语言问句
跨语言语义
```

## 6. Sparse Top-K 怎么计算

Sparse 检索步骤：

```text
query
-> 分词 / 分析器
-> 查询词项
-> 倒排索引查找
-> BM25 打分
-> 按分数降序排序
-> 取 top_k_sparse
```

Sparse 擅长：

```text
产品名
配置项
错误码
版本号
精确字符串
```

例如：

```text
TS-999
keepalived.conf
virtual_router_id
HTTP/2
```

Dense 可能找到“错误代码”，但不一定能精确命中 `TS-999`。

Sparse 会直接匹配这个字符串。

## 7. 为什么 Dense 和 Sparse 要各自 Top-K

不能直接做：

```text
dense_score + sparse_score
```

因为两个分数不是同一量纲。

Dense：

```text
范围通常是 [-1, 1] 或 [0, 1]
```

BM25：

```text
范围没有固定上限
```

例如：

```text
dense_score = 0.82
bm25_score  = 12.7
```

直接相加会让 BM25 主导结果。

所以要：

```text
分别取 Top-K
分别保留排名
再用 RRF 融合
```

RRF 使用的是：

```text
rank
```

而不是原始 score。

## 8. RRF 融合示例

假设：

```text
Dense top-4:
    1 A
    2 B
    3 C
    4 D

Sparse top-4:
    1 C
    2 A
    3 E
    4 F
```

`rrf_k = 60`。

计算：

```text
A:
    dense rank 1 -> 1/61
    sparse rank 2 -> 1/62

B:
    dense rank 2 -> 1/62

C:
    dense rank 3 -> 1/63
    sparse rank 1 -> 1/61
```

可以看到：

```text
RRF 看的是名次关系
不是 dense score 和 BM25 score 的直接相加
```

## 9. Recall@k 是什么

Recall@k 表示：

```text
在所有 relevant documents 中，
Top-k 结果覆盖了多少
```

公式：

```text
Recall@k =
    在 Top-k 中检索到的相关文档数
    / 数据集中全部相关文档数
```

例子：

```text
数据集里共有 4 个相关文档
Top-5 找到 3 个
```

则：

```text
Recall@5 = 3 / 4 = 0.75
```

如果 Top-10 找到 4 个：

```text
Recall@10 = 4 / 4 = 1.0
```

## 10. Recall@k 要分阶段看

不能只算最终结果的 Recall@k。

至少应该看：

```text
Dense Recall@k
Sparse Recall@k
Hybrid Recall@k
Rerank 前 Recall@k
Rerank 后 Recall@k
```

原因：

```text
Dense 可能有语义召回优势
Sparse 可能有精确匹配优势
RRF 可能把两路互补结果合起来
Reranker 只能在已有候选内重排
```

如果：

```text
Recall@20 = 0.95
Recall@5  = 0.70
```

说明：

```text
正确文档在更大候选集里
但最终截断太早
```

可能需要调大：

```text
candidate_k
rerank_candidate_k
```

如果：

```text
Dense Recall@20    = 0.55
Sparse Recall@20   = 0.50
Dense + Sparse RRF = 0.80
```

说明两路有互补性。

## 11. Candidate Depth 和 Recall 的关系

Candidate depth 是：

```text
在进入 RRF 和 Reranker 前保留多少候选
```

通常：

```text
candidate_k 越大
Recall@candidate_k 越高
```

但代价也增加：

```text
向量查询成本
BM25 候选更多
RRF 融合更多
Rerank 输入更多
延迟增加
```

所以目标是：

```text
在满足 Recall 目标的前提下
让 candidate_k 尽可能小
```

## 12. 常见参数关系

一个典型链路：

```text
dense candidate_k      20
sparse candidate_k     20
rrf_k                  60
rerank_candidate_k     20
final_top_k             5
context_k               5
```

逻辑关系通常是：

```text
candidate_k
    >= rerank_candidate_k

rerank_candidate_k
    >= final_top_k

context_k
    <= final_top_k
```

但这不是硬性数学关系，要根据：

```text
语料规模
chunk 粒度
上下文窗口
Reranker 成本
延迟要求
Recall 目标
```

调整。

## 13. Top-K 和相似度阈值不是一回事

Top-K：

```text
固定取前 K 条
```

阈值：

```text
只保留 score 超过阈值的文档
```

两者可以组合：

```text
先按 score 过滤 threshold
再取 top-k
```

如果只用 Top-K：

```text
即使所有文档都低相关
也会返回 K 条
```

如果只用阈值：

```text
可能返回 0 条
也可能返回过多
```

## 14. Top-K 和其他参数的区别

```text
chunk_size
    文本切块大小

top_k
    排名结果取前多少条

candidate_k
    每路召回保留多少候选

rrf_k
    RRF 的排名平滑常数

rerank_candidate_k
    送多少条给 Reranker

context_k
    送多少条给 LLM

token_budget
    LLM 上下文最大 token 数
```

这些参数互相影响，但不是同一个概念。

## 15. 当前项目如何实现

当前项目默认链路：

```text
Dense top-20
Sparse top-20
RRF 融合
Rerank top-20
最终返回 top-5
生成上下文 context_k=5
```

默认参数：

```text
candidate_k = 20
rrf_k       = 60
rerank_candidate_k = 20
top_k       = 5
context_k   = 5
```

实现位置：

```text
pipeline/hybrid_retriever.py
pipeline/rag_server.py
pipeline/static/index.html
```

核心逻辑：

```text
Dense search
    -> limit=candidate_k

Sparse search
    -> limit=candidate_k

RRF
    -> k=rrf_k
    -> limit=candidate_limit

Rerank
    -> 输入 rerank_candidate_k
    -> 输出 final top_k

Generation
    -> 使用 context_k 条上下文
```

当前实现的注意点：

```text
dense 和 sparse 各自返回候选
RRF 使用排名而非 score 直接相加
rerank 只能重排已经召回的候选
如果没有进入 rerank_candidate_k，reranker 无法补救
context_k 影响生成成本
candidate_k 影响召回和延迟
rrf_k 影响排名权重分布
```

## 16. 如何调参

推荐顺序：

```text
1. 先固定 chunk 策略
2. 分别评估 Dense Recall@k
3. 分别评估 Sparse Recall@k
4. 调整 candidate_k
5. 观察 RRF 后的 Recall@k
6. 调整 rerank_candidate_k
7. 评估 Rerank 后 nDCG / MRR
8. 最后调整 final_top_k 和 context_k
```

不要一开始就同时调：

```text
chunk size
embedding model
candidate_k
rrf_k
reranker
context_k
```

这样无法判断收益来自哪里。

## 17. 一页速记

```text
top_k
    取前 K 条

candidate_k
    每路召回的候选深度

rrf_k
    RRF 排名平滑常数，不是候选数量

rerank_candidate_k
    送入 Reranker 的候选数

final_top_k
    最终返回给用户的结果数

context_k
    送入 LLM 的上下文结果数

Recall@k
    Top-k 中命中的相关文档 / 全部相关文档
```

一句话：

```text
candidate_k 决定“能不能找回”，
rerank_candidate_k 决定“能给 reranker 多少机会”，
final_top_k 决定“展示多少”，
context_k 决定“LLM 能读多少”。
```

最重要的边界：

```text
RRF 负责融合排名，
Reranker 负责精排，
Top-K 负责截断，
Recall@k 负责衡量截断前是否找回来了。
```
