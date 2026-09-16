# 检索打分与 RRF 融合: 从词袋、TF-IDF、BM25 到混合检索

> 本文聚焦打分与融合:
>
> ```text
> 表示与打分: 词袋 / TF-IDF / BM25
> 融合算法: RRF(倒数排名融合)
> 精排算法: cross-encoder rerank
> ```
>
> 文中的 RRF 是标准写法, 不是 RFF。
>
> dense 和 sparse 属于共用检索概念, 已独立到:
>
> ```text
> sparse_vs_dense_retrieval.md
> ```
>
> 关联内容:
> - 当前通道问题: `../issues/04_channel_layer.md`
> - 通道对照实验脚本: `../pipeline/retrieval_channel_compare.py`
> - 检索实现: `../pipeline/rag_pipeline.py`
> - 通用 RAG 概念: `rag_chunking_concepts.md`

---

## 一、先给全链路定位

RAG 检索可以拆成四层。每一层解决的问题不同, 不能互相替代:

```text
query
  |
  +-- 稀疏通道: BM25 / TF-IDF / SPLADE -------- top-N1
  |
  +-- 稠密通道: BGE / Qwen embedding ---------- top-N2
                |
                v
          RRF 等排名融合 ---------------------- top-K
                |
                v
          cross-encoder 重排序 ---------------- top-k_final
                |
                v
               送给 LLM
```

对应关系:

```text
词袋、TF-IDF、BM25   负责"字面出现了什么"
BGE、Qwen embedding  负责"语义大概是什么"
RRF                   负责"如何合并多个候选排名"
rerank                负责"把少量候选精排一次"
```

最常见误区是把这些词都叫成"向量模型"。严格来说:

- BM25 是**检索打分算法**, 不是 embedding 模型。
- BGE 是**embedding 模型家族**, 不是融合算法。
- RRF 是**排名融合算法**, 本身不做文本表示。
- rerank 是**二次排序阶段**, 通常使用 cross-encoder, 不是召回器。

---

## 二、词袋模型(Bag of Words, BoW)

### 2.1 它解决什么问题

把一篇文本变成固定词表上的计数向量, 让计算机可以先做最基础的字面检索和分类。

### 2.2 定义

假设语料词表有 `V` 个词:

```text
V = {w1, w2, w3, ..., wV}
```

文档 `d` 的词袋向量:

```text
v_d = [c(w1, d), c(w2, d), ..., c(wV, d)]
```

其中:

```text
c(w, d) = 词 w 在文档 d 中出现的次数
```

例如:

```text
词表 = [keepalived, haproxy, 负载, 平衡]

文档 A = "keepalived 负载 平衡"
v_A   = [1, 0, 1, 1]

文档 B = "haproxy 负载 平衡"
v_B   = [0, 1, 1, 1]
```

### 2.3 作用

- 把变长文本表示为固定长度向量。
- 保留"某个词出现过多少次"。
- 计算简单, 可解释, 不需要 GPU。

### 2.4 问题

```text
1. 丢掉词序
   "狗咬人" 和 "人咬狗" 的词袋几乎一样。

2. 没有同义能力
   "汽车" 和 "automobile" 是两个不同维度。

3. 常见词和稀有词权重一样
   "的" 和 "VRRP" 都只是一个词, 但重要性完全不同。

4. 词表维度很大
   中文词表可能几十万维, 但每篇文档只有很少的非零值。
```

正因为第 3 个问题, 才发展出了 TF-IDF。

---

## 三、TF-IDF

TF-IDF 是两个统计量的乘积:

```text
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

### 3.1 TF: Term Frequency, 词频

它衡量一个词在当前文档中出现得有多频繁。

常见定义:

```text
原始词频:
TF(t, d) = f(t, d)

归一化词频:
TF(t, d) = f(t, d) / |d|

对数词频:
TF(t, d) = 1 + log(f(t, d))
```

其中:

```text
f(t, d) = 词 t 在文档 d 中的次数
|d|     = 文档 d 的词数
```

### 3.2 IDF: Inverse Document Frequency, 逆文档频率

它衡量一个词在整个语料中有多稀有。

常见定义:

```text
IDF(t) = log(N / df(t))
```

或加平滑:

```text
IDF(t) = log((N + 1) / (df(t) + 1)) + 1
```

其中:

```text
N       = 文档总数
df(t)   = 包含词 t 的文档数
```

直觉:

```text
"的" 出现在几乎所有文档中 -> df 很大 -> IDF 很小
"VRRP" 只出现在少数文档中 -> df 很小 -> IDF 很大
```

### 3.3 查询打分

查询 `q` 与文档 `d` 的 TF-IDF 分数:

```text
score(q, d) = Σ_{t ∈ q} TF-IDF(t, d)
```

如果查询词也在查询中重复, 还可以乘查询词频 `qtf(t, q)`:

```text
score(q, d) = Σ_{t ∈ q} qtf(t, q) * TF-IDF(t, d)
```

### 3.4 作用与局限

作用:

- 稀有词获得更高权重。
- 比纯词袋更能反映关键词重要性。
- 是 BM25 的历史前身。

局限:

```text
1. 词频线性增长
   一个词从出现 1 次变成 10 次, 权重可能一直线性上涨。

2. 文档长度处理粗糙
   长文档天然更容易包含更多词。

3. 没有词频饱和
   关键词出现 100 次不应该比出现 10 次重要 10 倍。
```

BM25 主要是针对这些问题做的改进。

---

## 四、BM25

BM25 全称通常是 Okapi BM25, 是概率检索模型中的经典排序函数。它在 TF-IDF 的基础上加入:

```text
1. 词频饱和
2. 文档长度归一化
3. 更好的稀有词权重形式
```

### 4.1 公式

查询 `q` 对文档 `d` 的 BM25 分数:

```text
score_BM25(q, d)
= Σ_{t ∈ q}
    IDF(t)
    *
    f(t, d) * (k1 + 1)
    -----------------------------------------
    f(t, d) + k1 * (1 - b + b * |d| / avgdl)
```

其中:

```text
q        查询
d        文档或 chunk
t        查询中的一个词
f(t, d)  词 t 在文档 d 中的出现次数
|d|      文档 d 的长度
avgdl    语料中所有文档的平均长度

k1       词频饱和参数, 常见 1.2 ~ 2.0
b        文档长度归一化强度, 常见 0.75
```

本项目的实现使用:

```text
k1 = 1.2
b  = 0.75
```

代码位置:

```text
../pipeline/retrieval_channel_compare.py
```

该实现使用的 IDF 形式是:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

不同搜索引擎可能使用略有差异的 IDF 平滑形式, 但核心思想相同。

### 4.2 逐项解析

#### `f(t, d) * (k1 + 1)`

词频越高, 分子越大。

#### 分母中的 `f(t, d) + k1 * (...)`

当词频增加时, 分数不是无限线性增长, 而是逐渐饱和。

```text
出现 1 次  -> 重要性快速上升
出现 5 次  -> 增幅开始变小
出现 50 次 -> 再增加很多次的边际收益很低
```

这就是**词频饱和**。

#### `|d| / avgdl`

长文档通常更容易命中查询词。BM25 用长度归一化降低这种天然优势。

#### `b = 0`

不做文档长度归一化。

#### `b = 1`

完全使用文档长度归一化。

#### `k1`

控制 TF 的饱和速度:

```text
k1 较小 -> 更快饱和
k1 较大 -> 词频继续增长的空间更大
```

### 4.3 一个简化的直觉例子

假设两个文档都包含 `keepalived`:

```text
文档 A: 长度正常, keepalived 出现 1 次
文档 B: 长度超长, keepalived 出现 10 次
```

TF-IDF 可能认为 B 因词频高而明显占优。BM25 会同时考虑:

```text
B 的 keepalived 词频虽然高, 但已经饱和;
B 的文档长度很长, 需要受到长度惩罚。
```

因此 BM25 通常比原始 TF-IDF 更稳。

### 4.4 BM25 的作用

BM25 擅长:

```text
产品型号           TS-999
配置项名           virtual_ipaddress
文件名             keepalived.conf
错误码             HTTP 502
函数名             SentenceWindowNodeParser
版本号和章节号      v2.4.1 / 第 3.2 节
用户直接粘贴的短语  "invalid ip number count"
```

BM25 不擅长:

```text
同义表达
    "主节点挂掉后自动切换"
    vs
    "故障转移"

跨语言表达
    "负载均衡"
    vs
    "load balancing"

多义词消歧
    "Apple 公司" vs "Apple 水果"

长距离语义关系
    查询讲的是机制, 文档用完全不同的词解释同一个机制
```

### 4.5 BM25 与中文分词

中文文档不能天然按空格切词。BM25 之前通常要先分词。

```text
原文:
    keepalived 使用 VRRP 发送通告

分词:
    ["keepalived", "使用", "VRRP", "发送", "通告"]
```

常见选择:

```text
jieba          简单、便宜、适合 Demo
Lucene 分词器    搜索引擎常用
行业词典        对产品名和专有名词很重要
```

分词错误会直接影响 BM25:

```text
错误切分 -> 查询词和文档词对不上 -> 明明文档里有, BM25 却匹配不到
```

### 4.6 BM25 变体

常见变体包括:

```text
BM25+      给每个匹配词增加下限项, 避免过长文档里的匹配被过度惩罚
BM25F      处理标题、正文、锚文本等不同字段
BM25L      调整长度归一化
```

实际工程里不需要一开始就换复杂变体, 标准 BM25 通常已经能提供明显的字面召回能力。

---

## 五、共用概念: 稀疏召回与稠密召回

dense 和 sparse 是两条通用的检索范式, 不属于 RRF, 也不属于某个框架。它们的完整原理统一维护在:

```text
sparse_vs_dense_retrieval.md
```

只在本文保留定位:

```text
稀疏召回   BM25 / TF-IDF / SPLADE     强在字面、编号、配置项
稠密召回   BGE / Qwen embedding       强在同义、改写、语义
```

本文后续只讨论一个独立问题:

```text
两路召回已经产生排名之后, 如何把排名融合成最终结果?
```

答案就是 RRF, 以及它前后的候选深度、rerank 和工程约束。

---

## 六、混合检索(Hybrid Retrieval)

### 6.1 基本结构

```text
query
  |
  +-- dense channel ------> top-20
  |
  +-- sparse channel -----> top-20
              |
              v
      RRF / 加权融合 ------> top-10
              |
              v
        rerank -----------> top-5
```

### 6.2 为什么要两路

假设查询是:

```text
"keepalived.conf 里 virtual_ipaddress 怎么配"
```

稠密通道可能召回:

```text
"Keepalived 的虚拟 IP 地址配置..."
```

稀疏通道更可能精确命中:

```text
"virtual_ipaddress"
"keepalived.conf"
```

两路各有一个正确答案的不同侧面。融合的目的是让两个信号都参与最终排序。

### 6.3 候选深度

常见做法是:

```text
每路先取 20 ~ 100
融合后取 10 ~ 50
rerank 后取 5 ~ 10
```

不要一开始只取 top-3 再融合。这样会把某一路本来在 top-8、但融合后可以进入 top-3 的文档提前丢掉。

---

## 七、RRF: Reciprocal Rank Fusion

RRF 的中文是**倒数排名融合**。它的目标是: 不比较不同检索器的原始分数, 只根据文档在各路结果里的排名来合并名次。

### 7.1 为什么需要排名融合

向量检索返回:

```text
cosine similarity = 0.83
```

BM25 返回:

```text
BM25 score = 18.72
```

这两个数不能直接相加:

```text
0.83 + 18.72 没有意义
```

即使做 min-max 归一化, 也会受当前候选集合、异常分数和分布影响。RRF 的解决方式是:

```text
不关心分数是多少
只关心文档在第 1 路排第几、第 2 路排第几
```

### 7.2 `rank_i(d)` 是什么

公式:

```text
rank_i(d)
```

它表示:

```text
文档 d 在第 i 条检索通道结果中的名次
```

它是一个函数, 不是乘法。

假设:

```text
dense 结果:
    1. A
    2. B
    3. C

BM25 结果:
    1. B
    2. C
    3. A
```

那么:

```text
rank_dense(A) = 1
rank_dense(B) = 2
rank_dense(C) = 3

rank_bm25(A) = 3
rank_bm25(B) = 1
rank_bm25(C) = 2
```

形式化定义:

```text
如果第 i 路排名列表 L_i 的第 j 个元素是 d:
    rank_i(d) = j

如果 d 没有出现在第 i 路列表里:
    rank_i(d) = ∞
```

名次通常从 `1` 开始计数, 不是从 `0`。

### 7.3 RRF 标准公式

```text
RRF(d) = Σ_i 1 / (k + rank_i(d))
```

其中:

```text
i          第 i 条检索通道
d          候选文档或 chunk
rank_i(d)  文档 d 在第 i 路结果中的名次
k          平滑常数, 常见默认值是 60
```

如果文档没有出现在某一路:

```text
1 / (k + ∞) = 0
```

实际实现里通常直接不对该路累加。

### 7.4 RRF 计算例子

假设:

```text
dense 排名:
    1. A
    2. B
    3. C
    4. D

BM25 排名:
    1. B
    2. C
    3. A
```

取 `k = 60`。

#### A

```text
A = 1 / (60 + rank_dense(A)) + 1 / (60 + rank_bm25(A))
  = 1 / (60 + 1) + 1 / (60 + 3)
  = 1 / 61 + 1 / 63
  = 0.032266
```

#### B

```text
B = 1 / (60 + 2) + 1 / (60 + 1)
  = 1 / 62 + 1 / 61
  = 0.032522
```

#### C

```text
C = 1 / (60 + 3) + 1 / (60 + 2)
  = 1 / 63 + 1 / 62
  = 0.032002
```

#### D

```text
D = 1 / (60 + 4) + 0
  = 1 / 64
  = 0.015625
```

最终顺序:

```text
1. B
2. A
3. C
4. D
```

B 虽然没有同时在两路排第一, 但它在两路都很靠前, 因此 RRF 认为它更稳定。

### 7.5 `k` 的作用

`k` 不是 top-k, 也不是取前 k 条。

它控制高排名的优势有多强:

```text
k = 0
    rank 1 -> 1
    rank 2 -> 0.5
    第 1 名优势非常大

k = 60
    rank 1 -> 1 / 61 = 0.016393
    rank 2 -> 1 / 62 = 0.016129
    第 1 名和平名次差距被压平
```

`k` 越大:

```text
不同排名之间的差异被压缩
融合更平滑
单个检索器更难靠一个第 1 名压过其他通道的共识
```

RRF 原始论文常用的默认值是:

```text
k = 60
```

但这不是固定物理常数。实际系统应该根据评估集和候选深度调整。

### 7.6 加权 RRF

如果某个通道更可信, 可以给通道加权重:

```text
RRF_weighted(d) = Σ_i w_i / (k + rank_i(d))
```

其中:

```text
w_i = 第 i 条通道的权重
```

例如:

```text
dense 权重 1.0
BM25  权重 0.8
```

权重不能凭感觉长期固定, 应该用评估集验证。否则容易变成"把参数当结论"。

### 7.7 RRF 伪代码

```python
def rrf(rank_lists, k=60):
    scores = {}

    for ranked_docs in rank_lists:
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0)
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)
```

调用:

```python
dense_top20 = dense_search(query, limit=20)
sparse_top20 = bm25_search(query, limit=20)

fused = rrf(
    [dense_top20, sparse_top20],
    k=60,
)

final_top10 = fused[:10]
```

### 7.8 RRF 的优点

```text
1. 不需要不同通道的分数可比
2. 不需要训练一个融合模型
3. 对异常分数不敏感
4. 实现简单, 容易调试
5. 对异构检索器很通用
```

### 7.9 RRF 的局限

```text
1. 丢掉了原始分数强度
   一个 0.99 的向量分数和一个 0.60 的向量分数,
   如果排名只差一位, RRF 基本看不出差异。

2. 不能修复错误候选
   如果 BM25 和 dense 都没有召回正确 chunk, RRF 不可能把它变出来。

3. 不负责分块质量
   如果 chunk 本身混合了多个主题, RRF 只会把错误 chunk 排得更稳定。

4. 融合权重仍然需要评估
   两路并不总是一样可靠。

5. 候选深度会影响结果
   只融合每路 top-3 和融合 top-50, 结果可能完全不同。
```

最重要的一点:

```text
RRF 是排序融合器, 不是万能纠错器。
它只能在已经召回出来的候选里重新组合优势。
```

---

## 八、RRF、BM25、BGE、Qdrant 的关系

```text
文本
  |
  +-- BM25 / TF-IDF  ---------> 稀疏相关性分数
  |
  +-- BGE / Qwen embedding ---> 稠密向量
                                  |
                                  v
                             Qdrant 等向量库
                                  |
                                  v
                              dense 排名
                                  |
BM25 排名 ------------------------+
                                  |
                                  v
                                RRF
                                  |
                                  v
                             融合后排名
                                  |
                                  v
                              cross-encoder
                                  |
                                  v
                             最终 top-k
```

一句话:

```text
Qdrant 负责存向量和做向量检索。
BM25 负责字面检索。
BGE/Qwen 负责把文本变成向量。
RRF 负责把多个排名列表合成一个排名列表。
Rerank 负责对少量候选做更贵、更准的精排。
```

### 关于“BGE 和 BM25 哪个更好”

这个问题本身就是错的。它们擅长不同信号:

```text
BM25 会问: 查询词是否字面出现在文档里?
BGE  会问: 文档整体语义是否接近查询?
```

生产系统通常不是二选一, 而是:

```text
BM25 + dense embedding + RRF + rerank
```

---

## 九、当前项目应该怎么落地

当前工程已经落地 dense + sparse + RRF:

```text
query
  |
  +-- Qdrant redhat dense ------> cosine top-20
  |
  +-- Qdrant redhat_sparse -----> BM25 top-20
              |
              v
          RRF(k=60)
              |
              v
        gte-rerank-v2
              |
              v
          最终 top-5
```

实现位于:

```text
../pipeline/hybrid_retriever.py
../pipeline/rag_server.py
../pipeline/static/index.html
```

API 和页面支持:

```text
mode=hybrid   两路召回 + RRF
mode=dense    只看稠密向量
mode=sparse   只看 BM25 sparse
```

原始验证输出和页面截图:

```text
../pipeline/experiments/16_hybrid_rrf.txt
../pipeline/experiments/16_hybrid_ui.png
../pipeline/experiments/16_hybrid_ui_mobile.png
```

生成阶段也已接入:

```text
../pipeline/generation.py
POST /api/answer
hybrid top-k -> qwen-turbo/qwen-plus -> answer + [C1] citations
```

生成实验见 `../pipeline/experiments/17_generation.txt`。
rerank 实验见 `../pipeline/experiments/19_rerank_compare.txt`。

问题记录在:

```text
../issues/04_channel_layer.md
```

下一阶段链路:

```text
query
  |
  +-- Qdrant dense ------> top-20
  |
  +-- BM25 -------------> top-20
              |
              v
             RRF
              |
              v
          top-10
              |
              v
        rerank
              |
              v
          top-5
```

### 9.1 先做最小可用版本

```text
1. 复用当前已经生成的 chunk 文本
2. 对 chunk 文本做中文分词
3. 建立内存 BM25 索引
4. 每路各取 top-20
5. 用 k=60 的 RRF 融合
6. 在线检索接口返回两路排名和融合排名
7. 建评估集后再调整权重和 k
```

### 9.2 需要落地的评估指标

没有评估集时, 只能看代理指标。真正应记录的是:

```text
Recall@k      正确 chunk 是否出现在前 k 条
MRR           正确 chunk 的平均倒数排名
Precision@k   前 k 条里有多少是真正相关
nDCG@k        相关性有多好, 且位置越靠前收益越大
```

### 9.3 先别做的事

```text
不要在还没有评估集时:
    - 固定宣称 BM25 权重必须是某个值
    - 固定宣称 RRF 一定优于分数融合
    - 用几个 query 的结果代表整体检索质量
    - 把"看起来更相关"当成真实指标
```

---

## 十、最后一页速记

```text
BoW
    文档 = 词频表
    只数词, 丢词序

TF-IDF
    词的重要性 = 词频 * 逆文档频率
    稀有词权重更高

BM25
    TF-IDF 的工程改进版
    加词频饱和 + 文档长度归一化
    强项: 精确词、配置项、错误码、文件名

Dense embedding
    文本 -> 固定长度向量
    强项: 同义词、语义表达、自然语言问题

Cosine similarity
    检索排序看相似度, 越大越好

Cosine distance
    语义分块看距离, 越大越可能是边界

RRF
    RRF(d) = Σ_i 1 / (k + rank_i(d))
    只看排名, 不比较原始分数
    k 常取 60, 但应通过评估集调参

Hybrid
    dense + sparse 两路召回
    RRF 融合
    rerank 精排
```

最终记住一句话:

```text
BM25 负责“字面命中”,
embedding 负责“语义相近”,
RRF 负责“合并排名”,
rerank 负责“精挑细选”。
```
