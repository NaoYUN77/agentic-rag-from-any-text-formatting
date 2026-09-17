# 混合检索与 RRF: 多路召回、候选深度、融合与重排

> 本文只讨论混合检索本身。
>
> Sparse 通道内部原理见:
> - `sparse_retrieval_concepts.md`
>
> Dense 与 Sparse 的系统差异见:
> - `sparse_vs_dense_retrieval.md`
>
> Top-K、candidate_k、rerank_candidate_k、context_k 见:
> - `top_k_and_candidate_depth.md`

---

## 一、先明确混合检索不是什么

混合检索不是：

```text
dense_score + sparse_score
```

也不是：

```text
把两个向量数据库拼起来
```

更不是：

```text
某个算法本身
```

混合检索是一个工程架构：

```text
同一查询
-> 进入多个独立召回通道
-> 每个通道产生自己的候选和排名
-> 在候选层做融合
-> 可选重排
-> 得到最终结果
```

BM25 可以是 sparse 通道的一种实现，但 BM25 不等于混合检索。

---

## 二、核心术语

### 2.1 Retrieval Channel

一条独立的召回通道必须具备：

```text
自己的文本表示
自己的索引
自己的相似度或打分方式
自己的排序结果
```

例如：

```text
Dense channel
Sparse channel
Metadata channel
Graph channel
Rule channel
```

通道不是数据库类型。

一个向量数据库可以承载多种通道。

一个进程也可以同时运行多个通道。

### 2.2 Candidate

候选是：

```text
某条 query 在某条通道中召回出来的文档或 chunk
```

候选通常带有：

```text
point_id / chunk_id
channel_name
rank
score
payload
```

### 2.3 Rank

Rank 是候选在该通道中的名次：

```text
rank = 1
    排名第一

rank = 2
    排名第二
```

Rank 和 score 不同：

```text
score
    原始打分或相似度

rank
    在一条通道内部排序后的位置
```

### 2.4 Fusion

Fusion 是：

```text
把多条通道的候选排名合并成一个统一排名
```

### 2.5 Reranker

Reranker 是：

```text
对已经召回的候选做更昂贵的精排
```

Reranker 不负责从全库召回。

它只能重排已经进入候选池的文档。

---

## 三、为什么需要混合检索

单一检索通道常有固定失效模式。

Dense 通道擅长：

```text
语义相似
同义改写
自然语言问句
跨语言语义
```

Dense 通道容易漏掉：

```text
精确编号
罕见字符串
配置项
错误码
版本号
```

Sparse 通道擅长：

```text
精确词项匹配
领域术语
专有名词
配置项
错误码
```

Sparse 通道容易漏掉：

```text
同义表达
语义改写
上下文依赖表达
```

混合检索的价值来自：

```text
两条通道的误差不完全相关
```

如果两个通道总是召回同一批文档：

```text
混合收益很低
```

如果两路擅长不同问题：

```text
混合可能显著提高 Recall
```

---

## 四、Sparse 通道不等于 BM25

Sparse 描述的是表示方式：

```text
高维
大部分维度为 0
非零维度通常对应词项或学习出来的稀疏特征
```

Sparse 通道可以有多种实现：

```text
BM25
TF-IDF
SPLADE
BGE-M3 sparse
其他 learned sparse
```

因此本文后面统一写：

```text
sparse channel
```

而不是：

```text
BM25 channel
```

BM25 的公式、词表和倒排索引细节放在 sparse 专题文档里。

---

## 五、Dense 通道不等于向量数据库

Dense 通道至少包含：

```text
embedding 模型
文本向量化
向量相似度
候选召回
```

向量数据库只负责：

```text
存储向量
建立 ANN 索引
执行相似度搜索
返回候选
```

所以：

```text
Dense channel
    = embedding model + vector search

Vector DB
    = 向量存储与检索基础设施
```

Qdrant、FAISS、Milvus 都没有替用户产生语义 embedding。

---

## 六、典型混合架构

### 6.1 并行召回

最常用：

```text
                    query
                      |
          +-----------+-----------+
          |                       |
          v                       v
     Channel A               Channel B
      top-N1                  top-N2
          |                       |
          +-----------+-----------+
                      |
                      v
                 candidate pool
                      |
                      v
                    fusion
                      |
                      v
                   reranker
                      |
                      v
                 final top-k
```

特点：

```text
各通道独立
可以并行执行
融合只发生在候选层
```

### 6.2 串行级联

例如：

```text
rule / metadata filter
-> dense recall
-> sparse recall
-> fusion
-> rerank
```

特点：

```text
前面的步骤缩小搜索空间
后面的步骤提高精度
```

缺点：

```text
早期过滤错误无法被后续步骤恢复
```

### 6.3 混合多级架构

实际系统通常是多级组合：

```text
query understanding
-> filters
-> parallel recall
-> fusion
-> rerank
-> final selection
-> context construction
```

---

## 七、候选池

### 7.1 候选池从哪里来

候选池可以是：

```text
union
    两路候选取并集

intersection
    两路候选取交集

quota
    每路保留固定数量

weighted
    按通道权重分配候选预算
```

默认最常用：

```text
union
```

因为：

```text
交集会直接丢掉单路独有的结果
```

### 7.2 候选池的去重

同一文档可能同时出现在：

```text
dense channel
sparse channel
metadata channel
```

融合前必须统一身份：

```text
chunk_id
point_id
document_id
```

否则同一内容会被当作不同候选。

推荐：

```text
所有通道使用同一个 chunk_id
```

---

## 八、分数不可比问题

Dense 分数可能来自：

```text
cosine similarity
dot product
L2 distance
```

Sparse 分数可能来自：

```text
未归一化 BM25
归一化后的词项权重
学习型稀疏打分
```

因此：

```text
dense_score 和 sparse_score 通常不在同一尺度
```

例如：

```text
dense_score = 0.82
sparse_score = 12.7
```

不能说：

```text
0.82 + 12.7
```

而应该：

```text
先分别排序
再基于 rank 融合
```

RRF 的核心价值就在这里。

---

## 九、RRF 的概念

RRF 全称：

```text
Reciprocal Rank Fusion
倒数排名融合
```

它使用名次，不使用原始 score。

### 9.1 基本公式

对文档 `d`：

```text
RRF_score(d) = sum_i( 1 / (k + rank_i(d)) )
```

其中：

```text
i
    第 i 条召回通道

rank_i(d)
    文档 d 在第 i 条通道中的名次

k
    RRF 平滑常数
```

如果文档没有出现在某条通道：

```text
rank_i(d) 不存在
该通道对它的贡献为 0
```

### 9.2 k 的作用

典型值：

```text
k = 60
```

`k` 越大：

```text
排名差值被压平
不同名次之间差距变小
```

`k` 越小：

```text
前几名优势更明显
```

`k` 不是 Top-K。

```text
rrf_k
    公式平滑常数

top_k
    保留多少条
```

### 9.3 RRF 的优点

- 不需要跨通道归一化 score
- 对异常分数不敏感
- 对通道数量扩展友好
- 实现简单
- 适合并行召回

### 9.4 RRF 的局限

- 丢掉原始 score 的幅度信息
- 通道质量差时也会被平等计入
- 通道强相关时收益有限
- k 和权重仍需评估
- 不能代替 Reranker

---

## 十、加权 RRF

不同通道通常可信度不同。

可以加入通道权重：

```text
RRF_score(d) = sum_i( w_i / (k + rank_i(d)) )
```

其中：

```text
w_i
    第 i 条通道的权重
```

例如：

```text
dense_weight  = 1.0
sparse_weight = 0.6
```

需要注意：

```text
权重乘在 RRF 贡献上
不是直接乘原始 score
```

常见情况：

```text
某个通道离线评估更好
-> 提高权重

某个通道在特定领域不可靠
-> 降低权重

某个通道只作为补充
-> 保持较小权重
```

---

## 十一、RRF 计算例子

假设：

```text
Dense ranks:
    1 A
    2 B
    3 C
    4 D

Sparse ranks:
    1 C
    2 A
    3 E
    4 F

k = 60
```

计算：

```text
A:
    dense  rank 1 -> 1/61
    sparse rank 2 -> 1/62
    total > B

B:
    dense  rank 2 -> 1/62
    sparse absent -> 0

C:
    dense  rank 3 -> 1/63
    sparse rank 1 -> 1/61

E:
    dense  absent -> 0
    sparse rank 3 -> 1/63
```

可以看到：

```text
同一个文档在两路都靠前时，RRF 会得到更高分
只在一个通道出现的文档，也能保留贡献
```

---

## 十二、候选深度

候选深度决定：

```text
在融合之前，每条通道保留多少候选
```

例如：

```text
dense_top_k  = 50
sparse_top_k = 50
```

或者：

```text
candidate_k = 50
```

含义可能是：

```text
每路各取 50
```

候选深度越大：

```text
召回率可能提高
后续融合和重排成本增加
```

候选深度太小：

```text
正确文档可能在融合前就被截掉
```

核心原则：

```text
candidate_k 通常应大于 final_top_k
```

如果正确文档没有进入候选池：

```text
RRF 无法把它找回
Reranker 无法给它重新排序
```

---

## 十三、Reranker 在混合检索中的位置

典型顺序：

```text
Channel A top-N
Channel B top-N
-> fusion
-> candidate pool
-> reranker
-> final top-k
```

Reranker 看到的是：

```text
已经召回的候选
```

它看不到：

```text
没有被召回的文档
```

所以：

```text
召回负责覆盖正确文档
Rerank 负责排列正确文档
```

### 13.1 Reranker 的收益

可能带来：

```text
更高精度
更高的前几个位置相关率
更好的上下文质量
```

### 13.2 Reranker 的成本

包括：

```text
额外延迟
额外费用
候选数增加时成本上升
```

所以通常会设置：

```text
rerank_candidate_k
```

只对少量候选重排。

---

## 十四、Final Top-K 和 Context K

融合后还要区分：

```text
final_top_k
    最终展示或返回给用户多少条

context_k
    实际送入 LLM 上下文多少条
```

可能关系：

```text
final_top_k = 10
context_k   = 5
```

表示：

```text
用户看到 10 条
LLM 只读取最前面的 5 条
```

也可能：

```text
final_top_k = context_k = 5
```

这取决于：

```text
上下文窗口
每 chunk 长度
生成成本
展示需求
```

---

## 十五、混合检索的失败模式

### 15.1 通道高度相关

如果 dense 和 sparse 召回几乎相同：

```text
混合收益很小
```

需要看：

```text
单路 Recall@k
两路并集 Recall@k
RRF Recall@k
```

### 15.2 单通道主导

如果某一路分数范围特别大：

```text
直接相加时可能吞掉另一路
```

RRF 可以缓解这个问题。

### 15.3 候选深度过低

表现为：

```text
最终结果差
Reranker 也无法修复
```

原因通常是：

```text
正确文档没有进入候选池
```

### 15.4 分数归一化错误

例如：

```text
把 dense cosine 和未归一化的 sparse score 直接融合
```

应优先使用 rank fusion 或经过验证的归一化方案。

### 15.5 身份不一致

两路没有使用同一个：

```text
chunk_id
point_id
```

会导致：

```text
同一内容重复出现
RRF 无法合并
```

### 15.6 Reranker 位置错误

Reranker 只能处理候选池。

如果候选池已经缺少正确文档：

```text
Reranker 无能为力
```

### 15.7 没有评估集

调参会变成：

```text
凭感觉看 top-5
```

无法判断：

```text
是召回问题
还是融合问题
还是重排问题
```

---

## 十六、混合检索评估

### 16.1 分通道评估

先看：

```text
Dense Recall@k
Sparse Recall@k
```

如果单路都低：

```text
先修召回器和 chunk
不要直接调 RRF
```

### 16.2 看候选并集

看：

```text
union Recall@k
```

它回答：

```text
两路合起来，正确文档有没有进入候选池
```

### 16.3 看融合结果

看：

```text
RRF Recall@k
MRR
nDCG
```

它回答：

```text
融合有没有把正确文档排到前面
```

### 16.4 看 Rerank 增益

比较：

```text
Rerank 前 nDCG
Rerank 后 nDCG
```

如果提升有限：

```text
说明召回候选已经不错，或者 reranker 不适合当前领域
```

---

## 十七、参数调节顺序

推荐顺序：

```text
1. 固定 chunk 策略
2. 评估 dense Recall@k
3. 评估 sparse Recall@k
4. 评估 union Recall@k
5. 调整 candidate_k
6. 调整 rrf_k
7. 调整通道权重
8. 调整 rerank_candidate_k
9. 选择 final_top_k
10. 最后确定 context_k
```

不要同时修改：

```text
chunk_size
embedding model
candidate_k
rrf_k
reranker
context_k
```

否则无法归因。

---

## 十八、当前项目实现

当前项目使用：

```text
Dense Channel
    Qwen embedding
    Qdrant dense vector
    cosine similarity

Sparse Channel
    当前实现为 jieba + BM25
    写入 Qdrant sparse vector

Fusion
    RRF
    默认 rrf_k = 60

Rerank
    gte-rerank-v2

Generation
    qwen-turbo
```

默认参数：

```text
candidate_k        = 20
rrf_k              = 60
rerank_candidate_k = 20
final_top_k        = 5
context_k          = 5
```

当前流程：

```text
query
-> dense top-20
-> sparse top-20
-> RRF
-> rerank candidate top-20
-> final top-5
-> context top-5
```

实现位置：

```text
pipeline/hybrid_retriever.py
pipeline/rag_server.py
pipeline/static/index.html
```

这里需要特别说明：

```text
BM25 只是当前 sparse 通道的实现
不是混合检索的定义
不是 RRF 的前提
```

未来可以替换：

```text
BM25
-> SPLADE
-> BGE-M3 sparse
-> 其他 learned sparse
```

只要它仍然提供：

```text
候选
rank
chunk_id
```

就可以继续进入同一套 RRF 融合。

---

## 十九、一页速记

```text
混合检索
    多条通道独立召回

Dense channel
    语义向量

Sparse channel
    稀疏词项或学习型稀疏表示
    当前项目用 BM25

Candidate pool
    多路候选统一去重

RRF
    基于 rank 的排名融合

rrf_k
    公式平滑常数

candidate_k
    每路候选深度

rerank_candidate_k
    送入 Reranker 的候选数

final_top_k
    最终展示条数

context_k
    送入 LLM 的上下文条数
```

最重要的一句话：

```text
混合检索不是把分数相加，
而是让不同召回机制各自提供候选，
再用排名融合和重排逐步提高最终上下文质量。
```