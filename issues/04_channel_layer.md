# 问题 04: 只有稠密一路检索, 字面匹配能力缺失

**优先级: P1**　**状态: 部分已解决**

## 现象

现在的检索链路只有一条:

```text
query -> 嵌入 -> 和 22 个 chunk 向量算余弦 -> top-k
```

**没有关键词/稀疏通道。**

## 根因

**稠密向量和稀疏匹配是"互补"的, 不是"替代"的。**

```text
查询 "Error code TS-999"

稠密向量    找"关于错误码的通用描述"      -> 语义对, 精确错
BM25        找"包含 TS-999 这个字符串"    -> 精确对
```

**两者的能力不重叠。**

具体到我们这次语料里的例子:

```text
"HAProxy 支持哪些负载均衡算法"
  -> 稠密向量成功, 因为"负载均衡算法"是语义概念

"VRRP 版本不匹配会有什么问题"
  -> 也成功了, 因为这句话语义完整

但如果查询是 "keepalived.conf 里 virtual_ipaddress 怎么配"
  -> 稠密向量可能失败 —— 那是文件路径 + 配置项名, 靠精确字符串
     BM25 能秒中
```

**根本原因: 稠密向量在池化那一步就把字面信息丢了。** 它编码的是"语义方向", 不保留"哪些字符出现了"。

## 影响

```text
专有名词     产品型号、函数名、错误码、配置项名 -> 容易漏召回
编号类       版本号、章节号、页码 -> 容易漏召回
精确短语     用户直接粘贴文档里的一句话 -> 语义匹配反而可能偏
```

## 解决方向

### 方向 1: 加 BM25 通道

```text
query
  ├─ 稠密向量 -> top-N1
  └─ BM25     -> top-N2
        ↓
    RRF 融合 -> top-k
```

**RRF(倒数排名融合)**: 不需要两路分数可比, 只用排名。

```text
score(d) = Σ  1 / (k + rank_i(d))
           i
```

**Anthropic 实测: 失败率从 5.7% 降到 2.9%(只加 Contextual BM25 的话)。**

### 方向 2: 用稀疏向量代替 BM25

如果要保持"向量库统一检索", 可以用学习型稀疏向量(SPLADE / bge-m3 的 sparse 输出), 存进同一个 Qdrant 集合。

**Qdrant 原生支持命名向量(named vectors)**, 可以在一个 collection 里同时存 dense 和 sparse:

```python
vectors_config={"dense": VectorParams(size=1024, distance=COSINE)}
sparse_vectors_config={"sparse": SparseVectorParams()}
```

**这样一次查询就能同时走两路, 不用维护两套索引。**

### 方向 3: Qdrant 的 Query API 融合

Qdrant 1.10+ 支持服务端融合:

```python
client.query_points(
    collection_name=...,
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=20),
        Prefetch(query=SparseVector(...), using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
)
```

**融合在服务端做, 客户端只发一次请求。**

## 待验证

```text
[ ] BM25 用 jieba 分词还是用别的? (中文需要分词器)
[ ] 用什么信号判断"这次检索需要关键词通道"?
[ ] dense + sparse 的权重怎么调? (Qdrant 的 RRF 支持权重)
[ ] 两路融合后, top-k 该取多少?
```

## 落地结果

```text
2026-09-13  基础版 dense + sparse + RRF 已实现
```

当前结构:

```text
redhat          22 个 dense points
redhat_sparse   22 个 BM25 sparse points

HybridRetriever
    dense top-20 + sparse top-20 -> RRF(k=60) -> top-5
```

实现文件:

```text
../pipeline/hybrid_retriever.py
../pipeline/rag_server.py
../pipeline/static/index.html
```

实测输出:

```text
../pipeline/experiments/16_hybrid_rrf.txt
../pipeline/experiments/16_hybrid_ui.png
../pipeline/experiments/16_hybrid_ui_mobile.png
```

剩余问题:

```text
[ ] 仍使用两个 collection, 尚未合并为同一 point 的 named vectors
[ ] 还没有 RRF 权重和 k 的评估集调参
[x] 已接入融合后的 gte-rerank-v2 rerank (第一版)
[ ] 还没有用 Recall@k / MRR 证明 hybrid 整体优于单路
```
