# 实验 06: Dense + Sparse + RRF 融合

## 1. 实验条件

```text
日期            2026-09-13
dense 集合      redhat (22 points)
sparse 集合     redhat_sparse (22 points)
dense 模型      qwen3-vl-embedding, 1024 维
sparse 分析器   jieba + BM25(k1=1.2, b=0.75)
每路候选数      20
RRF k           60
最终输出        5
实现            pipeline/hybrid_retriever.py
服务            pipeline/rag_server.py
页面            pipeline/static/index.html
原始输出        pipeline/experiments/16_hybrid_rrf.txt
桌面截图        pipeline/experiments/16_hybrid_ui.png
移动截图        pipeline/experiments/16_hybrid_ui_mobile.png
```

## 2. 实测结果

同一条查询分别运行三种模式:

```text
dense top-5    4, 0, 5, 12, 15
sparse top-5   0, 1, 4, 3, 8
hybrid top-5   0, 4, 3, 1, 15
```

融合明细:

```text
rank  point_id  dense_rank  sparse_rank  RRF
1     0         2           1            0.032522
2     4         1           3            0.032266
3     3         6           4            0.030777
4     1         9           2            0.030622
5     15        5           6            0.030536
```

## 3. 结论

### 3.1 RRF 确实把两路信号合并了

`point_id=0` 在 dense 排名第 2、sparse 排名第 1，最终 RRF 第 1。

`point_id=4` 在 dense 排名第 1、sparse 排名第 3，最终 RRF 第 2。

这说明 RRF 不是简单复制 dense 或 sparse 排名，而是奖励“两路都靠前”的 chunk。

### 3.2 单路强项可以互补

`point_id=1` 在 sparse 排名第 2，但 dense 排名第 9，融合后到第 4。

这类候选通常是关键词命中强、语义相似度一般的结果。RRF 让它保留在最终 top-5，
而不是被 dense 完全淘汰。

### 3.3 融合后的分差很小

本次 top1 - top2 的 RRF 分差约为 `0.0003`。在 `k=60` 时 RRF 分数本身很小，
不能拿它和 cosine 或 BM25 的绝对分数直接比较。

需要关注的是:

```text
排名是否稳定
是否把正确 chunk 提到前 k
Recall@k / MRR 是否改善
```

而不是 RRF 分数本身是否大于某个固定阈值。

### 3.4 页面已验证

FastAPI 服务已返回 hybrid 结果，页面能显示:

```text
RRF 分数
D#dense_rank
S#sparse_rank
point_id
section/page
```

桌面和 390px 移动视口都完成了 Playwright 渲染检查，移动端没有横向溢出。

## 4. 当前边界

```text
1. 只验证了一条查询, 还不能证明 RRF 的整体检索质量更好。
2. dense 和 sparse 仍然是两个 collection, 通过相同 point_id 对齐。
3. RRF 当前两路权重相同, 还没有权重调优。
4. candidate_k=20 和 k=60 尚未基于评估集调参。
5. 没有接入 cross-encoder rerank。
```

## 5. 下一步

```text
1. 建立带正确 chunk 标注的评估集。
2. 对比 dense / sparse / hybrid 的 Recall@k、MRR、nDCG。
3. 增加加权 RRF 或分数融合实验。
4. 在融合后接入 cross-encoder rerank。
5. 再评估是否迁移为同一个 collection 的 dense/sparse named vectors。
```
