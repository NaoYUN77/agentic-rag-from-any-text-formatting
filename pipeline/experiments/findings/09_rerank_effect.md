# 实验 09: RRF + Rerank 的 smoke 对比

## 1. 实验条件

```text
日期            2026-09-14
检索模式        hybrid
候选参数        candidate_k=20, rrf_k=60
rerank          gte-rerank-v2
rerank 候选     20
评估集          pipeline/eval/rerank_smoke.jsonl
查询数          8
指标            Recall@5 / Recall@10 / MRR@20
脚本            pipeline/rerank_compare_demo.py
原始输出        pipeline/experiments/19_rerank_compare.txt
```

这是小样本 smoke test, 不是完整评估集。

## 2. 聚合结果

```text
指标         RRF       RRF + rerank
Recall@5     0.808     0.777
Recall@10    0.902     0.933
MRR@20       0.875     0.938
```

解释:

```text
MRR 提升
    正确 chunk 整体排得更靠前

Recall@10 提升
    更多正确 chunk 进入前 10

Recall@5 略降
    有查询的某个正确 chunk 被重新排到第 5 名之后
```

## 3. 典型变化

### 3.1 arptables 查询

```text
第一个正确 chunk 排名:
RRF       #2
rerank    #1
```

rerank 把更直接回答问题的 chunk 提到了第一位。

### 3.2 VIP 查询

```text
Recall@5    0.500 -> 0.750
Recall@10   0.750 -> 1.000
```

正确证据被更集中地排进前 5 和前 10。

### 3.3 HAProxy 算法查询

```text
Recall@5    1.000 -> 0.500
```

一个相关 chunk 从第 5 名后被移到第 6 名，导致 Recall@5 下降，但 Recall@10 仍然为 1.0。

这说明 rerank 会改变边界位置，不能默认它在所有 `k` 上都只增不减。

## 4. 延迟

```text
rerank 耗时约 1.1 ~ 1.4 秒/查询
```

相比几毫秒到百毫秒级的召回，rerank 是明显的额外开销。

## 5. 结论

```text
rerank 在当前小评估集上:
    提升了整体排序质量
    提升了 Recall@10
    但轻微降低了 Recall@5
    增加了约 1 秒级延迟
```

不能据此宣布 rerank 全面优于 RRF。更准确的结论是:

```text
rerank 能把一部分正确证据提前,
但改变候选边界后也可能把个别结果推出 top-5。
```

## 6. 当前边界

```text
1. 只有 8 条 query, 统计意义有限。
2. gold chunk 是人工 smoke 标注, 不是完整 qrels。
3. 只测了 gte-rerank-v2, 没比较其它 reranker。
4. 未测试 rerank_candidate_k 从 10 / 20 / 50 的变化。
5. 未测 rerank 后再生成答案的最终质量。
```

## 7. 下一步

```text
1. 扩大评估集到 30 ~ 50 条。
2. 固定 dev/test split。
3. 比较 rerank 候选深度 10 / 20 / 50。
4. 比较 gte-rerank-v2 与本地 cross-encoder。
5. 同时看 Recall@5、Recall@10、MRR、nDCG 和 rerank 延迟。
```
