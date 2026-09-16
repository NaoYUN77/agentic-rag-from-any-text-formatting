# RAG 评估集

## qrels 是什么

`qrels` 是 **query relevance judgments**:

```text
query -> 哪些 chunk 是相关证据
```

它不是模型生成的检索结果, 而是人工确认后的标准标注。

当前使用 JSONL, 每一行是一条:

```json
{
  "query_id": "rh_01",
  "query": "什么是 keepalived",
  "gold": [0, 2],
  "source": "redhat",
  "category": "semantic",
  "split": "smoke"
}
```

字段说明:

```text
query_id  查询唯一编号
query     查询原文
gold      相关 chunk 的 point_id 列表
source    语料来源
category  semantic / exact / mixed / multi_hop / unanswerable
split     smoke / dev / test
```

`gold` 不是“模型认为相关的 chunk”, 而是评估时作为正确答案的 chunk。

## 当前文件

```text
pre_llamaindex/rerank_smoke.jsonl   最早的 8 条 smoke 查询(旧索引)
pre_llamaindex/mixed_qrels.jsonl    Red Hat + NGINX 混合 qrels(旧索引)
```

**注意: `pre_llamaindex/` 中的 point_id 属于旧分块索引。**
LlamaIndex 全量重建后, 这些 qrels 不能直接用于当前 354 chunk 索引,
只能作为迁移 evidence spans 的参考。

## 使用方式

运行 RRF / rerank 对比:

```powershell
python rerank_compare_demo.py --qrels eval/pre_llamaindex/mixed_qrels.jsonl
```

输出会包含:

```text
Recall@5
Recall@10
MRR@20
每条 query 的 RRF / rerank 排名变化
```

## 注意

```text
smoke 集用于开发阶段快速发现问题
不要用 smoke 集证明最终效果
调参和最终评估应分开使用 dev / test
```
