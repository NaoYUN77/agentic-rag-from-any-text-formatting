# 实验 05: 复用 dense chunk 并写入 Qdrant sparse collection

## 1. 实验条件

```text
日期            2026-09-13
来源 collection redhat (现有 dense collection)
来源 chunk      22 个 point
来源 payload    text/file_name/page/section/chunk_index/char_count/n_sentences
目标 collection redhat_sparse
分析器          jieba + 技术 token 保护 + 停用词过滤
索引            BM25 文档稀疏权重
BM25            k1=1.2, b=0.75
脚本            pipeline/sparse_pipeline_demo.py
原始输出        pipeline/experiments/15_sparse_qdrant_pipeline.txt
验证输出        pipeline/experiments/15_sparse_qdrant_verify.txt
结构化产物      pipeline/experiments/15_sparse_qdrant_artifacts/
```

## 2. 实测数据

```text
复用 chunk 数       22
文本规模            13208 字
词表大小            838
平均 token/chunk    169.73
非零 sparse 权重    2120
稀疏密度            2120 / (22 * 838) = 0.114992
Qdrant sparse 点    22
```

## 3. 结论

### 3.1 dense 链路切好的 chunk 可以复用

sparse 不需要重新解析 PDF 或重新分块。当前脚本从现有 `redhat` collection
读取 `payload.text` 和 metadata, 直接构建 sparse index。

共用关系:

```text
同一个 chunk 原文
同一个 chunk_index
同一个 section/page
同一个 Qdrant point id
```

### 3.2 sparse 结果已经写入 Qdrant

当前采用独立 collection:

```text
redhat          dense vector + payload
redhat_sparse   sparse vector + payload
```

选择独立 collection 的原因:

```text
现有 redhat collection 只配置了 dense vector
本阶段先验证 sparse 读写, 不改动 dense collection
后续可迁移到同一个 collection 的 named vectors
```

### 3.3 Qdrant 可以真正执行 sparse 检索

从 `redhat_sparse` 直接使用 `SparseVector(indices, values)` 查询:

```text
rank 1  id=0  score=2.6534
rank 2  id=1  score=2.0408
rank 3  id=4  score=1.7973
rank 4  id=3  score=1.6713
rank 5  id=8  score=1.3305
```

结果与脚本内存中的 BM25 top-k 一致。

### 3.4 词表和 BM25 统计仍然是应用侧数据

Qdrant sparse vector 只保存:

```text
indices = term_id
values  = BM25 文档权重
```

应用侧还需要保存:

```text
vocab.json       term -> term_id
bm25_stats.json  idf / k1 / b / avgdl
```

如果这些映射丢失, Qdrant 中的整数 indices 无法重新解释成词项。

## 4. 当前边界

```text
1. dense 和 sparse 目前仍是两个 collection, 不是同一个 point 的两个命名向量。
2. 查询时仍由脚本重建内存 BM25 索引, 还没有实现“只读 Qdrant 就完成查询”。
3. IDF 和文档权重目前使用当前语料计算, 加新文档后需要重建或增量更新。
4. 还没有跑 dense + sparse 的 RRF 融合。
```

## 5. 下一步

```text
1. 让脚本支持从 Qdrant sparse collection 直接执行查询。
2. 把 vocab 和 BM25 stats 作为 collection 的 sidecar metadata 保存。
3. 评估是否迁移到同一个 collection 的 dense/sparse named vectors。
4. 接入 dense top-k 和 RRF 融合。
```
