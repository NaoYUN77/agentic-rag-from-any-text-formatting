# 实验 11: 使用 LlamaIndex 800/400 切分器全量重建

## 1. 实验条件

```text
日期            2026-09-15
备份            pipeline/backups/20260915_pre_llamaindex/
chunker          FixedSemanticNodeParser
chunk_size       800 token
chunk_overlap    400 token
window_size      400 token
语料             Red Hat + NGINX
重建脚本         pipeline/rebuild_llamaindex_corpus.py
原始输出         pipeline/experiments/25_llamaindex_rebuild.txt
失败样本         pipeline/experiments/24_llamaindex_rebuild.txt
```

## 2. 重建结果

```text
Red Hat chunks   42
NGINX chunks     312
总 chunks        354
dense points     354
sparse points    354
```

token 分布:

```text
Red Hat   378 ~ 763
NGINX      9 ~ 800
```

NGINX 320+ token 的超级大块已经被超长句 token 硬切兜底限制在 800。

## 3. sparse/BM25 如何处理

sparse 没有沿用旧 chunk 的统计。重建时执行了:

```text
新 chunk 文本
  -> jieba 分析
  -> 新词表
  -> 新 tf / df / N / avgdl / IDF
  -> 新 BM25 sparse vector
  -> redhat_sparse 重建
```

因此:

```text
dense point 数量 = sparse point 数量 = 354
point_id 一一对应
```

## 4. 验证

重建后的 NGINX 查询和 Red Hat 查询都能被召回:

```text
NGINX 查询      返回 nginx_guide.pdf
Red Hat 查询    RedHat chunk 仍能进入结果
```

服务启动后:

```text
redhat          354 points
redhat_sparse   354 points
```

## 5. 重要影响

### 5.1 旧 qrels 失效

重建改变了 chunk 边界、point_id 和 chunk 数量, 所以重建前的 qrels 不能继续用于当前索引。

旧 qrels 已移到:

```text
pipeline/eval/pre_llamaindex/
```

它们只保留作为 evidence span 迁移参考, 不能直接拿来算当前指标。

### 5.2 section metadata 仍有噪声

MinerU 生成的 NGINX 配置块仍可能被识别为伪标题, section 过滤和引用展示仍存在问题。

## 6. 下一步

```text
1. 基于新索引重新标注 qrels。
2. 对 LlamaIndex 500/1000 token 参数做对照。
3. 修复 PDF 代码块到 section 的传播。
4. 对比重建前后的 Recall@k / MRR。
```
