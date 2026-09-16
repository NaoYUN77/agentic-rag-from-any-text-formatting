# 实验 07: 检索增强生成(answer + citations)

## 1. 实验条件

```text
日期            2026-09-13
检索            Hybrid + RRF
dense 集合      redhat
sparse 集合     redhat_sparse
上下文数        5
上下文上限      6000 字
生成模型        qwen-plus
实现            pipeline/generation.py
接口            POST /api/answer
页面            pipeline/static/index.html
原始输出        pipeline/experiments/17_generation.txt
桌面截图        pipeline/experiments/17_generation_ui.png
移动截图        pipeline/experiments/17_generation_ui_mobile.png
```

## 2. 实测结果

```text
answer 长度       705 字
citations         5
context_chars     3045
retrieval_ms      3 (查询向量已缓存)
generation_ms     9868
elapsed_ms        9871
```

答案中产生了 `[C1]` 到 `[C5]` 的引用标记。引用对象来自同一批 hybrid
检索结果，保留了:

```text
point_id
file_name
page
section
chunk_index
```

## 3. 已完成的链路

```text
query
  -> dense + sparse
  -> RRF
  -> context builder
  -> qwen-plus
  -> answer + citations
```

页面支持两种任务:

```text
问答生成    POST /api/answer
仅检索      POST /api/search
```

## 4. 结论

### 4.1 chunk 已从最终输出降级为证据

用户先看到生成答案，下面再看到 RRF 候选和引用来源。chunk 不再直接充当
最终回答。

### 4.2 引用由上下文编号绑定

生成时每个 chunk 被分配 `[C1]`、`[C2]`，所以答案中的引用可以映射回
point_id、页码和章节。

### 4.3 生成延迟明显高于检索

本次检索约 `3 ms`，生成约 `9.9 s`。后续需要评估:

```text
缩短上下文
流式输出
更换更快的 chat 模型
缓存高频问题
```

## 5. 当前边界

```text
1. 只验证了一条查询, 没有建立答案质量评估集。
2. 引用编号表示“上下文来源”, 还没有做逐句引用一致性校验。
3. 生成模型可能总结错误或遗漏, 需要 faithfulness 评估。
4. 尚未接入 rerank, 也尚未做流式输出。
5. 当前 qwen-plus 的延迟约 10 秒, 不适合直接作为低延迟交互体验。
```

## 6. 下一步

```text
1. 建立问题 -> 标准答案 -> 支持 chunk 的评估集。
2. 增加 faithfulness / answer relevance / citation accuracy 指标。
3. 接入 cross-encoder rerank, 只把更相关的 top-k 送入生成。
4. 加入流式输出和取消请求。
5. 评估 qwen-turbo / qwen-plus / 其他模型的延迟与质量取舍。
```
