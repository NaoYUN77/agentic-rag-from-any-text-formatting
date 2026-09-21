# 历史 qrels（已失效）

这里的文件属于**旧 LlamaIndex 链路**，不能用于当前索引。

```text
mixed_qrels.jsonl    13 条
rerank_smoke.jsonl    8 条
```

## 为什么失效

它们的 `gold` 是**裸 point_id**：

```json
{"query_id":"r01","query":"什么是 keepalived","gold":[0,2]}
```

而 `point_id` 的推导是：

```python
point_id = blake2b(chunk_id.encode()) & (2**63 - 1)
```

`chunk_id` 里含位置序号（`doc_XXX_c0001`），所以换 parser、改 cleaner、
改分块粒度之后编号全变 —— **标注会指向错误的内容**。

## 还能怎么用

只能作为**改写 query 意图的参考**。例如 `"什么是 keepalived"` 这个问法
可以复用到新语料上，但 `gold` 必须重新标注（锚在原文文本上）。

新 qrels 一律放 `../qrels/`，格式见 `../README.md`。

## 附带问题

`rerank_smoke.jsonl` 第 8 行有编码损坏（`"NAT è·`），已不再修复 ——
这份文件本身就是废弃状态。
