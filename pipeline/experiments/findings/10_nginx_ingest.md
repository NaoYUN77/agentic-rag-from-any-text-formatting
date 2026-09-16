# 实验 10: 增量加入 NGINX PDF 语料

## 1. 实验条件

```text
日期            2026-09-15
PDF             200 页
解析方式        MinerU flash-extract
页段            10 个 20 页区间
Markdown        169995 字
分块参数        max_chars=800, window_chars=400
新增 chunk      275
原有 chunk      22
总 chunk        297
dense points    297
sparse points   297
脚本            pipeline/ingest_markdown_corpus.py
sidecar         pipeline/index_artifacts/
验证输出        pipeline/experiments/20_nginx_ingest_verify.txt
统计输出        pipeline/experiments/21_nginx_ingest_stats.txt
```

## 2. 解析与入库结果

```text
NGINX chunks    275
字符 min/avg/max 39 / 571.2 / 1504

Red Hat chunks  22
字符 min/avg/max 208 / 600.4 / 800
```

检索验证:

```text
NGINX 查询      命中 nginx_guide.pdf
Red Hat 查询    redhat_p1-20.md 仍能被召回
```

说明增量入库没有覆盖原有语料。

## 3. 工程结论

### 3.1 增量入库可行

```text
保留旧 dense points
只嵌入新文档 chunk
重新计算全语料 BM25 稀疏统计
统一更新 redhat_sparse
```

比从头重新嵌入所有语料更省计算。

### 3.2 PDF 解析质量仍会影响 metadata

本次发现:

```text
代码块中的 name1:password1 等行被误判为 section
部分 section 路径包含大段配置代码
```

正文检索仍可用, 但按 section 过滤和引用展示会变差。

### 3.3 chunk 长度需要继续约束

`max_chars=800` 是目标上限, 不是绝对上限:

```text
单个超长句子没有合适切点时, 块可能超过 800
NGINX 最大块达到 1504 字
```

后续需要 min/max 双向约束和超长单句处理。

## 4. 当前边界

```text
1. MinerU flash-extract 每段只有 20 页, 本次通过 10 段拼接完成。
2. 图片、表格和复杂代码格式没有做精细保真校验。
3. 新语料还没有完整人工 qrels。
4. section metadata 噪声还没有修复。
5. 增量节点还没有建立版本化 manifest 和回滚机制。
```

## 5. 下一步

```text
1. 建立多文档 corpus manifest。
2. 对 PDF 代码块和标题做二次解析。
3. 增加 chunk 最大长度硬约束。
4. 为 NGINX 语料补 qrels, 重新评估 hybrid 和 rerank。
5. 把 collection 重命名为通用 RAG corpus, 或在 metadata 中显式区分语料来源。
```
