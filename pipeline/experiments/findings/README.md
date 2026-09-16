# 实验结论档案

本目录存放**每次都实验的结论**，一次实验一份，**写完不再修改**（保留历史）。

## 为什么单独建一层

```text
experiments/*.txt             原始运行输出(机器产生, 只增不改)
experiments/findings/*.md     实验结论(记录 前提 + 结论 + 适用边界)   ← 本目录
rag_chunking_concepts.md      当前认知(会随新实验迭代, 引用 findings 作为证据)
```

**关键区别**：`findings/` 记录"当时在什么条件下得出了什么"，`concepts.md` 记录"现在我们认为什么是对的"。

后者会变，前者不会 —— 所以新实验推翻旧结论时，能回头看清是在哪个条件下翻的。

## 每份结论的固定结构

```text
1. 实验条件     语料 / 模型 / 参数 / 原始数据文件
2. 结论         带数字的核心结论
3. 适用边界     这组结论能推广到哪、不能推广到哪
4. 待验证       还没搞清的部分
```

**"适用边界"这一节是重点** —— 很多结论只在一组参数下成立，不写清楚就会被误用。

## 清单

| 编号 | 主题 | 数据来源 | 状态 |
|---|---|---|---|
| 01 | 断点判定的四种统计方法 | `01_semantic_chunking_qwen.txt` | 已完成 |
| 02 | 边界窗口宽度的影响 | `05_window_sweep.txt` | 已完成 |
| 03 | 嵌入机制与成本 | `08_embedding_mechanism_compare.txt`<br>`09_embedding_config_compare.txt` | 已完成 |
| 04 | 独立 jieba + BM25 稀疏链路 | `14_sparse_pipeline.txt` | 已完成 |
| 05 | 复用 dense chunk 并写入 Qdrant sparse collection | `15_sparse_qdrant_pipeline.txt`<br>`15_sparse_qdrant_verify.txt` | 已完成 |
| 06 | Dense + Sparse + RRF 融合 | `16_hybrid_rrf.txt` | 已完成 |
| 07 | 检索增强生成(answer + citations) | `17_generation.txt` | 已完成 |
| 08 | 生成模型延迟对比 | `17_generation.txt`<br>`18_generation_turbo.txt` | 已完成 |
| 09 | RRF + Rerank smoke 对比 | `19_rerank_compare.txt` | 已完成 |
| 10 | 增量加入 NGINX PDF 语料 | `20_nginx_ingest_verify.txt`<br>`21_nginx_ingest_stats.txt` | 已完成 |
| 11 | LlamaIndex 800/400 全量重建 | `24_llamaindex_rebuild.txt`<br>`25_llamaindex_rebuild.txt` | 已完成 |
