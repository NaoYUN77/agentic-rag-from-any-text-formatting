# Phase 0 语料扩充与功能闭环验证

更新时间：2026-09-17

## 1. 本次目标

不做正式评估（qrels / Recall@k / nDCG），先把 Phase 0 的**整体功能闭环**跑通并验证可用性。

```text
扩充语料
-> Phase 0 摄取
-> 统一重建 dense/sparse 索引
-> 在线服务指向 Phase 0 collection
-> 验证 dense / sparse / hybrid / rerank / generation
```

OCR 链路本次不涉及。

## 2. 语料扩充

从 Anthropic Engineering 和 OpenAI News 抓取公开技术文章，覆盖两个不同站点的页面结构。

| 目录 | 标题 | parser | blocks | chunks | quality |
|---|---|---|---|---|---|
| `experiments/anthropic_contextual_retrieval` | Contextual Retrieval in AI Systems | readability_structured | 78 | 17 | high 0.9875 |
| `experiments/anthropic_building_effective_agents` | Building effective agents | readability_structured | — | 20 | high |
| `experiments/anthropic_context_engineering` | Effective context engineering for AI agents | readability_structured | 64 | 12 | high 1.0 |
| `experiments/anthropic_multi_agent_research` | How we built our multi-agent research system | readability_structured | 47 | 12 | high 1.0 |
| `experiments/anthropic_agent_skills` | Equipping agents for the real world with Agent Skills | readability_structured | 44 | 9 | high 1.0 |
| `experiments/anthropic_code_execution_mcp` | Code execution with MCP | readability_structured | 63 | 14 | high 1.0 |
| `experiments/anthropic_demystifying_evals` | Demystifying evals for AI agents | readability_structured | 127 | 23 | high 1.0 |
| `experiments/openai_scaling_storage` | Rapidly scaling online storage to serve 1B ChatGPT users | trafilatura | 74 | 11 | high 0.9959 |
| `experiments/openai_agents_api` | Introducing the Agents API | trafilatura | 24 | 5 | high 1.0 |
| `experiments/openai_model_misalignment` | Our framework for reporting model misalignment | trafilatura | 24 | 6 | high 1.0 |

观察结论：

```text
Anthropic 文章 -> readability_structured 胜出
OpenAI   文章 -> trafilatura 胜出
```

两个站点的 HTML 结构差异确实触发了 HTML Parser 内部的 heading 完整性比较逻辑，
说明该分支在真实多样页面上是生效的，不是只在单元测试里可用。

### 去重处理

初次抓取时重复摄取了 Contextual Retrieval 一文（已存在 `anthropic_contextual_retrieval`），
产生了 `anthropic_contextual_retrieval_doc` 目录。

经比对：

```text
chunk 文本      : 完全一致
artifact_id     : doc_5c3e74c436ef  vs  doc_b267fb766937
chunk_id        : 不同
```

同一篇文章因重新抓取而得到不同 `artifact_id`，会在索引中形成重复内容。
已删除该重复目录。

**注意**：`artifact_id` 基于来源计算，重新摄取同一文档不保证得到相同 id，
增量索引前需要显式的去重或来源校验机制。

## 3. 统一索引重建结果

```text
export dirs          : 10
chunks               : 129
dense points         : 129
sparse points        : 123
parents              : 17
sparse vocabulary    : 4267
```

校验结果：

```text
phase0_dense points_count  = 129
phase0_sparse points_count = 123
sparse point id ⊆ dense point id  : True
仅进入 dense 的 chunk            : 6
```

6 个 chunk 无可用 sparse 文本，只进入 dense，与设计一致。

### 重建期的一个警告

重建过程中 `safe-delete` 对旧 collection 目录清理失败：

```text
[safe-delete][SAFE_DELETE_FAIL_CLOSED] target=...phase0_dense reason=trash-failed
[safe-delete][SAFE_DELETE_FAIL_CLOSED] target=...phase0_sparse reason=trash-failed
```

但最终结果正确：collection 目录被重新创建，点数为 129 / 123，
没有残留旧的 37 点。判定为**进回收站失败但不影响结果**，
索引正确性未受影响，可以接受。

## 4. 功能闭环验证

验证脚本：`pipeline/verify_phase0_loop.py`

启动服务（先停掉占用 `qdrant_data` 的进程）：

```powershell
$env:RAG_DENSE_COLLECTION = "phase0_dense"
$env:RAG_SPARSE_COLLECTION = "phase0_sparse"
$env:RAG_SPARSE_ARTIFACT_DIR = "index_artifacts/phase0_combined"

.\.venv_rag\Scripts\python.exe -m uvicorn rag_server:app --host 127.0.0.1 --port 8000
```

运行验证：

```powershell
.\.venv_rag\Scripts\python.exe verify_phase0_loop.py
```

结果：

```text
结果: PASS — dense / sparse / hybrid / rerank / generation 全部可用
```

覆盖内容：

```text
5 条 query × dense / sparse / hybrid / rerank
score == 该通路原生分(dense_score / sparse_score / rrf_score)
hybrid 结果 ⊆ 单通道候选池
rerank applied=True 且 score == rerank_score
rerank 未破坏 hybrid top1 召回
generation 返回非空 answer 且带 citations
```

## 5. 验证过程中踩到的两个坑

### 5.1 rerank 默认为 True，会覆盖 score 语义

`SearchRequest.rerank` 默认值为 `true`。

因此只写 `{"mode": "dense"}` 时，请求**实际上被 rerank 了**，
返回的 `score` 是 `rerank_score` 而不是 `dense_score`：

```text
mode=dense rerank=False -> score=0.790689 (== dense_score)
mode=dense rerank=True  -> score=0.663654 (== rerank_score)
```

这**不是 bug**，是 `_to_hit` 的既定优先级：`rerank_score > rrf_score > 通道分`。
但验证单通道分数时必须显式传 `"rerank": false`，否则会误判。
验证脚本已加入该断言。

### 5.2 RRF 在 candidate_k 深度上融合，hybrid 结果可以来自单通道 top-k 之外

最初写了「hybrid 的 chunk_id 必须出现在 dense 或 sparse 的 top-5 里」，
结果 Q2 / Q3 报错。

实际原因：

```text
doc_38a690a6ba57_c0005: dense rank=11, sparse rank=9
-> 在 hybrid 中排到 #4
```

`rrf_fuse` 融合的是 `candidate_k=20` 的完整候选列表，而不是展示用的 `top_k=5`。
所以 hybrid 结果**本来就应该**能来自单通道 top-5 之外——这正是 RRF 的价值。

正确的断言是：

```text
hybrid 的每个 chunk_id 至少出现在某个通道的 candidate 池内
```

断言已修正。这是验证逻辑的问题，不是检索实现的问题。

## 6. 一个观察（非缺陷）

`sparse unknown_terms` 出现过 `['dense']`。

排查结果：

```text
analyze('dense') = ['dense']        # 分词器正常
全部 123 个 chunk 中含 'dense' 的 : 0 个
全部 123 个 chunk 中含 'sparse' 的: 1 个
```

即 `dense` 这个词**本来就没出现在任何 chunk 的 `sparse_text` 里**
（当前语料讨论的是 sparse 检索，但正文没有用 "dense" 这个词）。
`unknown_terms` 的语义就是「query 里、但词表中没有的词」，行为正确。

## 7. 当前状态

已可用：

```text
多格式摄取 -> Block -> Chunk -> dense/sparse -> Qdrant
-> dense / sparse / hybrid(RRF) / rerank / generation + citations
-> FastAPI + Web 页面
```

语料规模：

```text
10 篇文档 / 129 chunks / 17 parents
```

仍缺失（本次不在范围）：

```text
正式评估集与 qrels
增量索引(当前只能整体 rebuild)
Parent expansion
字段化 BM25F
OCR / 多模态 / 动态网页 / Office / Docling
```

## 8. 环境注意事项

```text
1. 本机 HTTP_PROXY 会拦截 127.0.0.1 请求并返回 502,
   脚本调用本机服务必须绕过代理或设置 NO_PROXY=127.0.0.1,localhost。
2. 运行 qdrant 索引器前必须停止 FastAPI, 否则 qdrant_data 文件锁。
3. 后台启动的服务在 shell 会话结束后可能被回收, 需要确认进程存活后再验证。
4. git 在 E:\Code\python_dev_agent_demo 下报 dubious ownership,
   可用 git -c safe.directory=E:/Code/python_dev_agent_demo。
```

## 9. 相关文件

```text
pipeline/verify_phase0_loop.py    功能闭环可用性验证脚本(新增)
index_artifacts/phase0_combined/  统一索引产物
qdrant_data/collection/phase0_dense, phase0_sparse
experiments/anthropic_*, openai_* 语料摄取结果
```
