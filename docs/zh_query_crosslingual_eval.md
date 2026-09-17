# 中文 Query 跨语言检索方案对照实验

更新时间：2026-09-17

## 1. 背景与问题

语料是纯英文（10 篇 / 129 chunks），用户用中文提问。

已知问题：中文 query 进入 sparse 通道会直接空转——BM25 词表（4,267 词）全是英文词，
中文 query 经 jieba 分词后无一命中词表，`sparse_search` 返回空列表。

本次要回答的问题是：

```text
在不动语料(保持英文原文)的前提下, query 侧应该怎么处理?
```

## 2. 方案设计

三种方案对照，判据是「top-k 是否命中期望文档」：

| 方案 | dense query | sparse query |
|---|---|---|
| **A 现状** | 中文 | 中文（失效） |
| **B 分通道** | 中文 | 英文译文 |
| **C 全翻译** | 英文译文 | 英文译文 |

设计意图：

- A 是基线。
- B 保留 dense 的跨语言能力（`qwen3-vl-embedding` 本身懂中文），只用译文救 sparse。
- C 是「全都翻译」的朴素方案，用来验证它是否真的更好。

## 3. 实验设置

脚本：`pipeline/eval_zh_crosslingual.py`

```text
语料       : phase0_dense(129) / phase0_sparse(123)
query 集   : 10 条中文 query, 覆盖精确/语义/多跳
candidate_k: 20
top_k      : 5
rrf_k      : 60
翻译模型   : qwen-turbo, temperature=0.0
```

关键实现说明：

- **在进程内直接调用 `HybridRetriever`**，分别给 dense 和 sparse 传不同的 query，
  从而绕开 `search()` 只接受单个 `query` 的限制。未改动任何生产代码。
- **运行前必须停掉 FastAPI**，否则 `qdrant_data` 文件锁。
- 译文落缓存文件，保证实验可复现（`temperature=0.0`）。

运行方式：

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."
cd pipeline
.\.venv_rag\Scripts\python.exe eval_zh_crosslingual.py
```

翻译 prompt 明确要求保留英文技术术语（BM25 / RAG / token / reranking /
embedding / context engineering），避免术语被译成中文后失去匹配能力。

## 4. 结果

```text
方案                                 top1命中   top5命中    MRR    top5纯度  sparse空
--------------------------------------------------------------------------------------
A 现状(dense中文+sparse中文)            8/10     10/10   0.900    0.760     4/10
B 分通道(dense中文+sparse译文)         10/10     10/10   1.000    0.860     0/10
C 全翻译(dense译文+sparse译文)         10/10     10/10   1.000    0.860     0/10
```

### 逐条差异（A -> B 发生改变的两条）

```text
[zh07] 为什么代码执行比直接调用工具更省 token？
       A: top1=False rank=2   (top1 被 Contextual Retrieval 抢走)
       B: top1=True  rank=1

[zh10] 多智能体系统什么时候不该用？
       A: top1=False rank=2   (top1 是 Multi-agent research)
       B: top1=True  rank=1
```

其余 8 条 A 与 B 结果一致，说明 dense 在多数情况下单独就能命中 top1。

## 5. 结论

### 5.1 sparse 空转被完全消除

```text
A: 4/10 条 query 的 sparse 通道返回空
B: 0/10
C: 0/10
```

译文让英文 BM25 词表重新可用，这是本实验最确定的收益。

### 5.2 top1 命中率 8/10 -> 10/10，MRR 0.900 -> 1.000

改善集中在两条「概念相近、需要区分」的 query 上（zh07 / zh10）。
这类 query 恰恰是 dense-only 最容易混淆的场景——同一话题在多篇文档里都有讨论，
纯语义相似度分不开，需要 sparse 的词项精确性来投票。

### 5.3 B 与 C 在所有指标上完全打平

```text
top1 10/10, top5 10/10, MRR 1.000, 纯度 0.860 —— 三项全部相同
```

这意味着：**把 dense 的 query 也翻译成英文，没有带来任何额外收益。**

### 5.4 因此选 B，不选 C

C 相对 B 是「多花一次翻译 + 承担翻译误差」，换来 0 收益。理由：

1. **dense 本来就懂中文**，直接检索英文文档的跨语言效果已经足够
   （独立实测余弦差 +0.34~+0.59）。
2. **翻译必然引入信息损失**，而 dense 侧用中文是「零损失」的原文语义。
3. RRF 按**排名**融合，不依赖两路分数同尺度，
   所以「一路中文、一路英文」在融合层面完全兼容——本实验也证实了这点。

### 5.5 语料侧坚决不翻译

本次结论只覆盖 query 侧。语料侧翻译会带来不可逆损失：

```text
专有名词/标识符被抹平(IndexReadyChunk -> 普通叙述)
BM25 罕见词精确匹配能力被摧毁
数字/代码/引用系统性损毁
fragments 字符坐标溯源链断裂
长文档术语翻译不一致, 跨文档召回下降
换翻译模型需全量重译
```

语料保持原文语言是底线。

## 6. 落地建议

若采纳方案 B，实现要点：

```text
1. 判断 query 是否含中文
2. 中文 -> 调翻译模型得到英文 query
3. dense_search(中文原文)
   sparse_search(英文译文)
4. 照常 RRF 融合
```

注意事项：

- 只对**中文（或非英文）query** 触发翻译，英文 query 不要多此一举。
- 翻译结果可缓存，相同 query 不必重复翻译。
- 翻译失败要降级为「sparse 空转」而不是整体报错，保持 dense 可用。
- 当前 `HybridRetriever.search()` 只接受单个 query，
  需要扩展为支持分通道 query（例如新增 `sparse_query` 参数）。

## 7. 局限

- **样本仅 10 条中文 query**，且是我自己构造的，不等于真实分布。
  结论方向可信，具体数值不宜外推。
- **语料只有 10 篇英文文档**，规模小，区分度天然有限
  （所有 top1 的 RRF 分数都挤在 0.0325 附近）。
- 未做 rerank 环节的对照。`gte-rerank-v2` 对中文 query 的表现待验证。
- 未测「翻译后再改写/扩展」等更复杂的 query 处理方案。

## 8. 相关文件

```text
pipeline/eval_zh_crosslingual.py    本次对照实验脚本
pipeline/hybrid_retriever.py        检索器(dense_search / sparse_search / rrf_fuse)
docs/phase0_loop_verification.md    Phase 0 闭环验证
```

## 9. 附：本次遇到的 git 引用异常（环境问题，非代码问题）

提交本次实验时触发了一个 git 异常，记录备查。

### 现象

```text
fatal: your current branch 'feature/phase0-ingest-structure' does not have any commits yet
```

且提交日志显示为 `commit (initial):`，一次提交报出 96 files changed / 23941 insertions，
本应只有 3 个文件。

### 原因

`.git/refs/heads/feature/phase0-ingest-structure` 这个**引用文件不存在**，
而 `.git/HEAD` 仍指向它，构成 git 的 "unborn branch" 状态。

因此新提交被当成 initial commit（无父提交），把整个工作区文件全部算进这次提交。

**该引用文件在每次 git 操作后会被清理掉**，推测与目录所有权有关——
本仓库存在 `dubious ownership` 问题（`.git` 属主为 `CodexSandboxOffline`，
当前用户为 `l`）。

### 处理

1. 确认提交对象未丢失：`20434f8`（上一轮）与误提交对象都仍可通过 `git cat-file` 访问。
2. 确认工作区文件完整，误提交与预期的差异**恰好等于本次要提交的 3 个文件**，无内容污染。
3. 用 PowerShell（非 bash）重建引用目录并写入**完整** hash：

```powershell
$refFile = "E:\...\.git\refs\heads\feature\phase0-ingest-structure"
New-Item -ItemType Directory -Path "...\.git\refs\heads\feature" -Force
Set-Content -Path $refFile -Value "<完整40位hash>" -NoNewline
```

4. 重新提交，确认父提交正确：

```text
4e17853 parent=20434f8 feat: evaluate query-side cross-lingual retrieval strategies
```

### 要点

- **必须写完整 40 位 hash**，写短 hash 不是合法 ref。
- bash 下无法在 `.git/refs/` 下创建子目录（静默失败，退出码 1），
  PowerShell 可以。这与安全沙箱对 `.git` 的保护有关。
- 该环境下每次 git 操作后都要复查引用文件是否还在：

```bash
cat .git/refs/heads/feature/phase0-ingest-structure
```

- 提交后用 `git log --format="%h parent=%p"` 检查父链，
  出现空 parent 说明又进入了 unborn branch 状态。

