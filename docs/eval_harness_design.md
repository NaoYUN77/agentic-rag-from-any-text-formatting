# 评估闭环设计: qrels + 分阶段指标

> 目标：让「改了有没有用」变成一个可以回答的问题。
>
> 状态：harness 已实现（`pipeline/eval/`，2026-09-18）；qrels 已扩到 39 条 draft
> （2026-09-19，见文末执行记录）。真实检索指标仍缺 `DASHSCOPE_API_KEY`。
> 文中其余部分保留设计稿原貌，实测数字均已在当前索引上复现。

---

## 1. 为什么现在必须做

项目里已经积累了一批**无法验证收益**的待办：

```text
语义边界微调（issues/09）        —— 不知道该不该实现
Parent expansion                —— 不知道展开后有没有帮助
Contextual Retrieval            —— Anthropic 报告 49%/67%，但那是英文同语言条件
candidate_k / rrf_k 调参         —— 当前是拍的
```

没有评估集，这些问题只能靠"感觉变好了"来回答，而这正是
`docs/README.md` 约定第 3 条要防的事：

```text
代理指标: 切点距离 / 块大小分布 / 嵌入成本   （便宜，随时可算）
真实指标: recall@k / MRR / Precision@k      （需要评估集）
```

**当前评估集是空的（对新链路而言）**：

```text
eval/pre_llamaindex/rerank_smoke.jsonl    8 条   ← gold 是裸 point_id，已失效
eval/pre_llamaindex/mixed_qrels.jsonl    13 条   ← 同上
```

这两份的 `gold: [0, 2]` 是**旧 LlamaIndex 索引的 point_id**。
`point_id = blake2b(chunk_id) & (2^63-1)`，而 `chunk_id` 含位置序号，
换 parser / 改 cleaner / 换分块粒度 → 编号全变 → 标注指向错误内容。

**结论：不能迁移这两份 qrels，必须新建。但可以复用它们的 query 意图。**

---

## 2. 三个必须先解决的设计问题

这三个问题都是实测出来的，不解决的话 harness 会**静默给出错误结论**。

### 2.1 gold 锚在文本上，不锚在任何 ID 上

ID 体系的不稳定性：

```text
artifact_id  doc_5c3e74c436ef      uuid4 生成，重摄取即变
block_id     b0001                 文档内位置序号，换 parser 即错位
chunk_id     doc_XXX_c0001         含位置序号，改分块粒度即失义
point_id     1850931163052542526   chunk_id 的纯函数，随之而变
```

**四层 ID 没有一层是稳定的。** 所以 gold 必须锚在**原文文本**上：

```json
{"source_uri": "https://www.anthropic.com/engineering/contextual-retrieval",
 "block_id": "b0012",
 "span": [0, 273],
 "text": "Contextual Retrieval uses two sub-techniques: ..."}
```

评估时用 `text` 动态映射到**当前索引**里的 chunk。这样即使重建索引、
换 parser、改分块参数，标注依然有效。

### 2.2 `block_id` 跨文档会碰撞 —— 必须用复合键

**这是实测发现的最容易踩的坑。**

`block_id` 是**文档内**的位置序号（`b{order:04d}`），每篇文档都从 `b0001` 开始。
实测当前 10 篇语料：

```text
只用 block_id              →   127 个不同的键
用 (artifact_id, block_id) →   638 个不同的键

发生碰撞的 block_id: 99 个
  b0002 出现在 10 篇文档
  b0003 出现在 10 篇文档
  b0006 出现在 10 篇文档
  ...
```

**后果**：任何以 `block_id` 为键的索引都会把不同文档的块混在一起，
gold 反查会命中错误的 chunk，而且**不会报错**，只会给出错误的指标。

**规则：所有以 block_id 为键的结构，一律用 `(artifact_id, block_id)` 复合键。**
更稳的是用 `(source_uri, block_id)`，因为 `source_uri` 不随重摄取变化。

### 2.3 一个 gold 片段会对应多个 chunk —— gold 必须是集合

`overlap_tokens=400` / `chunk_tokens=800` 会让同一段文字出现在相邻 chunk 中。
用复合键实测（**注意：必须用 `(artifact_id, block_id)`，用裸 `block_id` 会得到错误结论**）：

```text
每个 block 被切进几个 chunk:
   1 个 chunk: 560 个 block
   2 个 chunk:  68 个 block
   3 个 chunk:  10 个 block      ← 真实最大值就是 3

对比: 若误用裸 block_id, 会算出"最多 14 个 chunk"
      那 99 个跨文档碰撞的 block_id 把不同文档混在一起了
```

**后果**：如果 gold 只记一个 chunk_id，那么检索命中它的"邻居 chunk"
（内容包含同一段文字）会被判为**未命中**，指标被系统性低估。

**规则**：gold 是 `set[chunk_id]`。指标同时报两个口径：

```text
hit@k     至少命中一个 gold chunk        （宽松，看"有没有找到"）
recall@k  命中数 / gold 总数             （严格，看"找全没有"）
```

**实际严重程度比预想的小**：636 个唯一片段中 570 个只出现在 1 个 chunk，
重复存储倍数仅 **1.11x**。所以 gold 多对多是**存在但有限**的问题，
仍然必须按集合处理，但不必为它设计复杂的加权逻辑。

### 2.4 附带发现：image fragment 是空文本

实测 729 个 fragment 中有 24 个 `text=''`、`span=[0,0)`，全是 `image` 块
（图片只有 URI，没有 OCR 文字）。这些 fragment 对文本反查无用，
映射时应跳过 `block_type == "image"`。

---

## 3. qrels 新格式

### 3.1 Schema

```jsonc
{
  "query_id": "cr_01",                    // 唯一编号，前缀标识文档
  "query": "Contextual Retrieval 降低了多少检索失败率？",
  "query_lang": "zh",                     // zh / en，用于分组统计

  // ---- gold：锚在文本上 ----
  "gold": [
    {
      "source_uri": "https://www.anthropic.com/engineering/contextual-retrieval",
      "block_id": "b0012",                // 辅助定位（可能失效）
      "span": [0, 273],                   // 辅助定位
      "text": "Contextual Retrieval uses two sub-techniques..."   // 主锚点，前 80~120 字符
    }
  ],

  // ---- 元信息 ----
  "category": "semantic",                 // semantic / exact / mixed / multi_hop / unanswerable
  "split": "dev",                         // dev / test（smoke 不用于出结论）
  "notes": "答案在正文第 3 段，含 49% / 67% 两个数字",
  "annotated_by": "human",
  "annotated_at": "2026-09-18"
}
```

### 3.2 字段设计说明

```text
text            主锚点。评估时用它在当前索引里反查 chunk。
                长度建议 80~120 字符：太短会多命中，太长会被 overlap 截断。
block_id/span   辅助定位。反查失败时用于人工排查，不参与自动映射。
source_uri      文档标识。比 artifact_id 稳定（artifact_id 是 uuid4）。
category        分类，用于分维度看指标。
split           dev/test 隔离。调参只看 dev，结论只报 test。
```

### 3.3 category 的判据

```text
semantic       语义相近但字面不同（考 embedding）
exact          含标识符/参数名/专有名词（考 BM25，如 "k1" "b=0.75"）
mixed          两者都有（考 RRF 融合是否真的有用）
multi_hop      需要跨段/跨文档才能回答（考 parent expansion）
unanswerable   语料里没有答案（考系统会不会硬答）
```

`unanswerable` 是**必须有的**。没有它，就无法发现"检索总是返回 top5、
哪怕全是噪声"这种失效模式。

---

## 4. 组件设计

新增目录 `pipeline/eval/`，五个文件，职责单一：

```text
pipeline/eval/
├── qrels.py          qrels 读写 + schema 校验
├── resolve.py        gold 文本 -> 当前索引的 chunk_id 集合   ← 核心
├── metrics.py        Recall@k / MRR / nDCG / hit@k
├── run_eval.py       主入口：跑分通道、出报告
└── README.md         用法与口径说明
```

### 4.1 `resolve.py` —— 最关键的一个文件

```python
"""把 gold 的文本锚点解析成当前索引里的 chunk_id 集合。

设计要点：
  1. 用 (artifact_id, block_id) 复合键，绝不用裸 block_id
  2. 跳过 image 类型的空 fragment
  3. 返回集合（overlap 导致一个片段出现在多个 chunk）
"""

def build_fragment_index(chunks_jsonl: Path) -> dict:
    """构建 (artifact_id, block_id) -> [fragment 记录] 的索引。"""
    index = defaultdict(list)
    for rec in iter_jsonl(chunks_jsonl):
        m = rec["metadata"]
        aid = m["artifact_id"]
        for fr in m.get("fragments") or []:
            if fr.get("block_type") == "image":     # 空文本，跳过
                continue
            if not fr.get("text"):
                continue
            index[(aid, fr["block_id"])].append({
                "chunk_id": m["chunk_id"],
                "point_id": rec["point_id"],
                "start": fr["start_char"],
                "end": fr["end_char"],
                "text": fr["text"],
            })
    return index


def resolve_gold(gold: dict, index: dict, doc_key: dict) -> set[str]:
    """把一个 gold 条目解析成 chunk_id 集合。

    优先用 text 子串匹配（对 block_id 失效免疫）；
    block_id 只作为加速和校验。
    """
    text = gold["text"]
    hits = set()

    # 主路径：全库文本反查
    for (aid, bid), frags in index.items():
        for fr in frags:
            if text in fr["text"]:
                hits.add(fr["chunk_id"])

    # 校验：如果给了 block_id，检查是否落在同一篇文档
    if not hits and gold.get("block_id"):
        raise GoldResolveError("text 反查无结果，标注可能已失效: %s" % text[:50])

    return hits
```

**必须有的自检**：解析后如果 `hits` 为空，**直接报错而不是静默跳过**。
静默跳过会让 recall 虚高（因为分母变小了）。

### 4.2 `metrics.py`

```python
def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """命中数 / gold 总数。gold 为空时返回 None（unanswerable 不算 recall）。"""
    if not gold:
        return None
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """top-k 里是否至少有一个 gold。"""
    if not gold:
        return None
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """第一个 gold 的倒数排名，没命中记 0。"""
    if not gold:
        return None
    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """二值相关性 nDCG。"""
    if not gold:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], 1)
        if cid in gold
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), k) + 1)
    )
    return dcg / ideal if ideal else 0.0
```

**注意 `gold` 为空时的处理**：`unanswerable` 类别的 query 没有 gold，
不能参与 recall/mrr 计算（分母为 0）。它们单独用另一个指标衡量：

```text
abstain_rate  = top1 分数低于阈值时"拒答"的比例
noise_ratio   = top-k 里与 query 无关的 chunk 比例（人工抽检）
```

### 4.3 `run_eval.py` —— 分阶段跑

**关键：一次检索，多阶段复用。** 不要为每个指标重跑一遍检索。

```python
def evaluate_one(retriever, reranker, qrels_item, config):
    q = qrels_item["query"]
    gold = resolve_gold(...)          # 集合

    # ---- 一次拿到所有通道的候选 ----
    dense  = retriever.dense_search(q, config.candidate_k)
    sparse, _, unknown = retriever.sparse_search(q, config.candidate_k)
    fused  = HybridRetriever.rrf_fuse([dense, sparse],
                                      k=config.rrf_k,
                                      limit=config.candidate_k)
    reranked = reranker.rerank(q, fused, top_k=config.final_top_k) if config.rerank else []

    # ---- 五个阶段各自算指标 ----
    stages = {
        "dense":    [h.payload["chunk_id"] for h in dense],
        "sparse":   [h.payload["chunk_id"] for h in sparse],
        "union":    _union_order(dense, sparse),      # 按 RRF 顺序但不过滤
        "rrf":      [h.payload["chunk_id"] for h in fused],
        "rerank":   [h.payload["chunk_id"] for h in reranked],
    }

    return {name: {
        "recall@5":  recall_at_k(ids, gold, 5),
        "recall@20": recall_at_k(ids, gold, 20),
        "mrr@20":    mrr_at_k(ids, gold, 20),
        "ndcg@5":    ndcg_at_k(ids, gold, 5),
        "hit@5":     hit_at_k(ids, gold, 5),
    } for name, ids in stages.items()}
```

### 4.4 报告产物

```text
eval/runs/<timestamp>/
├── config.json        本次运行的全部参数（可复现）
├── manifest.json      索引快照（chunks / parents / vocab / 产物 hash）
├── per_query.jsonl    每条 query 的五阶段指标 + 排名变化
├── summary.json       按 stage / category / split 聚合
└── report.md          人读的结论
```

`manifest.json` 是**必须的**。索引一变，历史指标就不可比。
记录：

```json
{"chunks": 129, "parents": 16, "dense_points": 129, "sparse_points": 123,
 "vocab": 4267, "chunks_jsonl_sha256": "...", "built_at": "..."}
```

---

## 5. 执行计划

分四步，每步都能独立验收。

### Stage 0：qrels 起草 + 映射器（1~2 天）

```text
[ ] 写 resolve.py + metrics.py，用 3 条手写 gold 做单元测试
[ ] 起草 30~40 条 query，覆盖 5 个 category
[ ] 用 resolve.py 反查，人工核对每条 gold 是否指向正确内容
[ ] 冻结 dev(20 条) / test(15 条) 划分
```

**验收标准**：`resolve_gold` 对全部条目返回非空集合，
且人工抽查 10 条，gold chunk 内容确实包含答案。

**这一步的产出是"评估集"，不是"代码"。不要急着跑指标。**

### Stage 1：单通道基线（半天）

```text
[ ] 跑 dense / sparse / union 三个阶段
[ ] 输出按 category 分组的 recall@5 / recall@20 / MRR
```

**这一步会暴露最重要的信息**：哪一路在什么类别上有用。

预期（待验证）：

```text
exact 类        sparse 应显著优于 dense
semantic 类     dense 应显著优于 sparse
中文 query      sparse 应接近 0（已知限制，词表全英文）
```

**如果预期没被验证，先怀疑 harness，不要急着改检索。**

### Stage 2：融合与重排（半天）

```text
[ ] 跑 rrf / rerank 两阶段
[ ] 对比 rrf vs dense、rerank vs rrf 的增量
[ ] 记录每条 query 的排名变化（谁升了、谁降了）
```

**判据**：RRF 相对最佳单通道应有提升，否则说明两路没有互补性。

### Stage 3：参数扫描（1 天）

```text
[ ] candidate_k ∈ {10, 20, 50}
[ ] rrf_k ∈ {10, 30, 60}
[ ] final_top_k ∈ {3, 5, 10}
```

**必须只扫 dev，结论只报 test。** 在 test 上调参等于把 test 变成 dev。

---

## 6. 已知陷阱（都实测过）

```text
① block_id 跨文档碰撞
   99 个 block_id 在多篇文档中重复。用裸 block_id 做键 → 静默命中错误 chunk。
   → 一律用 (artifact_id, block_id) 或 (source_uri, block_id)

② artifact_id 不稳定
   同篇文章重摄取会得到不同 artifact_id（doc_5c3e74c436ef vs doc_b267fb766937）。
   → gold 用 source_uri 做文档标识

③ overlap 导致 gold 多对多
   一个 block 最多被切进 3 个 chunk（用复合键才算得对，见 2.3）。
   重复存储倍数 1.11x，代价很小但必须按集合处理。
   → gold 必须是集合，同时报 hit@k 和 recall@k

③b 用裸 block_id 统计会得到错误结论（本设计稿踩过一次）
   裸 block_id 算出"最多 14 个 chunk"，是 99 个跨文档碰撞造成的假象。
   → 任何涉及 block 的统计，键一律带文档标识

④ image fragment 是空文本
   24 个 fragment text='' span=[0,0)。参与反查会污染结果。
   → 跳过 block_type == "image"

⑤ 解析失败必须报错，不能静默跳过
   静默跳过会让 recall 虚高（分母变小）。
   → resolve 失败直接 raise

⑥ 索引变了指标不可比
   → 每次运行必须落 manifest.json

⑦ SearchRequest.rerank 默认 True，会覆盖 score 语义
   验证单通道分数必须显式 rerank=false，否则拿到的是 rerank_score。
   （_to_hit 优先级：rerank_score > rrf_score > 通道分）

⑧ RRF 在 candidate_k 深度融合，不在 top_k 上融合
   hybrid 结果可以来自单通道 top-5 之外（实测 dense rank 11 + sparse rank 9 → hybrid #4）。
   验证 hybrid 的正确断言是「⊆ 单通道候选池」，不是「⊆ 单通道 top-k」。

⑨ 运行前必须停掉 FastAPI
   Qdrant local 模式会锁 qdrant_data 目录。

⑩ 本机代理会拦 localhost
   HTTP_PROXY=http://127.0.0.1:17161。调本机服务需 NO_PROXY=127.0.0.1,localhost。
```

---

## 7. harness 自身的验证

**最容易犯的错是：harness 写错了，却拿它去指导优化决策。**

所以必须有自检：

```text
[ ] 单元测试: 用 3 条已知 gold 的 query, 断言 resolve_gold 返回预期 chunk 集合
[ ] 退化测试: 把 gold 换成"top1 的 chunk", recall@1 必须是 1.0
[ ] 空集测试: gold 为空的 unanswerable, recall 必须返回 None 而不是 0
[ ] 复现测试: 同一份 qrels + 同一份索引, 跑两次指标完全一致
[ ] 敏感性测试: 人为把 top1 和 top5 对调, nDCG 必须下降
[ ] 交叉校验: 抽 5 条 query, 人工看 top5, 确认与指标判断一致
```

**第 2 条（退化测试）最有价值** —— 它能一次性验证 resolve、检索、指标三处都没接错。

---

## 8. 做完之后能回答什么问题

这才是评估闭环的价值所在：

```text
· dense 和 sparse 各自在哪些类别上有用？RRF 真的带来增量吗？
· rerank 提升了多少？值不值这次 API 调用？
· candidate_k 从 20 提到 50，recall 提升多少？成本增加多少？
· Parent expansion 展开后，multi_hop 类别的 recall 提升多少？
· 中文 query 下 hybrid 退化为纯 dense，损失了多少？（已知限制的量化）
· Contextual Retrieval 在本项目语料上，能复现 Anthropic 的收益吗？
· issues/11 修好之后，section 相关的指标有没有变化？
· issues/09 的语义边界，值不值得实现？（用 dev 集 A/B）
```

---

## 9. 关联文档

- `eval/README.md` —— 旧 qrels 说明（已失效，保留作历史）
- `docs/project_overview_for_agents.md` 第 7.3 节、第 12 节
- `docs/block_structure_and_blockification.md` —— ID 体系与 fragments 不变式
- `issues/09_window_tokens_unused.md` —— 需要评估集才能决策的典型
- `issues/10_parent_grouping_degraded.md` —— Parent expansion 的前置障碍
- `pipeline/eval_zh_crosslingual.py` —— 现有实验脚本，检索器初始化可复用

---

## 10. Stage 0 执行记录（2026-09-19）

### qrels 扩写：13 → 39 条 ✅

工具：`pipeline/expand_phase0_qrels.py`（幂等，marker 唯一性强制校验，
gold 由脚本从 `experiments_rebuilt/<doc>/artifact.json` 原文逐字切片）。

```text
total 39   by_split  dev 21 / test 18   by_lang  zh 31 / en 8
by_category  exact 11 / semantic 14 / mixed 5 / multi_hop 4 / unanswerable 5
覆盖文档      10/10（ag/cx/oa/om 四篇此前为 0 条）
```

扩写过程挖出并修掉的问题：

1. **unanswerable 候选必须 grep 验证**。"部署在哪个云平台"看似无答案，
   实测语料有 8 处 Azure —— 换成了 rate limit / fine-tune 定价 / SSD 型号
   三个验证过的空白点。这就是 3.3 节"unanswerable 必须有"的日常操作形态。
2. **归一化缺两类规则**（OLD 语料 3 条 FAIL 的根因）：
   Markdown 链接 `[anchor](url)`、DOM 行内边界挤出的标点前空格（`Agents , on`）。
   已加入 `normalize_text`，`NormalizeTests` 补 2 个用例锁边界。
   修复后新旧语料双份 39/39 解析通过。
3. **`run_eval --chunks` 只收单文件，而瘦身 sidecar 无 fragment.text**。
   payload 瘦身（P0-2）移除了 `fragment.text`，`index_artifacts/*/chunks.jsonl`
   反查必然 0 命中。改为 nargs="+" + `build_fragment_index_multi`，
   指向原始 per-doc dump；chunk_id 与 sidecar 已验证 117/117 对齐。
4. **"gold 是集合"第一次被真实覆盖**：7 条 query 的 gold 跨 2~3 个 chunk
   （oa_01 跨 3 个，overlap 所致），严格口径 recall 的必要性从推断变成实测。

验证：harness dry-run 全链路 PASS（桩检索器指标完美，n=34 条可答 query），
单元测试 69/69。

### 当前卡点

```text
真实检索指标   需 DASHSCOPE_API_KEY（当前 .dashscope_key 是占位符）
               -> rebuild_phase0_index.py --backend qwen 重跑后
                  再跑 run_eval（不带 --dry-run）
qrels 正式化   39 条仍标 annotated_by: draft，需人工逐条确认
dev/test 冻结  划分已写进每条 split 字段，调参前建议导出 test 只读副本
```
