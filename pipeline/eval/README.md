# 评估 harness

分阶段检索评估。回答的问题是：**"改了有没有用"**。

## 为什么 gold 锚在文本上

项目里有四层 ID，**没有一层稳定**：

```text
artifact_id  doc_5c3e74c436ef      uuid4 生成，重摄取即变
block_id     b0001                 文档内位置序号，换 parser 即错位
chunk_id     doc_XXX_c0001         含位置序号，改分块粒度即失义
point_id     1850931163052542526   chunk_id 的哈希，随之而变
```

所以 qrels 里存**原文片段**，评估时用文本反查当前索引。这样重建索引、
换 parser、**改分块策略**之后标注依然有效 —— 这是能跨切块策略对比的前提。

## 目录

```text
qrels.py      qrels schema + 读写 + 校验
resolve.py    gold 文本 → 当前索引的 chunk_id 集合   ← 核心
metrics.py    Recall@k / hit@k / MRR / nDCG
run_eval.py   主入口，分阶段跑并出报告
qrels/        qrels 数据
runs/         评估产物（含 report.md，已 gitignore）
legacy/       旧链路的 qrels，已失效，仅作历史参考（见 legacy/README.md）
```

## 用法

```powershell
cd pipeline

# 1) 先验证 harness 本身（不调 API，指标应完美）
.\.venv_rag\Scripts\python.exe -m eval.run_eval `
    --qrels eval/qrels/phase0_url_draft.jsonl `
    --chunks experiments_rebuilt\*/chunks.jsonl `
    --artifact-dir index_artifacts/phase0_rebuilt `
    --stages dense,sparse,union --dry-run

# 2) 真实评估（需要 DASHSCOPE_API_KEY；先停掉 FastAPI，否则 Qdrant 被锁）
$env:DASHSCOPE_API_KEY = "sk-..."
.\.venv_rag\Scripts\python.exe -m eval.run_eval `
    --qrels eval/qrels/phase0_url_draft.jsonl `
    --chunks experiments_rebuilt\*/chunks.jsonl `
    --artifact-dir index_artifacts/phase0_rebuilt `
    --dense-collection phase0_dense `
    --sparse-collection phase0_sparse `
    --out-dir eval/runs
```

**`--chunks` 必须指向原始 per-doc dump（`experiments_rebuilt/<doc>/chunks.jsonl`），
不能指向 `index_artifacts/` 下的瘦身 sidecar。** payload 瘦身（2026-09-18）把
`fragment.text` 从索引产物里移除了，sidecar 的 644 个 fragment 全是空文本，
文本反查会得到 0 命中。两者 chunk_id 已验证完全对齐（117/117），解析出的
gold chunk_id 就是检索器返回的 ID。

常用参数：

```text
--stages dense,sparse,union,rrf,rerank   选阶段（rerank 要调 API，最慢）
--no-rerank                              跳过重排
--candidate-k / --final-top-k / --rrf-k  调参
--dry-run                                桩检索器，只验 harness 本身
```

## qrels 格式

每行一条 JSON：

```json
{
  "query_id": "cr_01",
  "query": "Contextual Retrieval 把检索失败率降低了多少？",
  "query_lang": "zh",
  "gold": [
    {
      "text": "This method can reduce the number of failed retrievals by 49% and, when combined with reranking, by 67%.",
      "block_id": "b0003"
    }
  ],
  "category": "exact",
  "split": "dev",
  "notes": "答案含 49% 和 67% 两个数字",
  "annotated_by": "draft",
  "annotated_at": "2026-09-18"
}
```

```text
text            主锚点。评估时用它反查 chunk。建议 60~200 字符 ——
                太短会多命中，太长容易被 overlap 截断。
block_id        辅助定位，只用于人工排查，不参与自动映射。
source_uri      文档标识。比 artifact_id 稳定（后者是 uuid4）。
category        semantic / exact / mixed / multi_hop / unanswerable
split           dev / test。调参只看 dev，结论只报 test。
annotated_by    标 draft 表示尚未人工确认，不可用于出结论。
```

**`unanswerable` 是必须有的** —— 没有它，就发现不了"不管问什么都返回
top5、全是噪声"这种失效模式。它没有 gold，指标返回 `None` 而不是 0
（不能进分母）。

## 指标口径

gold 是 **chunk_id 的集合**而不是单个 ID —— overlap 会让同一段原文进多个
chunk（实测一个 block 最多进 3 个）。若 gold 只记一个，命中它的"邻居
chunk"会被判未命中，指标被系统性低估。

所以每个 query 同时报两个口径：

```text
hit@k      至少命中一个 gold        —— 宽松，看"有没有找到"
recall@k   命中数 / gold 总数       —— 严格，看"找全没有"
MRR@k      第一个 gold 的倒数排名
nDCG@k     二值相关性
```

## 分阶段的意义

```text
dense   纯语义召回     —— 看 embedding 的能力上限
sparse  纯字面召回     —— 看 BM25 在精确匹配上的优势
union   两路并集       —— 看"两路合起来覆盖了多少"，是上限
rrf     融合后         —— 看融合是否真的带来增量（对比 union）
rerank  重排后         —— 看重排是否真的把对的提上来
```

**一次检索，五阶段复用** —— 不为每个指标重跑检索。

## 产物

```text
runs/<时间戳>/
├── report.md       人读的结论（分阶段表 + 分类别表 + 逐条排名）
├── summary.json    机读的聚合指标
├── per_query.jsonl 逐条明细
├── config.json     本次参数（可复现）
└── manifest.json   索引快照（chunks_sha256 / points 数）
```

**`manifest.json` 是必须的** —— 索引一变历史指标就不可比。

## 三条硬规则

```text
① 键必须带文档标识
   block_id 是文档内位置序号，每篇文档都从 b0001 开始。
   实测 10 篇语料里 99 个 block_id 跨文档碰撞 —— 只用 block_id 建索引
   会把不同文档的块混在一起，gold 反查命中错误 chunk 且【不报错】。
   → 一律用 (artifact_id, block_id)。

② 跳过 image 的空 fragment
   实测 729 个 fragment 里 24 个 text='' span=[0,0)，全是 image 块。

③ 解析失败必须抛错，不能静默跳过
   静默跳过会让 recall 虚高（分母变小），是最隐蔽的错误来源。
```

## harness 自检

`tests/test_eval_harness.py` 有 29 项自检。**最容易犯的错是 harness 自己
写错了，却拿它去指导优化决策**，所以重点验证 harness 本身。

其中【退化测试】最有价值：把 gold 换成检索结果 top1，`recall@1` 必须是
`1.0` —— 这一条能一次性验证 resolve、检索、指标三处都没接错。

`--dry-run` 是同一思路的整链路版本：它跑真实的 `run_eval` 代码路径，
只是把检索器换成"直接返回 gold"的桩，因此指标必须完美。

## 已知局限

```text
[ ] 当前 qrels 是 draft（39 条），需人工确认后才能用于出结论
    2026-09-19 从 13 条扩到 39 条: dev 21 / test 18，zh 31 / en 8，
    5 个 category 全覆盖（exact 11 / semantic 14 / mixed 5 / multi_hop 4 /
    unanswerable 5），10 篇语料全部有 query。扩写用
    expand_phase0_qrels.py 生成 —— gold 由脚本从 artifact 原文逐字切片，
    marker 唯一性强制校验，避免手抄错标点/引号。
[x] "反查命中多个 chunk"已验证（2026-09-19）: 39 条里 7 条 gold 跨 2~3 个
    chunk（oa_01 跨 3 个），gold 集合语义被真实覆盖
[x] 新旧语料双份解析 39/39（check_qrels_resolution.py）—— 归一化新增
    链接语法剥离 + 标点前空格折叠后，gold 跨 parser 存活
[ ] 只有 URL 语料的 qrels，PDF 语料还没有
[ ] 单句策略的 qrels 尚未验证（同一份 qrels 理论可评，但 gold 粒度需复核）
[ ] 没有 generation 侧指标（faithfulness / 引用准确率）
[ ] dev/test 尚未"冻结": 当前划分写在每条 split 字段里（dev 21 / test 18），
    但没有单独的冻结清单文件；调参前应把 test 集导出为只读副本
```

---

## 历史：legacy/（已失效）

```text
legacy/mixed_qrels.jsonl     13 条
legacy/rerank_smoke.jsonl     8 条
```

这两份的 `gold` 是**裸 point_id**（如 `"gold": [0, 2]`），属于旧
LlamaIndex 索引。`point_id = blake2b(chunk_id) & (2^63-1)`，而
`chunk_id` 含位置序号 —— 换 parser / 改 cleaner / 换分块粒度后编号全变，
标注指向错误内容。

**它们不可迁移**，只能作为改写 query 意图的参考。新 qrels 一律锚在文本上。
详见 `legacy/README.md`。
