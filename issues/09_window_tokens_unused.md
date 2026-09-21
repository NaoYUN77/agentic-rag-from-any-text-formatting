# 问题 09: window_tokens 参数空转, 语义边界微调未实现

**优先级: P2**　**状态: ✅ 已解决(2026-09-21, 按方案三: 删除参数)**

> **解决记录（2026-09-21）**
>
> 采用**方案三：删除该参数**。`window_tokens` 已从 `BlockAwareHierarchicalChunkBuilder.__init__`、
> `IngestPipeline.ingest`、`format_router_demo.py` 的 CLI 中一并移除；`chunker.py` 里
> "预留字段、当前未使用"的注释段也删掉了。
>
> 选择方案三而非方案一（只声明）的理由：参数空转本身就是误导，而"标记为预留"仍会让
> CLI 暴露一个调了没效果的开关。**直接删掉，需要时再加回来**——反正将来真要实现语义边界，
> 参数名和语义（窗口大小）也不一定就是现在的 `window_tokens`。
>
> 同一批次还**取消了 `overlap`**（见 `CHANGELOG.md` 的 Removed 段）：实测 overlap 在无 heading
> 的文档上占该文 token 的 43%~47%，而移除后检索指标 hit@5 全阶段持平、总 token 降 13.5%。
> 因此"用 overlap 回填拼成 800"这条路径也一起消失了 —— 语义切分方向现在**完全空白**，
> 不再有半成品实现。
>
> 方案二（实现 embedding 语义边界）**仍然保留为待办**，但要遵守下面这条判断：
> **只应在无结构文档上做**（有 heading 的地方再用语义切分会与作者边界打架）。
>
> 下方原始记录保留，供追溯当时的分析。

## 现象

`BlockAwareHierarchicalChunkBuilder.__init__` 接收 `window_tokens` 参数并保存为
`self.window_tokens`,但**该字段在后续代码中从未被读取**。

实测全代码库引用:

```text
format_router_demo.py:43     window_tokens=args.window_tokens,   ← 调用方传参
ingest/chunker.py:129        window_tokens: int = 400,           ← 签名声明
ingest/chunker.py:139        self.window_tokens = window_tokens   ← 唯一一次赋值
ingest/pipeline.py:41        window_tokens: int = 400,           ← 调用方传参
ingest/pipeline.py:78                    window_tokens=window_tokens,
ingest_markdown_corpus.py:123/131/197    ← 旧 LlamaIndex 链路(用 window_size)
rebuild_llamaindex_corpus.py:157/207     ← 旧 LlamaIndex 链路(用 window_size)
```

`grep -n "self.window" ingest/chunker.py` 只有第 139 行(赋值),
**零处读取**。参数在 Phase 0 链路上是纯粹的死重。

## 根因

规划文档设想的切分算法是:

```text
① 用 window_tokens 大小的滑动窗口做 embedding
② 计算相邻窗口的语义距离
③ 在语义距离的局部峰值处切分
④ 再用 chunk_tokens / overlap_tokens 做长度约束
```

实际实现只做了 ④,跳过了 ①~③。`_split_text_block()`(`chunker.py:149-175`)的做法是:

```python
def _split_text_block(self, block: DocumentBlock) -> List[_Piece]:
    total = self._token_size(text)
    if total <= self.chunk_tokens:
        return [_Piece(block, 0, len(text), text, total)]

    splitter = SentenceSplitter(          # ← 纯句子边界, 无 embedding
        chunk_size=self._text_piece_tokens,   # = chunk_tokens - overlap_tokens = 400
        chunk_overlap=0,
    )
    parts = splitter.split_text(text)
    ...
```

即:**用 `SentenceSplitter` 按句子边界切到 ≤400 token,再靠 overlap 回填拼成 800**。
`chunker.py` 全文不含 `embedding` / 语义距离 / 相似度计算。

## 影响

```text
① 文档承诺与实现不符
   README 与规划文档都写了"边界语义微调", 实际只有长度约束 + 句子边界
   → 误导读者, 也误导后续开发者以为该能力已具备

② 断点质量未知
   规划的核心假设是"语义断点比长度断点好", 但该假设在本项目【从未被验证】
   → 无法判断"接入 embedding 边界"是否真的能提升检索质量

③ 参数误导
   CLI 暴露 --window-tokens, 用户以为可调, 实际调了没有任何效果
```

**注意**:这不一定是缺陷。当前设计是「Block 边界优先 + 句子边界兜底」,
而 Block 边界本身就是**结构驱动**的(章节、段落),质量可能高于语义距离驱动的切分。
问题在于:**这个取舍从未被显式声明,也没被量化验证**。

## 解决方向

```text
方案一(最小, 推荐先做): 诚实声明
   - 把 CLI 的 --window-tokens 标记为"预留, 当前无效"
   - 在 README / 规划文档里明确写"语义边界未实现"
   - 难度: 极低; 收益: 消除误导
   - 但注意: 这会暴露一个"功能缺口", 需产品上接受

方案二: 实现 embedding 语义边界
   - 在 _split_text_block 里对超长 text block 做窗口滑动的 embedding 距离计算
   - 在距离峰值处切, 再用 chunk_tokens 约束
   - 难度: 中; 成本: 每次切块都要调 embedding API(切块阶段而非检索阶段!)
   - 风险: 语料仅 10 篇, 无评估集, 无法证明收益

方案三: 删除该参数
   - 承认不需要语义边界, 从签名和 CLI 里移除
   - 难度: 低; 但会破坏与旧 LlamaIndex 链路的参数对称性
```

**推荐顺序**:先做方案一(声明),等评估 harness 建好后再决定方案二是否值得做。
在没有 qrels 的前提下实现语义边界,属于"无法验证收益的优化"。

## 关联

- `docs/plan_vs_implementation.md` 偏差 A、参数对照表
- `docs/block_structure_and_blockification.md` 第 8 节
- `issues/02_constraint_layer.md` —— 同属"边界质量"问题族,但 02 针对旧链路

## 待验证

```text
[ ] 当前"Block + 句子边界"的断点, 相对"纯长度切分"好多少?
    (可做 A/B: 关闭 Block 边界 vs 开启, 比 recall@k)
[ ] 超长 text block 有多少? 它们的切点是否合理?
    (78 个 block 被切进多个 chunk, 需逐个检查断点位置)
[ ] 接入 embedding 边界的边际收益, 是否值得每次切块的 API 成本?
```
