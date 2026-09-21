# Block-aware Chunking 当前实现说明

更新时间：2026-09-17

## 1. 当前实现范围

当前文档处理链路已经实现：

```text
Source
-> Source Fetcher
-> Format Router
-> Parser
-> DocumentArtifact / DocumentBlock
-> Raw Parse Quality Gate
-> Block Cleaner
-> Index Quality Gate
-> Block-aware Hierarchical ChunkBuilder
-> ParentNode / IndexReadyChunk
-> artifact.json / parents.json / chunks.jsonl
-> Dense / Sparse
-> Qdrant
```

当前尚未实现：

```text
embedding 语义边界微调
完整 Parent expansion
字段化 BM25F
固定 qrels 评测
动态网页渲染
图片 OCR 与多模态 embedding
```

核心代码：

```text
pipeline/ingest/chunker.py
pipeline/ingest/qdrant_indexer.py
```

## 2. 输入与输出

### 输入

`BlockAwareHierarchicalChunkBuilder.build()` 接收：

```python
DocumentArtifact
```

其中包含已经解析和清洗过的：

```python
List[DocumentBlock]
```

每个 Block 至少具有：

```text
block_id
type
text
section_path
order
page
quality_score
```

### 输出

```python
(
    List[ParentNode],
    List[IndexReadyChunk],
)
```

`ParentNode` 表示 section 级上下文。

`IndexReadyChunk` 表示最终进入 Dense / Sparse 索引的检索单位。

## 3. 总体流程

```text
DocumentArtifact
  |
  v
按顶层 section 分组 Block
  |
  v
每个 Block 转成一个或多个 _Piece
  |
  v
按 chunk_tokens 累积 _Piece
  |
  v
遇到 section、heading、token 上限时 emit Chunk
  |
  v
生成 fragments / full_text / dense_text / sparse_text
  |
  v
生成 ParentNode
```

## 4. Block 如何分片

Block 分片发生在 `_block_pieces()`。

它根据 Block 类型走三条路径：

```text
原子 Block
    -> 不拆

heading Block
    -> 不拆

普通长文本 Block
    -> 按 token 预算继续拆
```

### 4.1 原子 Block

当前原子类型：

```python
ATOMIC_TYPES = {
    "code",
    "formula",
    "table",
    "image",
}
```

这些 Block 不会被从中间切开。

每个原子 Block 转成一个 `_Piece`：

```python
_Piece(
    block=block,
    start_char=0,
    end_char=len(block.text),
    text=block.text,
    token_count=tokens,
    is_atomic=True,
    oversized=tokens > chunk_tokens,
)
```

如果原子 Block 本身超过 `chunk_tokens`：

```text
仍然不拆
单独形成一个 Chunk
添加 oversized_atomic_block 标记
```

这样能避免：

```text
代码围栏被截断
公式被截断
Markdown 表格结构损坏
图片 caption 和图片地址分离
```

### 4.2 heading Block

heading 永远作为一个独立 Piece：

```python
_Piece(
    block=block,
    start_char=0,
    end_char=len(block.text),
    text=block.text,
    token_count=tokens,
    is_heading=True,
)
```

heading 不直接作为最终 Chunk 的单独检索单位。

当当前 Chunk 已经包含正文时，再遇到新 heading：

```text
先 flush 当前 Chunk
不把旧 Chunk 的 overlap 带到新 section
再把 heading 放入新 Chunk
后续正文附着到这个 heading
```

因此逻辑是：

```text
heading 跟随后续正文
```

### 4.3 普通文本和 list Block

如果文本 Block 的 token 数不超过 `chunk_tokens`：

```text
整个 Block 转成一个 _Piece
```

如果超长：

```python
SentenceSplitter(
    chunk_size=chunk_tokens,
    chunk_overlap=0,
)
```

⚠️ **2026-09-21 变更**：此前这里用 `chunk_size = chunk_tokens - overlap_tokens`
（800-400=400），为的是给"回填 overlap"预留空间。overlap 取消后改为直接
用 `chunk_tokens` —— 让句子尽量吃满预算，避免产出偏小的 chunk。

历史理由（已废弃，保留供追溯）：

```text
chunk_tokens   = 800
overlap_tokens = 400
实际文本切片预算 = 400
-> 400 overlap + 400 new content = 800 token Chunk
```

拆分后，每个 Piece 记录原 Block 内的字符范围：

```text
start_char
end_char
text = block.text[start_char:end_char]
```

## 5. Piece 如何组装成 Chunk

`build()` 维护：

```text
current
    当前正在累积的有序 _Piece 列表
```

处理每个 Piece 时只有三种情况（overlap 取消后逻辑已简化）：

### 情况 A：遇到新 heading，且当前已有正文

```text
flush 当前 Chunk
开始新 section 上下文
```

### 情况 B：加入当前 Piece 会超过 `chunk_tokens`

```text
emit 当前 Chunk
再追加新 Piece（它起一个新 Chunk）
```

### 情况 C：超大原子 Block

```text
不拆
单独形成一个 Chunk
```

其余情况直接 append 到 `current`。

## 6. 如何保证普通 Chunk 不超限

`chunk_tokens` 是**硬上限**，靠两条规则保证：

1. 超限的文本 Block 在 `_split_text_block()` 阶段就被切成 `<= chunk_tokens` 的 Piece；
2. 主循环在 `current_tokens + piece.token_count > chunk_tokens` 时先收口。

所以：

```text
普通 Chunk 不超过 chunk_tokens
超大原子 Block 允许超出，但会被标记
```

（此前还需要在 emit 后裁减回填的 overlap，overlap 取消后这一步不存在了。）

## 7. overlap 已取消（2026-09-21）

**当前实现没有 overlap。** 收口就是纯粹的"造块 + 清空累积"，
不再有"从上一块尾部回填文本"的分支。

### 取消的原因（实测）

`overlap_tokens=400` 的分布是**双峰**的，且与 heading 精确互补：

| 文档类型 | overlap 触发情况 | overlap 占该文 token |
|---|---|---|
| **有 heading** | 章节本身即语义边界，几乎不触发（0~1 个） | 0% ~ 17% |
| **无 heading** | 唯一信号就是长度+overlap | **43% ~ 47%** |

全局：14/117 chunks（12%）带 overlap，占 5,407/40,069 token（**13.5%**）。

代价（以 `openai_agents_api` 为例）：

```text
总计入 token = 3,373
其中 overlap = 1,600 (47.4%)
去重后唯一 token = 1,773
```

即近一半 token 预算花在重复内容上；且重叠文本会**同时进两个 chunk 的向量**
→ 同一内容可被双命中，挤占 `candidate_k`。

而收益侧：overlap 真正起作用的只是"接住跨块指代"。实测它携带的
**2~6 句里只有第 1 句在起作用**（如 `compaction logic.`、`in production.` 这类半句）。

### 取消后的实测影响

| 项 | 取消前 | 取消后 | 变化 |
|---|---|---|---|
| chunks | 117 | **113** | −4 |
| 总 token | 40,069 | **34,662** | **−5,407 (−13.5%)** |
| 内容覆盖字符 | — | — | 无丢失（见下） |
| dense hit@5 | 0.941 | 0.941 | 持平 |
| sparse hit@5 | 0.765 | 0.765 | 持平 |
| rrf hit@5 | 0.971 | 0.971 | 持平 |
| rrf recall@20 | 1.000 | 1.000 | 持平 |

- 减少的 token 数（5,407）与独立测得的 overlap token 数**逐位相同** → 确认只移除了重复部分。
- 各文档的**片段覆盖字符数**与取消前一致（一篇因句子边界位移 +1 字符），
  说明**没有丢内容**。
- 检索指标（hit@5 全阶段、rrf recall@20/MRR）**完全不变**，个别 MRR/nDCG 有 ±0.015 波动。
  → **overlap 对检索质量没有可测收益，纯属成本。**

### 历史实现（已删除，保留供追溯）

旧实现由 `_tail_overlap()` + `_slice_tail()` 完成：从 `current` 尾部向前遍历 `_Piece`，
heading 不进入 overlap、原子块整块回退、普通 text/list piece 超预算时按 token 二分截取尾部。

### 关联：滑动窗口参数也已删除

`window_tokens` 从未参与任何计算（只被接收并保存）。它属于"语义切片"的规划范围
（用滑动窗口算 embedding 语义距离、在距离峰值处切分），该方向暂缓，故把参数一并删除，
避免留下"看起来能用、实际是死参数"的坑。详见 `issues/09`。

## 8. Fragment 来源追踪

每个最终 Chunk 都包含：

```python
List[ChunkFragment]
```

一个 Fragment：

```json
{
  "block_id": "b13",
  "start_char": 80,
  "end_char": 180,
  "block_type": "text",
  "text": "..."
}
```

坐标规则：

```text
start_char / end_char
    相对于 CleanDocumentArtifact 中该 Block 的规范化 block.text

区间
    [start_char, end_char)

单位
    Unicode code point，对应 Python str 下标
```

验证关系：

```python
fragment.text == block.text[fragment.start_char:fragment.end_char]
```

同一个 Block 可以进入多个 Chunk：

```json
{
  "chunk_id": "c01",
  "block_id": "b13",
  "start_char": 0,
  "end_char": 42
}
```

```json
{
  "chunk_id": "c02",
  "block_id": "b13",
  "start_char": 30,
  "end_char": 80
}
```

这通常来自：

```text
长 Block 内部切分
重新切 Chunk
```

## 9. full_text / sparse_text（dense_text 已删除）

### full_text

由 Piece 对应的 Block 重新渲染，**但 heading 默认不进正文**（见 `_render_block`
的 `strip_headings`，默认 `True`）：

```text
heading -> ""            (剥离! 只作 metadata, 见下)
code    -> fenced code
formula -> $$...$$
image   -> Markdown image
text    -> 原文
```

用途：

```text
dense embedding 的输入
展示 / 引用 / Parent context
```

heading 被剥离的理由：

1. 标题是"这段来自哪一章"的**引用信息**，不是内容本身；
   它已由 `chunk.section_path` + `section_block_id` + fragment 完整承载。
2. 标题行的 token 占比极小（实测 1.39%，601/43109），剥掉几乎不省 token，
   但让 chunk 正文更"纯语义"，且引用信息仍可经 metadata 还原。
3. 纯 heading 的 chunk（如 PDF 封面标题，只有标题、无正文）会被显式丢弃
   （`_make_chunk` 返回 `None`），不进索引。

被剥离的标题去哪了？存在 `chunk.metadata.headings`
（本 chunk 覆盖的标题清单）+ `chunk.section_path`（祖先链）。
生成阶段 `generation.build_context` 据此拼"背景"行，让 LLM 知道
这段属于哪一章、讲了哪几个小节 —— 详见 §9.5。

### dense_text（已删除）

历史上 `dense_text` 与 `full_text` **100% 相同**（实测 88/88），纯冗余。
已在 `IndexReadyChunk` 中删除；索引器现在直接用 `full_text` 生成 dense embedding
（见 `qdrant_indexer._chunk_payload`）。`sparse_text` 同理，不再有 dense_text 概念。

### sparse_text

只包含 BM25 需要的文本：

```python
SPARSE_TYPES = {
    "text",
    "list",
    "code",
    "formula",
    "table",
}
```

以及 image 的 caption/text。**从不包含 heading**（无论 `strip_headings` 与否）。

heading 不进 `sparse_text` 的原因：若每个 Chunk 都重复 `KEEPALIVED`，
会导致 `df` 上升、`IDF` 下降、关键词区分度降低。

> 注意：因为 `sparse_text` 本就不含 heading，而 gold 反查只读 `fragments[].text`
> （裸标题，不含 `#`），所以"剥离正文标题"对稀疏检索与评测**零影响**。

### 9.5 生成阶段的 metadata → 背景补全

正文剥离标题后，一段正文可能以"这个参数…"开头，LLM 不知道它属于哪一章。
`generation.build_context` 用 metadata 拼一行"背景"补回来：

```text
[C1] 文档: Agent Skills | 章节: The anatomy of a skill | 页码未知
背景: 文档《Agent Skills》；章节路径: The anatomy of a skill；本段涵盖小节: The anatomy of a skill；出处: https://...
<chunk body>
```

来源字段（`generation._chunk_background`）：

```text
title / file_name       文档名
section_path            章节面包屑
headings                本段覆盖的标题清单（payload 新增字段）
url                     块级来源 URL（HTML）或 source_uri（PDF）
```

`include_background` 默认 `True`，可关闭做 A/B。`headings` 缺失时优雅降级
（旧 chunks.jsonl 无此字段 → 背景行为空，只用 header 行）。

## 10. Chunk 质量与索引决策

Chunk 质量由 Fragment 对应 Block 的质量按 token 加权：

```text
chunk_score =
    sum(block_quality * block_tokens)
    / sum(block_tokens)
```

状态阈值：

```text
>= 0.85      high
>= 0.60      medium
>= 0.35      low
其他         reject
```

索引决策：

```text
high
    dense + sparse
    sparse_weight = 1.0

medium
    dense + sparse
    sparse_weight = 0.6

low
    dense only
    review_required = true

reject
    不写入索引
```

`sparse_weight` 会真正乘到 BM25 document vector 上。

## 11. ParentNode

`ParentNode` 当前按较粗的顶层 scope 分组：

```text
block.section_path[0]
```

如果没有 section_path，则使用：

```text
artifact.title
```

如果也没有，则使用：

```text
文档开头
```

Parent 保存：

```text
parent_id
artifact_id
scope_title
section_path
child_chunk_ids
full_text
token_count
page_start
page_end
```

当前 `ParentNode.full_text` 是 child chunk 文本的直接拼接。

overlap 取消后，child chunk 之间不再有重复文本，因此 Parent 文本也不再重复
（此前这是已知限制，现已随 overlap 移除而消失）。

## 12. JSON 到 Qdrant

Phase 0 导出：

```text
artifact.json
parents.json
chunks.jsonl
```

通过：

```text
pipeline/ingest/qdrant_indexer.py
```

写入 Qdrant。

### dense

输入：

```text
chunk.dense_text
```

写入：

```text
phase0_dense
```

### sparse

输入：

```text
chunk.sparse_text
```

经过：

```text
jieba
-> term
-> BM25
-> sparse vector
```

写入：

```text
phase0_sparse
```

### 同一个 point id

业务 `chunk_id` 通过稳定哈希映射为 Qdrant point id：

```python
blake2b(chunk_id) & ((1 << 63) - 1)
```

因此：

```text
dense point id == sparse point id
```

原始 `chunk_id` 保留在 payload 中。

### payload

包含：

```text
chunk_id
artifact_id
parent_id
section_path
fragments
quality
index_decision
source_uri
page_start / page_end
token_count / char_count
```

## 13. 当前已支持的文档类型

```text
本地文件
URL
HTML
Markdown
plain text
PDF
```

Parser：

```text
HTML       Trafilatura + Readability fallback + BeautifulSoup 代码块补强
Markdown   heading / paragraph / list / code / formula / table
PDF        MinerU，fallback PyMuPDF
Office     只有路由和 optional adapter 外壳
Docling    只有 optional adapter 外壳
```

## 14. 当前验证数据

测试：

```text
9 tests
OK
```

合成长文本：

```text
chunk_tokens=100
```

结果：

```text
没有普通 Chunk 超过 100 token
块之间无重复文本（overlap 已取消）
所有 Chunk 都有 fragments
fragments 能精确还原 Block 字符范围
```

Phase 0 JSON -> Qdrant 端到端：

```text
dense points: 49
sparse points: 49
sparse vocabulary: 842
manifest: ok
```

## 15. 当前限制

```text
chunk 只有上限约束，没有独立的最小 token 目标
ParentNode 按顶层 scope 分组，不是完整层级树
原子 Block 超限时仍允许单独超过 chunk_tokens
稀疏向量尚未使用字段化 BM25F
图片尚未接入 OCR、caption 模型和多模态 embedding
语义边界微调（块间 embedding 语义距离）尚未实现 —— 见下方说明
```

已消除的限制（保留供追溯）：

```text
window_tokens 参数存在但未使用        -> 参数已删除 (2026-09-21)
Parent full_text 拼接导致 overlap 重复 -> overlap 取消后不再重复
```

## 16. 一句话总结

当前实现是：

```text
结构化 Block
-> 可拆分 Piece
-> section / heading / token 约束（固定大小, 无 overlap）
-> 可追溯 Fragment
-> ParentNode + IndexReadyChunk
-> Dense / Sparse Qdrant
```

其中最关键的设计是：

```text
Chunk 是 Block 的检索索引视图，
Fragment 记录这个视图如何从原始 Block 组装而来。
```
