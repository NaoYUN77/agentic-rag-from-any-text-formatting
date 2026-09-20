# 代码健康审计：三层架构与冗余

> 审计日期：2026-09-18
> 方法：AST 解析取 dataclass 字段 + 全库正则统计引用次数 + 真实产物量化存储开销

---

## 一、三层架构的实现状态

### ① Parser 输出 → Canonical Document / Block ✅ 已实现

```text
DocumentArtifact   18 字段
DocumentBlock      17 字段
```

**但"canonical"这个说法要打个折扣**：所有 parser 都先转成 Markdown 再进
`parse_markdown_text()`，所以 Block 的表达力上限 = Markdown 的表达力。
详见 `docs/block_structure_and_blockification.md` 与 `issues/12`。

### ② 检索系统吃什么 → Node / Sentence / Chunk ✅ 已实现（两套并存）

```text
IndexReadyChunk    17 字段
ChunkFragment       7 字段（page / bbox 是 2026-09-18 新增）

BlockAwareHierarchicalChunkBuilder  → 800-token chunk（当前默认）
SentenceWindowChunkBuilder          → 单句 + metadata 窗口（备选）
```

**哪一套是"the"检索单元尚未决定** —— 这正是评估集要回答的问题。

### ③ 引用信息 → Metadata ✅ 已实现

```text
Citation           16 字段
fragment 级 bbox   100% 填充（528/528）
chunk 级 bbox      54/88（跨页的不给，符合设计）
```

引用链路的六个环节见 `docs/eval_harness_design.md` 与 memory 记录。

---

## 二、冗余审计：payload 有 79.2% 写了没人读

实测：88 个 chunk，payload 合计 **350.6 KB**。

| 字段 | 大小 | 占比 | 状态 |
|---|---|---|---|
| `fragments` | 129.7 KB | **37.0%** | ❌ 写了没人读 |
| `text` | 59.8 KB | 17.0% | ✅ 被读 |
| `dense_text` | 59.8 KB | **17.0%** | ❌ **与 text 100% 相同** |
| `sparse_text` | 57.2 KB | **16.3%** | ❌ 写了没人读 |
| `index_decision` | 13.4 KB | 3.8% | ❌ 写了没人读 |
| `quality` | 11.5 KB | 3.3% | ❌ 写了没人读 |
| 其余 20 个字段 | 19.2 KB | 5.5% | 混合 |

```text
写了但没人读的字段合计: 277.5 KB (79.2%)

其中 dense_text + sparse_text: 117.0 KB
其中 fragments:                129.7 KB
```

### 两个确凿的浪费

**① `dense_text` 与 `text` 完全相同**

```text
text == dense_text : 88 / 88  （100%）
```

`dense_text` 是纯重复。它只在**建索引时**用于向量化，检索时没人读。
把它写进 payload 等于每条 chunk 多存一份全文。

**② `fragments` 比正文还大 225%**

```text
fragments 总字节 132796  vs  full_text 总字节 59006  → 225%
```

因为每个 fragment 都带 `block_id` / `start_char` / `end_char` /
`block_type` / `page` / `bbox` / `text` 七个键，JSON 开销远超文本本身。

**而 `fragments` 在检索侧一次都没被读过** —— 引用走的是 `page` / `bbox`，
不是 fragments。

### 附带发现：payload 里 14/30 个字段无人读

```text
parent_id             0    （parent expansion 未实现）
section_block_id      0
source_type           0
mime_type             0
parser                0
token_count           0
dense_text            0
sparse_text           0
fragments             0
quality               0
index_decision        0
overlap_from_previous 0
window_sentences      0    （只有 window_text 被读）
window_token_count    0
```

---

## 三、死字段（模型里定义了但没人用）

| 字段 | 所属 | 引用次数 |
|---|---|---|
| `DocumentBlock.ocr_confidence` | Block | **0** |
| `DocumentBlock.quality_flags` | Block | 1 |
| `IndexReadyChunk.section_block_id` | Chunk | 1 |

`ocr_confidence` 是 0 —— OCR 链路完全没实现，字段是预留的。

---

## 四、死代码与空转参数（前几轮已发现，此处汇总）

| 项 | 位置 | 问题 |
|---|---|---|
| `window_tokens` | `chunker.py:139` | 赋值后零读取（`issues/09`） |
| chunker 去重 | `chunker.py:427-434` | key 含唯一 chunk_id，永不生效 |
| `overlap` | `chunker.py` | 实测 84.6% 的 chunk 根本不触发 |
| `_tail_overlap` / `_slice_tail` | `chunker.py:303-366` | 为 overlap 服务，若关掉 overlap 即无用 |

---

## 五、为已否决方案写的代码

| 项 | 状态 | 处置建议 |
|---|---|---|
| `sentence_chunker.py`（230 行） | 用户已决定不用单句（成本 ×9.4） | **保留作备选**，但需在文档标注"未采用" |
| `pdf_pymupdf.py`（143 行） | AGPL 许可，pdfplumber 已够用 | 保留为可选适配器，`requirements.txt` 已注释 |

保留这两个是合理的（都是"可切换的备选"），但**必须让人一眼看出它们不是主路径**。

---

## 六、空壳与失效

```text
ingest/parsers/docling_optional.py    15 行，只抛 NotImplementedError
ingest/parsers/office_optional.py     15 行，同上
eval/pre_llamaindex/                  2 个失效 qrels（gold 是裸 point_id）
```

`docling_optional` / `office_optional` 是**占位契约**（路由表里指向它们），
保留合理。但 `eval/pre_llamaindex` 已确认失效，建议移入 `eval/legacy/`。

---

## 七、清理建议（按性价比排序）

```text
P0  从 payload 移除 dense_text / sparse_text
    —— 它们只用于建索引，检索侧零读取。省 117 KB / 33%
    —— 风险：极低（确认过无人读）

P0  精简 fragments 或改为按需返回
    —— 当前占 37%，且比正文大 225%
    —— 方案 a: 只存 block_id + span，不存 text（引用不需要重复文本）
    —— 方案 b: 干脆不进 payload（引用走 page/bbox）
    —— 风险：低，但需确认没有任何下游依赖

P1  移除 ocr_confidence 等死字段
    —— 或明确标注"预留，未实现"

P2  清理 window_tokens 空转参数
    —— 要么实现，要么从签名和 CLI 移除（issues/09）

P3  清理 chunker 去重死代码

P3  eval/pre_llamaindex 移入 legacy/
```

---

## 九、清理执行记录（2026-09-18）

### P0-1 + P0-2：payload 瘦身 ✅ 已完成

**关键点**：`_chunk_payload` 的返回值**既喂给索引器又存进 Qdrant**，
所以不能直接删字段 —— 必须先让索引器从 `chunk` 直接读。

改动：

```text
ingest/qdrant_indexer.py
  新增 _slim_fragments()   去掉 fragment.text，只留定位信息
  _chunk_payload()         移除 dense_text / sparse_text / quality /
                           index_decision / overlap_from_previous /
                           window_sentences，并精简 fragments
  build_index_plan()       dense_text / sparse_text / index_decision
                           改为从 chunk 直读（不再经 payload）
```

**实测效果：**

```text
payload 体积（88 chunks）:  392.4 KB  →  193.9 KB
减少: 198.5 KB (51%)
```

**验证：**

```text
引用字段全部保留   page / page_start / page_end / section_path /
                  bbox / title / file_name / text   ✅
索引文本仍正确     dense[0] = "# Red Hat Enterprise Linux 7..."     ✅
                  sparse[0] = "Last Updated: 2023-11-27..."         ✅
单页 chunk bbox   54 / 54 (100%)
跨页 chunk bbox   34 个按设计不给，靠 fragment 级精确定位
fragment 级       每个都带 page + bbox，跨页也能定位到具体某页
```

### P2：window_tokens 诚实声明 ✅ 已完成

```text
chunker.py
  BlockAwareHierarchicalChunkBuilder 加类 docstring，明确写
  "⚠️ window_tokens 目前不参与任何计算"，并指向 issues/09
  字段注释改为"预留字段：当前未使用"

format_router_demo.py
  --window-tokens 的 help 改为"预留参数, 当前不参与计算 (见 issues/09)"
```

**保留参数而非删除**：删掉会破坏既有调用方；文档标注足以消除误导。

### P3：chunker 去重死代码 ✅ 已删除

```text
chunker.py
  删除"去重重叠产生的重复 chunk"整段（原 427-434 行）
  替换为直接的 parent_id 赋值
  保留注释说明为什么它是死代码（避免以后有人再加回来）
```

### P3：eval/pre_llamaindex → legacy ✅ 已完成

```text
git mv eval/pre_llamaindex/mixed_qrels.jsonl    eval/legacy/
git mv eval/pre_llamaindex/rerank_smoke.jsonl   eval/legacy/
新增 eval/legacy/README.md  说明为什么失效、还能怎么用
更新 eval/README.md 的引用
.gitignore 追加 pipeline/eval/runs/（评估产物不入库）
```

用 `git mv` 保留了文件历史。

### 未做（有意保留）

```text
ocr_confidence        OCR 链路是规划中的功能，字段是合法预留，不该删
                      （已在本文件第三节记录其 0 引用状态）
sentence_chunker.py   用户已决定不用单句，但保留作可切换备选
pdf_pymupdf.py        AGPL，保留为可选适配器（requirements.txt 已注释）
docling/office 空壳    路由表里的占位契约，保留合理
```

### 清理后验证

```text
单元测试        42 / 42 通过
端到端          PDF 460 blocks → 10 parents → 88 chunks, quality high
build_index_plan  dense 88 / sparse 88，payload 字段 24 个
```

---

## 八、一句话结论

**三层架构都实现了，引用链路是通的。存储层的冗余已清理 51%：
payload 从 392.4 KB 降到 193.9 KB，同时引用字段（page / bbox /
section_path）全部保留。**

剩下的冗余是"合法预留"（OCR 字段）或"可切换备选"（单句切块器、
PyMuPDF 适配器），不属于该删的东西。
