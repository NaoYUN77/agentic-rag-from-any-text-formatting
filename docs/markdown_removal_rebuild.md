# 去掉 Markdown 层之后的第一次真实对照

**日期**：2026-09-18
**目的**：用新 parser 重建 Phase 0 的 10 篇语料索引，回答"去掉 Markdown 中转层到底值不值"。
**结论一句话**：**值。** 结构层拿到了 Markdown 表达不了的四类信息（块级 URL、代码语言、图片 caption、超链接），并且顺带修掉了一个真实缺陷（6 个 chunk 只有 dense 没有 sparse）。

---

## 一、先说清楚这次能得出什么、不能得出什么

| 问题 | 能否回答 | 为什么 |
|---|---|---|
| 结构上变好了吗？ | ✅ **能** | 离线、确定性、可复现 |
| 检索指标变了吗？ | ❌ **不能** | 需要 `DASHSCOPE_API_KEY`，当前 `.dashscope_key` 是占位符 |
| 管线还通吗？ | ✅ **能** | 已用本地哈希向量端到端跑通 |

**必须强调**：本次索引建立在 **本地哈希向量**（`HashingEmbeddings`）之上，它**没有语义**，
dense 检索退化为字符 n-gram 重合度。所以本次重建**只能证明管线连通**，
**不能**用来判断召回率/排序质量。检索指标要等真实 Key。

manifest 里已写入 `"embedding_backend": "local"`，避免日后误读。

---

## 二、源数据：7 篇真 HTML，3 篇被迫回退

| 来源 | 篇数 | 拿到什么 | 路径 |
|---|---|---|---|
| Anthropic Engineering | 7 | **真实 HTML**（playwright 抓取，190–321 KB） | `html_readability`（DOM 直产） |
| OpenAI News | 3 | 源 HTML **拿不到** → 回退到旧产物的 `document.md` | `markdown` |

OpenAI 三篇被 Cloudflare 挡住：非浏览器 UA 返回 403，
真浏览器打开也只停在 `请稍候…` 挑战页（11458 B）。已尝试并失败的路径：
`requests` / `curl`（403）、headless Chrome `--dump-dom`（exit 21）、
Playwright 真浏览器（challenge 页）、Googlebot UA（403）、`r.jina.ai`（不可达）。

> **这个回退本身就是论据的一部分**：`document.md` 里根本没有 URL / caption / links，
> 所以那三个字段在 OpenAI 三篇上必然是 0 —— 不是解析器不行，
> 是**信息在 Markdown 那一层就已经丢了**。

**重要**：OpenAI 三篇的新旧结果**逐字段完全相同**（见下表），
它们构成一个干净的**对照组**，证明代码改动对既有路径零副作用。

---

## 三、总量对照（10 篇）

| 指标 | 旧 | 新 | 变化 |
|---|---:|---:|---|
| blocks | 644 | 607 | −37 (−6%) |
| heading | 95 | 95 | 0 |
| 代码块 | 21 | 19 | −2 |
| **代码带语言** | **0** | **16** | **+16（从无到有）** |
| **表格块** | **0** | **4** | **+4（从无到有）** |
| **带 caption** | 10 | 32 | +22 (+220%) |
| **带 links** | **0** | **72** | **+72（从无到有）** |
| **block 带 url** | **0** | **607** | **+607（从无到有，100%）** |
| parents | 17 | 16 | −1 |
| chunks | 129 | 117 | −12 (−9%) |
| chunk tokens | 44,870 | 40,069 | −4,801 (−11%) |

### 关于 block 级 url：485 → 607 是怎么补上的

第一次重建时这一格是 **485**，恰好等于 7 篇 Anthropic 的块数总和
（70+90+62+44+39+62+118 = 485），也就是说 **只有 html 路径写了 url，
3 篇 OpenAI（走 Markdown 回退）的 122 个块 url 全是 `None`**。

追下去发现根因不在抓取，而在 parser：

```
url 的唯一源头是 SourcePayload.final_uri  (ingest/source.py:112)
        │
        ├─ html.py      → url=url     ✓
        ├─ plain.py     → url=url     ✓
        ├─ markdown.py  → 5 个构造点全漏 ✗   ← 断点
        └─ pdf_common.py→ 2 个构造点全漏 ✗   ← 同类问题
```

三条 OpenAI 文档的 `artifact.source_uri` **都是有值的**，
只有 `block.url` 是 `None` —— 说明 URL 拿到手了，只是没往下传。

已补齐 markdown（5 处）与 pdf（2 处）的全部构造点。
同时加了两道防线：

1. **AST 静态检查**（`test_every_block_constructor_passes_url`）：
   扫描 4 个 parser 的所有 `DocumentBlock(...)` 调用，
   任何一个忘了传 `url=` 就直接失败。以后新增构造点不会重蹈覆辙。
2. **逐 parser 的行为测试**：html / markdown / plain 各验证一遍
   block.url 等于 source 的 url。

修复后：**607 / 607 = 100%**。

> 顺带一提：`pdf_common.py` 有同样的问题，属于**同类缺陷、顺手修掉**。
> 本地 PDF 时 `final_uri` 是文件路径，引用展示同样需要它。

---

## 四、逐篇对照（7 篇真 HTML）

| 文档 | blocks | chunks | code(带语言) | 表格 | caption | links |
|---|---|---|---|---|---|---|
| anthropic_contextual_retrieval | 78 → **70** | 17 → 16 | 4 → 2 (0 → **2**) | 0 → 0 | 6 → 6 | 0 → **9** |
| anthropic_building_effective_agents | 99 → **90** | 20 → 19 | 0 → 0 | 0 → 0 | 8 → 8 | 0 → **9** |
| anthropic_context_engineering | 64 → **62** | 12 → **9** | 0 → 0 | 0 → 0 | 2 → 2 | 0 → **16** |
| anthropic_multi_agent_research | 47 → **44** | 12 → 11 | 0 → 0 | 0 → 0 | 3 → 3 | 0 → **7** |
| anthropic_agent_skills | 44 → **39** | 9 → **8** | 0 → 0 | 0 → 0 | 5 → 5 | 0 → **7** |
| anthropic_code_execution_mcp | 63 → **62** | 14 → 13 | 12 → 12 (0 → **12**) | 0 → 0 | 1 → 1 | 0 → **6** |
| anthropic_demystifying_evals | 127 → **118** | 23 → **19** | 2 → 2 (0 → **2**) | 0 → **4** | 5 → 5 | 0 → **18** |

**两个最有说服力的单点**：

1. `anthropic_code_execution_mcp`：**12 个代码块全部拿到语言**（旧链路 0 个）。
   这篇是讲 MCP 代码执行的，代码语言对检索几乎就是元数据级别的价值。
2. `anthropic_demystifying_evals`：**第一次抽出了 4 个表格**
   （`Methods | Strengths | Weaknesses` 这类评测方法对照表），
   Markdown 表格表达不了表头/合并单元格语义，旧链路直接丢了。

---

## 五、顺带修掉的一个真实缺陷：dense/sparse 不对齐

| | 旧 | 新 |
|---|---:|---:|
| dense points | 129 | 117 |
| sparse points | **123** | **117** |
| 差值 | **6** | **0** |

旧索引有 **6 个 chunk 只有 dense 没有 sparse** —— 它们在 BM25 通道里**不可见**。
根因是 `chunker._scope()` 对纯 heading 块处理不当，产出空正文 chunk。
修掉 `_scope()` 之后，**dense == sparse == 117 完全对齐**。

这个修复不是本次"去 Markdown"的目标，是之前清理 `_scope` 时一并做的，
但它的效果在这次重建里第一次显性地量化出来了。

---

## 六、section_path 的实际改善与仍然存在的问题

### 改善：Markdown 残留标记消失了

```
旧一级前缀:  **Excessive token consumption from tools makes agents less efficie…
新一级前缀:  Excessive token consumption from tools makes agents less efficient…
```

旧链路把 `**加粗**` 的星号带进了章节标题，并且直接进了检索索引。

### 仍然存在：`issues/11` 的"幻影根标题"

`anthropic_contextual_retrieval` 的章节树：

```
├─ A note on simply using a longer prompt        ← 正文第一个 heading 是 lv3
  ├─ A primer on RAG: scaling to larger knowledge bases
  ├─ Introducing Contextual Retrieval
  │   ├─ Implementing Contextual Retrieval
  │   ├─ Using Prompt Caching to reduce the costs of Contextual Retrieval
  │       ├─ Methodology
  │       ├─ Performance improvements
  │       ├─ Implementation considerations
  ├─ Further boosting performance with Reranking
  ├─ Conclusion
  ├─ Appendix I
  ├─ Acknowledgements
```

整棵树挂在一个 **"补充说明"性质的 lv3 小节**下面。根因是 readability 剥掉了
网站导航里的 `h1/h2`，导致正文第一个 heading 就是 lv3。

**关键澄清：这不是本次改动引入的，也不是本次改动能修的。**
旧产物 `experiments/anthropic_contextual_retrieval/artifact.json` 里
`b0005 lv3 sp=[]` 一模一样，`parents` 同样只有 2 个。
本次改动对 section_path 的语义是一个**恒等变换**。

---

## 七、索引重建结果

```
dense collection   : phase0_dense    117 points
sparse collection  : phase0_sparse   117 points
sparse 词表        : 4158（旧 4267）
chunks             : 117（旧 129）
parents            : 16（旧 17）
embedding backend  : local  ← 无语义，仅供连通性验证
```

产物：
- `experiments_rebuilt/<doc>/{artifact.json, parents.json, chunks.jsonl}`
- `index_artifacts/phase0_rebuilt/`
- 备份：`backups/20260918_pre_markdown_removal/`（含旧 `phase0_combined` 与旧 `qdrant_data`）

---

## 七之二、qrels 在重建语料上的存活验证

gold 锚在**文本片段**上，所以"重建后还能不能解析"是评估链路的前置条件。
它不需要 embedding、不需要 API Key，可以离线验证。

工具：`pipeline/check_qrels_resolution.py`（新旧语料各跑一遍，逐条列出）

### 最终结果

```
[OLD] experiments          fragments=700  keys=612  gold ok=13/13
[NEW] experiments_rebuilt  fragments=644  keys=575  gold ok=13/13
结果: PASS — gold 在新旧语料上都能解析
```

### 这个过程实际挖出了三类问题

**① 标注里带了 Markdown 标记（预期内）**

`ce_01` / `ev_02` 的 gold 片段写成 `**Context** refers to ...`，
而新语料里已经没有 `**`。经核对，去掉标记后句子在**新旧语料里都存在**，
且都落在同一个 `(文档, block)` 上 —— 内容没有丢，只是标记层被去掉了。
含字面 `**` 的 block 数量从 9 → 0。

**② 两个 block_id 抄错了 + 缺少文档归属（真实缺陷）**

`block_id` 是**文档内序号**，跨文档必然重名，而 qrels 里没写它属于哪篇文档。

| query | 声明 | 实际 |
|---|---|---|
| `ev_01` | `b0023` | `anthropic_demystifying_evals` **`b0021`** |
| `ex_01` | `b0027` | `anthropic_agent_skills` **`b0023`** |
| `ev_02` | `b0037` | `anthropic_demystifying_evals` **`b0035`** |

已按文本片段反查修正，并给每条 gold 补上 `source_doc` / `source_uri`。
这也再次印证：**block_id 只能当提示，文本片段才是锚点** ——
block_id 会随 parser 改动漂移（本次就漂了），文本不会。

**③ 归一化过度，删掉了标识符里的下划线（实现缺陷）**

第一版 `normalize_text` 用 `[*_`]` 无脑删除，把 `doc_a` 变成了 `doca`。
技术文档里 `snake_case`、`doc_a` 遍地都是，这是会静默污染检索的错。
改成**成对匹配**（`**x**` / `__x__` / `` `x` `` 才删），
并加了 `NormalizeTests` 锁住这个边界。

### 结论

去 Markdown 层**没有造成 gold 失效**。
最初的 11/13 是三个独立原因叠加的假象：标记差异（设计如此）+ 标注抄错（标注问题）
+ 归一化实现 bug（实现问题）。修正后 13/13 通过。

---

## 八、下一步（按优先级）

1. **补 `DASHSCOPE_API_KEY`，用 `--backend qwen` 重跑** —— 这是拿到检索指标的唯一路径。
   命令：`.\.venv_rag\Scripts\python.exe rebuild_phase0_index.py --backend qwen`
2. **qrels 的 gold 要锚在文本上**，不能锚 ID（id 含位置序号，一重建就变）。
   设计已在 `pipeline/eval/` 里，`phase0_url_draft.jsonl` 有 13 条草稿；
   已验证 13/13 在新旧语料上均可解析。**下一步是补正式标注**（当前 `annotated_by: draft`）。
3. **`issues/11` 是 P1 且真实存在** —— 它阻断 Contextual Retrieval
   （章节归属不可信时，注入章节上下文没有依据）。但本次已确认：
   **它和"去 Markdown 层"是两件独立的事**，不要混在一起做。
4. **OpenAI 三篇要么补 sidecar HTML，要么明确接受 Markdown 路径** ——
   建议后者，并在 index manifest 里标出来。

---

## 附：复现命令

```bash
cd pipeline

# 1. 抓源（Anthropic 7 篇会成功；OpenAI 3 篇因 Cloudflare 失败，属预期）
.venv_rag/Scripts/python.exe fetch_phase0_sources.py

# 2. 结构对照（不写索引）
.venv_rag/Scripts/python.exe rebuild_phase0_corpus.py

# 3. 重建索引（有真 Key 时换成 --backend qwen）
.venv_rag/Scripts/python.exe rebuild_phase0_index.py --backend local

# 4. gold 存活验证（不需要 API Key）
.venv_rag/Scripts/python.exe check_qrels_resolution.py

# 5. 单篇文章的全链路实况
.venv_rag/Scripts/python.exe show_url_doc.py --doc anthropic_contextual_retrieval

# 6. 测试
.venv_rag/Scripts/python.exe -m pytest tests/ -q     # 67 passed
```
