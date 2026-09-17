# Block 结构 与 Block 化过程

> 本文回答两个问题:**项目的 block 结构长什么样?** 一篇文档是如何被 block 化的?
> 所有结论均来自对 `pipeline/ingest/` 源码的阅读 + 对 `experiments/` 真实产物的实测验证。

---

## 0. 一句话总览

**Block 化 = 把任意格式的文档,拆成"结构化的原子段落列表"。**

它不是切片(chunking),也不涉及任何 embedding。它是"理解文档结构"这一步的产物,是后续 clean → chunk → index 的**唯一输入**。

```
Source(URL/PDF/MD)
   ↓ ① SourceLoader       抓取/读取,得到 bytes + sha256
   ↓ ② FormatRouter       看 mime/magic bytes,决定用哪个 parser
   ↓ ③ Parser             HTML→Markdown, Markdown→Block[]   ★ 这里发生 block 化
   ↓ ④ Cleaner            去样板/去重/去空白,重排 order
   ↓ ⑤ QualityGate        打分,决定 dense/sparse 是否入库
   ▼
DocumentArtifact { artifact_id, blocks: [DocumentBlock, ...] }
```

---

## 1. Block 的数据结构

定义在 `pipeline/ingest/models.py:38`:

```python
@dataclass
class DocumentBlock:
    block_id: str            # b0001 —— 位置序号
    type: str                # text / heading / list / code / formula / image / table
    text: str                # 纯文本内容
    markdown: str = ""       # 保留格式的原始形式
    parent_id: Optional[str] # 所属 heading 的 block_id
    section_path: List[str]  # 面包屑,如 ["A note...", "A primer on RAG"]
    order: int = 0           # 文档内顺序(cleaner 后重排)
    page: Optional[int]      # PDF 页号(HTML 语料为 None)
    bbox: Optional[List[float]]   # 版面坐标(OCR/PDF,当前未使用)
    code_language: Optional[str]  # 代码块语言
    latex: Optional[str]          # 公式 LaTeX
    image_uri: Optional[str]      # 图片地址
    caption: Optional[str]        # 图片说明
    ocr_confidence: Optional[float]
    quality_score: Optional[float]
    quality_flags: List[str]
    metadata: Dict[str, Any]      # 如 {"level": 2, "source": "html_code_supplement"}
```

关键区分:
- **`text` vs `markdown`**:`text` 是纯内容(便于检索/切片),`markdown` 保留 `##`、``` 等标记(便于还原渲染)。heading 块的 `text` 是标题文字,`markdown` 是 `## 标题`。
- **`block_id` vs `order`**:`block_id` 是生成时的编号,`order` 是 cleaner 重排后的顺序。**两者在正常流程下相等,但并非同一字段**。

---

## 2. Block 的 7 种类型

实测 10 篇正式语料(644 个 block)的类型分布:

| type | 数量 | 含义 | 是否可切分 | 是否进 sparse |
|---|---|---|---|---|
| `text` | 466 | 普通段落 | ✅ 可切 | ✅ |
| `heading` | 95 | 标题(`metadata.level` = 1~6) | ❌ 原子 | ❌ |
| `image` | 32 | 图片(仅有 URI,无 OCR 文字) | ❌ 原子 | 仅 caption |
| `list` | 30 | 有序/无序列表整体 | ✅ 可切 | ✅ |
| `code` | 21 | 代码块 | ❌ 原子 | ✅ |

> 另有 `formula`、`table` 两种类型已在代码中定义,但当前 10 篇 URL 语料里未出现。

分类逻辑写死在 `chunker.py:16-18`:

```python
ATOMIC_TYPES   = {"code", "formula", "table", "image"}   # 不切分
MERGEABLE_TYPES = {"heading", "text", "list"}            # 可合并成 chunk
SPARSE_TYPES   = {"text", "list", "code", "formula", "table"}  # 进 BM25(不含 heading/image)
```

**注意 `heading` 的双重身份**:它既不属于 ATOMIC(因为它很短),也不被切分;它在 chunk 组装时起**断点**作用(`chunker.py:404`:遇到 heading 且当前已有正文 → 立刻 emit,不携带 overlap)。

---

## 3. Block 化过程(一篇 HTML 文档的完整旅程)

### 3.1 路由:决定用哪个 parser

`router.py` 按优先级判断:

```
magic bytes == %PDF-        → pdf_mineru (fallback: pdf_docling, pdf_pymupdf)
mime 含 html 或 <html> 出现 → html_trafilatura (fallback: html_readability)
后缀 .md/.markdown          → markdown
后缀 .txt/.rst              → plain_text
后缀 .docx/.pptx/.xlsx      → office_optional
```

实测 10 篇的 `parser` 字段分布,证明**同是 HTML,会走不同分支**:

| 来源 | parser | 篇数 |
|---|---|---|
| Anthropic Engineering | `readability_structured` | 7 |
| OpenAI News | `trafilatura` | 3 |

原因在 `html.py:84-102`:trafilatura 和 readability 各抽一遍,`_heading_count()` 比较谁抽出的标题多;readability 标题更多且正文长度够(≥ `min(300, len/2)`)就用 readability,否则用 trafilatura。

### 3.2 抽取:HTML → Markdown

- **trafilatura 路径**:`trafilatura.extract(output_format="markdown", include_links/images/formatting/tables=True, favor_precision=True)`
- **readability 路径**:`readability.Document(html).summary()` → 得到净化后的 HTML → `markdownify(heading_style="ATX")` 转 Markdown

两条路最终都产出 **Markdown 文本**,再统一交给 `parse_markdown_text()`。

### 3.3 核心:Markdown → Block[]

`parsers/markdown.py:29` 的 `parse_markdown_text()` 是 block 化的心脏。它是一个**逐行状态机**:

```
维护 4 个状态:
  blocks:        已产出的 block 列表
  paragraph:     正在累积的段落行缓冲
  heading_stack: 标题层级栈(用于算 section_path)
  current_parent: 当前所属 heading 的 block_id
  order:         递增计数器
```

逐行扫描,遇到不同语法就触发不同分支:

| 遇到 | 动作 |
|---|---|
| ` ``` ` | ① flush 当前段落 ② 吞到闭合 ``` ③ 产出 `code` 块 |
| `#{1,6} 标题` | ① flush 段落 ② 弹出栈中层级 ≥ 当前的标题 ③ 压入新标题 ④ 产出 `heading` 块 ⑤ `current_parent = 本 heading` |
| 空行 | flush 段落 |
| 其他 | 追加到 `paragraph` 缓冲 |

`flush_paragraph()` 里再做子分类:

```python
# markdown.py:51-88
if raw.startswith("$$") and raw.endswith("$$"):   → type="formula"
elif IMAGE_RE.match(raw):                          → type="image"
elif re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", raw):   → type="list"
else:                                              → type="text"
```

**编号规则**(`markdown.py:37`):

```python
def next_id() -> str:
    return f"b{order:04d}"
```

每个分支都在 `next_id()` 之前先 `order += 1`,所以编号是 **b0001, b0002, b0003...** 连续递增。

### 3.4 后置补充:HTML 里被漏掉的代码块

trafilatura/readability 有时会把 `<pre class="language-ts">` 里的代码吃掉或格式错乱。`html.py:111-123` 做了一个补丁:

```python
existing_code = {b.text.strip() for b in artifact.blocks if b.type == "code"}
for language, code in _extract_code_blocks(html):     # BeautifulSoup 直接选 <pre code>
    if code in existing_code:
        continue                                      # 已存在则跳过
    artifact.blocks.append(DocumentBlock(
        block_id=f"b{len(artifact.blocks) + 1:04d}",  # ← 注意:用 len(blocks),不是 order
        order=len(artifact.blocks) + 1,
        metadata={"source": "html_code_supplement"},
    ))
```

实测影响:`doc_58b1b15bbd49` 补了 3 个块(24 块中 3 个),`doc_5c3e74c436ef` 补了 2 个(b0077/b0078)。

**这两个补充块有两个结构缺陷**(可在真实产物中观察到):

1. **`section_path` 为空 `[]`** —— 因为它们是 parse 完直接 append 到尾部,没有经过 heading_stack 推导,丢失了章节归属。实测:
   ```
   b0076 text  sec=['A note...', 'Acknowledgements']   ← 正常块
   b0077 code  sec=[]                                   ← 补充块, 丢了归属
   b0078 code  sec=[]                                   ← 补充块, 丢了归属
   ```
2. **顺序错乱** —— 代码块被怼到文档最尾部,而不是它原本所在的章节位置。`chunker._group_blocks()` 按 `section_path[0]` 分组,这两个空 section_path 的块会被归到"文档开头"组,从而**跨章节漂移**。

### 3.5 清洗:Cleaner

`cleaner.py:29` 的 `clean_blocks()` 做四件事:

```python
1. 按类型归一化空白:
   code    → 统一换行符,strip("\n") 两端空行
   formula → strip latex
   其他    → 多空行压成一个 \n\n,strip 两端
2. 丢弃空块:        if not block.text and not block.image_uri: continue
3. 丢弃样板块:      text/heading/list 且 len < 300 且命中 _BOILERPLATE 正则 → 丢弃
4. 丢弃连续重复:     type 与 text 都与上一个相同 → 丢弃
```

最后**重排 order**:

```python
for order, block in enumerate(cleaned, 1):
    block.order = order
```

**这里产生了一个重要的实测现象**:`order` 从 1 重新计数,但 `block_id` **保持原样不动**。所以如果 cleaner 过滤掉了前面的块,两者就会脱钩。

实测证据(`experiments/router_phase0_url/artifact.json`):

```
blocks[0] = b0003 / order=1     ← b0001、b0002 被 cleaner 吃掉了
blocks[1] = b0004 / order=2
...
blocks[-1] = b0054 / order=52   ← 52 个块, 编号却是 b0003~b0054
```

**但正式索引的 10 篇语料全部是 `b0001/order=1` 起步、零脱钩**:

```
doc_5c3e74c436ef  n= 78  OK  首个=b0001/order1
doc_1d0b6b72bdf2  n= 99  OK  首个=b0001/order1
...
合计 644 blocks, 不一致 0
```

原因:这 10 篇是"结构化正文页",cleaner 没有在前面过滤掉任何块。所以 **脱钩是潜在风险,而不是当前故障**。

### 3.6 质量门:QualityGate

`quality.py:15` 的 `assess_raw_quality()` 计算:

```
指标: char_count / block_count / duplicate_ratio / garbled_ratio ...
扣分: char_count < 100        -0.45
      char_count < 300        -0.15
      duplicate_ratio * 0.30
      garbled_ratio * 5       (上限 0.40)
      无 title                -0.05
分级: ≥0.85 high | ≥0.60 medium | ≥0.35 low | 否则 reject
```

实测 `doc_5c3e74c436ef`:

```json
{"score": 0.9875, "status": "high", "flags": [],
 "metrics": {"char_count": 16714, "block_count": 78, "text_block_count": 68,
             "code_block_count": 4, "image_block_count": 6,
             "duplicate_ratio": 0.0417, "garbled_ratio": 0.0, "language": "en"}}
```

`decide_index()` 再换算成入库策略:

| quality | dense | sparse | sparse_weight | 备注 |
|---|---|---|---|---|
| high | ✅ | ✅ | 1.0 | |
| medium | ✅ | ✅ | 0.6 | |
| low | ✅ | ❌ | 0.0 | rerank_penalty=0.2,需人工复核 |
| reject | ❌ | ❌ | 0.0 | 直接不入库 |

---

## 4. section_path:block 的"面包屑"

这是 block 结构里**最有价值的字段**——它让每个 block 都知道自己属于哪个章节。

### 4.1 生成规则

在 `markdown.py:118-140` 的 heading 分支:

```python
level = len(heading.group(1))          # ## 就是 2
while len(heading_stack) >= level:     # 弹出所有层级 ≥ 当前的
    heading_stack.pop()
heading_stack.append(title)            # 压入自己
# heading 块自己的 section_path 不含自己:
section_path = list(heading_stack[:-1])
# 子块的 section_path 含完整栈:
current_parent = block_id
```

### 4.2 实测验证

`doc_5c3e74c436ef` 的标题序列(b0005 是 lv3,诡异吗?往下看):

```
b0005 lv=3 sec=[]                                    "A note on simply using a longer prompt"
b0006 text   sec=['A note...']                        ← b0005 的子块,parent=b0005
b0009 lv=2 sec=['A note...']                          ← lv2 入栈前先 pop 掉 lv3
b0010 text   sec=['A note...', 'A primer on RAG...']  ← b0009 的子块
b0022 lv=3 sec=['A note...', 'A primer on RAG...']
b0025 text   sec=['A note...', 'A primer...', 'The context conundrum...']
b0026 lv=2 sec=['A note...']                          ← lv2 → pop 掉 primer 和 conundrum
b0027 text   sec=['A note...', 'Introducing Contextual Retrieval']
```

**栈行为完全正确**:`while len(stack) >= level: pop()` 保证了 lv2 出现时,lv3 和 lv2 都被弹掉。

### 4.3 一个真实的文档结构缺陷

注意 `b0005` 是 **lv=3**,但它是文档里第一个标题——前面**没有 lv1/lv2**。

原因是 `readability_structured` 抽取时把网站导航/文章标题当成 `<h1>` 剥掉了,导致正文从 `<h3>` 开始。后果:

- `heading_stack` 第一层就是 `"A note on simply using a longer prompt"`(本该是顶级章节的一个小子节)
- 这个 lv3 标题**污染了整篇文档的 section_path 前缀**——后面 71 个 block 的面包屑全都以它开头

实测 `_scope()` 分组结果(`chunker.py:201`:取 `section_path[0]`,空则用 `artifact.title`):

```
doc_5c3e74c436ef:  parents=2
    7 blocks  <- (文档开头)                              ← 开头 4 段 + 尾部补充块
   71 blocks  <- A note on simply using a longer prompt  ← 71 个块挤在一个"章节"里
```

对比结构更健康的 `doc_723b16a48c5d`(lv2 作为顶层,抽取正常):

```
b0004 lv=2 sec=[]                                     "Context engineering vs. prompt engineering"
b0010 lv=2 sec=['Context engineering vs. ...']         "Why context engineering is important"
b0017 lv=2 sec=['Context engineering vs. ...']         "The anatomy of effective context"
```

### 4.4 这个缺陷对 parent 分组的直接影响

10 篇文档的 parent 分组数实测:

| artifact_id | parents | heading 总数 | 分组 |
|---|---|---|---|
| doc_af05975673c5 | 2 | 7 | `(文档开头)`, `The anatomy of a skill` |
| doc_1d0b6b72bdf2 | 2 | 18 | `(文档开头)`, `What are agents?` |
| doc_723b16a48c5d | 2 | 7 | `(文档开头)`, `Context engineering vs. ...` |
| doc_5c3e74c436ef | 2 | 15 | `(文档开头)`, `A note on simply using a longer prompt` |
| doc_978f022cd39f | 2 | 19 | `(文档开头)`, `Introduction` |
| doc_3ce065f25856 | 1 | 0 | `Get the developer newsletter` |
| ... | | | |

**几乎所有文档都只有 2 个 parent**——意味着 `_scope()` 的"按 section_path[0] 分组"实际上退化成了"把整篇文档当一个大组"。这正是 129 个 chunk 只对应 17 个 parent 的原因。

> ⚠️ 这是后续做 Contextual Retrieval 时的重要背景:标题层级抽取不准 → parent 划分过粗 → 失去"同章节上下文"这个可利用的信号。

---

## 5. Block 到 Chunk:fragments 锚点

Block 化之后,chunker 才登场。**Chunk 引用 Block 的方式是 `ChunkFragment`**(`chunker.py:21`):

```python
@dataclass
class ChunkFragment:
    block_id: str      # 引用哪个 block
    start_char: int    # 在该 block.text 内的起点(Unicode 码点, [start, end) 左闭右开)
    end_char: int
    block_type: str
    text: str          # 冗余存一份, 便于直接展示
```

**核心不变式(实测全部成立)**:

```
fragment.text == block.text[fragment.start_char : fragment.end_char]
```

实测 `doc_5c3e74c436ef_c0001`(79 个 fragment 零误差):

```json
fragments: [
  {"block_id": "b0001", "start_char": 0, "end_char": 273, "block_type": "text"},
  {"block_id": "b0002", "start_char": 0, "end_char": 445, "block_type": "text"},
  {"block_id": "b0003", "start_char": 0, "end_char": 441, "block_type": "text"},
  {"block_id": "b0004", "start_char": 0, "end_char": 172, "block_type": "text"}
]
```

验证:block b0001 全长 273,`text[0:273] == fragment.text` → **True**

### 5.1 一个 Block 可以被"部分切"进多个 Chunk

这是 block 结构里最反直觉的一点。真实案例(`doc_723b16a48c5d`,text block b0023 长 562 字符):

```
c0005:  b0023 [  0, 562)   ← 完整内容
c0006:  b0023 [305, 562)   ← 只有尾部 257 字符
```

原因是 `chunker.py:149-175` 的 `_split_text_block()`:text 块超过 `chunk_tokens` 时,用 `SentenceSplitter` 按句切开;然后 `_tail_overlap()` 会把上一 chunk 的尾巴(默认 400 token)拎出来做 overlap,拼进下一 chunk 的开头。

实测统计:**644 个 block 中,有 78 个 block 被切进了多个 chunk**。

这直接推导出一条重要结论:

> **gold 标注的粒度必须是"证据片段" `(block_id, [start_char, end_char))`,而不是整个 `block_id`。**
> 因为同一个 `block_id` 的不同字符区间,落在不同 chunk 里,内容也不同。

---

## 6. 完整结构层次图

```
DocumentArtifact  (doc_5c3e74c436ef)
├── artifact_id     文档级
├── title / source_uri / parser / language
├── quality         {score, status, metrics...}
├── index_decision  {tier, dense_index, sparse_index...}
└── blocks[78]      DocumentBlock 列表  ← block 化的产物
     ├── b0001 text    sec=[]                     order=1
     ├── b0002 text    sec=[]                     order=2
     ├── b0005 heading sec=[]  lv=3               order=5
     ├── b0006 text    sec=['A note...']  parent=b0005
     ├── ...
     └── b0078 code    sec=[]  meta={source: html_code_supplement}

        ↓ BlockAwareHierarchicalChunkBuilder.build()

ParentNode (doc_5c3e74c436ef_p0001)      ← 按 section_path[0] 分组
└── child_chunk_ids: [c0001, c0002, ...]
    full_text: 2103 chars / 427 tokens

IndexReadyChunk (doc_5c3e74c436ef_c0001)  ← 索引单元
├── chunk_id / artifact_id / parent_id
├── fragments[4]    ← 指向 block 的字符区间 ★ 锚点
├── full_text       带 markdown 标记(用于展示)
├── dense_text      纯文本(用于 embedding)
├── sparse_text     去掉 heading(用于 BM25)
├── token_count=261 / char_count=1331
└── point_id → 1850931163052542526      ← Qdrant 主键
```

---

## 7. 实测数据总览(10 篇正式语料)

```
文档数            10
parser 分布       readability_structured × 7, trafilatura × 3
block 总数        644
  其中 text       466  (72.4%)
       heading     95  (14.8%)
       image       32  ( 5.0%)
       list        30  ( 4.7%)
       code        21  ( 3.3%)
block_id/order   零脱钩(644/644)
fragment 不变式   全部成立
chunk 总数        129
parents 总数      17         ← 平均每篇仅 1.7 个,分组过粗
跨 chunk 共享     78 个 block 被切进多个 chunk
```

---

## 8. 从中得出的三个工程结论

### 结论 1:block_id 可以用,但不能当 gold 锚点

`block_id` = `b{order:04d}`,纯位置序号。以下任一变更都会让它错位:

- 换 parser(同是 HTML,trafilatura ↔ readability 的分支切换就会改变 block 划分)
- cleaner 多过滤/少过滤任何一块(`_BOILERPLATE` 规则调整)
- `html_code_supplement` 补丁的触发与否
- 修掉 `html.py:116` 用 `len(blocks)` 而非 `order` 的问题

**所以 gold 必须锚在原文文本上**(即 `fragments` 的字符区间),而非任何 ID。

### 结论 2:heading 层级抽取是当前的结构瓶颈

顶层标题从 lv3 开始、补充代码块 `section_path` 为空、parent 分组退化成 2 个 —— 这三件事同源。

**它们共同导致了"同章节上下文"这一信号在实践中几乎不可用。** 这对后续 Contextual Retrieval 是个明确的前置改进项:如果章节归属本身不可信,那么"给 chunk 注入章节上下文"就失去了依据。

### 结论 3:补充代码块的位置需要修正

`html.py:111-123` 的两个问题:
- `block_id=f"b{len(artifact.blocks) + 1:04d}"` 应该与 `next_id()` 一致(用 `order`),避免未来与 cleaner 脱钩
- 补进的块应插到它原本的章节位置,并继承 `section_path` / `parent_id`,而不是无脑 append 到尾部

当前不影响正确性(因为 cleaner 恰好没过滤这些文档),属于**潜伏问题**。

---

## 附:相关源码位置索引

| 环节 | 文件:行 |
|---|---|
| 数据模型 | `ingest/models.py:38` (DocumentBlock) / `:88` (DocumentArtifact) |
| 格式路由 | `ingest/router.py:20` |
| HTML 抽取双路对比 | `ingest/parsers/html.py:84-102` |
| 代码块补充(潜伏问题) | `ingest/parsers/html.py:111-123` |
| Markdown → Block 状态机 | `ingest/parsers/markdown.py:29-148` |
| block_id 生成 | `ingest/parsers/markdown.py:37` |
| section_path 维护 | `ingest/parsers/markdown.py:118-140` |
| 清洗与 order 重排 | `ingest/cleaner.py:29-60` |
| 质量门 | `ingest/quality.py:15` / 入库决策 `:89` |
| 原子/可切类型定义 | `ingest/chunker.py:16-18` |
| 文本块切分 | `ingest/chunker.py:149-175` |
| 尾部 overlap 切片 | `ingest/chunker.py:303-366` |
| fragment 构造 | `ingest/chunker.py:222-228` |
| section 分组(_scope) | `ingest/chunker.py:201-212` |
| 编排顺序 | `ingest/pipeline.py:55-87` |
