# 格式路由与格式清洗方案

> 目标: 把 URL/HTML、中文 PDF、英文 PDF、扫描件和 Markdown 统一成
> 可追溯、可质量评估、可被 LlamaIndex 继续切分和检索的文档结构。
>
> 当前状态:
> - 已实现 Markdown / MinerU PDF 入口
> - 已实现 LlamaIndex FixedSemanticNodeParser
> - Phase 0 已实现 URL/HTML/Markdown/PDF 路由、DocumentBlock 输出、Raw Quality Gate
> - 已实现 Block-aware Hierarchical ChunkBuilder 的 Parent/Leaf 和 fragments 输出
> - 已实现 Phase 0 JSON -> Qdrant dense/sparse 入库和稳定 point id 映射
> - Docling、Office、图片多模态和质量校准仍是规划内容

---

## 一、设计目标

```text
1. 支持 URL、HTML、PDF、扫描件、Markdown、Office
2. 不同格式进入不同的专用解析器
3. 最终统一成 DocumentArtifact / DocumentBlock
4. 保留代码块、数学公式和图片
5. 扫描件有 OCR 置信度和质量门控
6. 低质量 OCR 不污染 sparse/BM25
7. 后续仍由 LlamaIndex 负责 chunking
8. 每个结论都能追溯来源、页码、section 和 parser
```

非目标:

```text
1. 第一阶段不训练自己的 OCR/公式识别模型
2. 第一阶段不做任意 PDF 的完美版面还原
3. 第一阶段不把所有图片都做多模态向量化
4. 第一阶段不直接替换现有 chunk/retrieval pipeline
```

---

## 二、总体架构

```text
Source
  |
  v
Source Fetcher
  |
  v
Format Detector
  |
  v
Format Router
  |
  +----------------------+----------------------+
  |                      |                      |
  v                      v                      v
HTML Parser         PDF Parser             Office Parser
Trafilatura         MinerU/Docling         MarkItDown
  |                      |                      |
  +----------------------+----------------------+
                         |
                         v
              RawDocumentArtifact
                         |
                         v
             Raw Parse Quality Gate
             |
             +-- accept -------------------------+
             |                                   |
             +-- fallback parser --+             |
             |                     |             |
             +-- re-OCR -----------+-- retry ----+
             |                                   |
             +-- manual review                     |
                                                 v
                                         Block Cleaner
                                                 |
                                                 v
                                        CleanDocumentArtifact
                                                 |
                                                 v
                                          Index Quality Gate
                                                 |
                        +------------------------+------------------------+
                        |                        |                        |
                        v                        v                        v
                dense + sparse           dense + sparse low       dense only / review
                        |                        |                        |
                        +------------------------+------------------------+
                                                 |
                                                 v
                                      LlamaIndex NodeParser
                                                 |
                                                 v
                                          IndexReadyChunk
                                                 |
                          +----------------------+----------------------+
                          |                                             |
                          v                                             v
                    dense_text                                     sparse_text
                          |                                             |
                          +----------------------+----------------------+
                                                 v
                                          same chunk_id
```

三个核心对象:

```text
RawDocumentArtifact
    Parser 原始输出
    保留原文、页码、bbox、OCR 原始结果和 parser 版本

CleanDocumentArtifact
    Block Cleaner 清洗后的结构化文档
    保留 code / formula / image / section / provenance

IndexReadyChunk
    同一 chunk_id 下的 dense_text / sparse_text / full_text
```

---

## 二点一、Block 和 Chunk 的关系

不要把 Block 和 Chunk 设计成一对一。

```text
DocumentArtifact
    |
    v
Block 1..N
    |
    v
Chunk 1..N
```

关系是:

```text
一个 Document 包含多个 Block
一个 Chunk 可以包含一个或多个 Block
一个文本 Block 可以因为 overlap 出现在多个 Chunk 中
一个超长 Block 也可以被拆成多个 Block 片段
```

示例:

```text
Block B1: heading "2.1 负载均衡"
Block B2: paragraph "Keepalived 使用 VRRP..."
Block B3: code "location / { proxy_pass ... }"
Block B4: formula "E = mc^2"
Block B5: image + caption

Chunk C01:
    B1 + B2

Chunk C02:
    B2 + B3 + B4

Chunk C03:
    B4 + B5
```

这意味着:

```text
B2 同时属于 C01 和 C02
B4 同时属于 C02 和 C03
```

这不是重复错误, 而是 overlap 或相邻上下文扩展的结果。

### Block 的职责

```text
保留 parser 输出结构
标记 heading/text/code/formula/table/image
保留 page/bbox/section
保留 OCR confidence 和质量 flags
提供稳定来源和 provenance
```

### Chunk 的职责

```text
成为 dense/sparse 的实际文档单位
控制 token 大小和 overlap
保持主题和章节边界
生成 dense_text / sparse_text / full_text
成为 Qdrant point_id
成为 RRF 和 rerank 的比较单位
```

### 推荐 Chunk 结构

```json
{
  "chunk_id": "artifact_001_c017",
  "artifact_id": "artifact_001",
  "block_ids": ["B4", "B5", "B6"],
  "section_path": ["第 2 章", "2.1 负载均衡"],
  "page_start": 7,
  "page_end": 8,
  "token_count": 612,
  "full_text": "...",
  "dense_text": "...",
  "sparse_text": "...",
  "quality": {
    "score": 0.82,
    "flags": ["partial_low_ocr"]
  },
  "index_decision": {
    "dense_index": true,
    "sparse_index": false,
    "sparse_weight": 0.0
  }
}
```

### Chunk 是 Block 的索引视图

正确关系:

```text
Document
    |
    v
Block tree
    |
    v
Chunk view
```

Block 保存文档结构:

```json
{
  "block_id": "b13",
  "type": "paragraph",
  "text": "MySQL 是一种关系型数据库...",
  "parent_id": "b12",
  "order": 13,
  "page": 5,
  "section_path": ["第 3 章", "3.1 MySQL"],
  "quality": {
    "score": 0.95,
    "flags": []
  }
}
```

Chunk 只保存检索视图:

```json
{
  "chunk_id": "c01",
  "section_block_id": "b12",
  "fragments": [
    {"block_id": "b12", "start": 0, "end": 9},
    {"block_id": "b13", "start": 0, "end": 42},
    {"block_id": "b14", "start": 0, "end": 36}
  ],
  "full_text": "3.1 MySQL\n\nMySQL 是一种关系型数据库...\n它支持事务...",
  "dense_text": "3.1 MySQL\n\nMySQL 是一种关系型数据库...\n它支持事务...",
  "sparse_text": "MySQL 关系型数据库 事务"
}
```

不要只保存 `block_ids`。超长段落可能只被 Chunk 使用一部分，overlap 也可能只复用某个 Block 的尾部。

因此推荐使用:

```text
fragments = [
    block_id
    start_char
    end_char
]
```

其中 `start_char/end_char` 统一表示 `block.text` 内的字符范围，使用半开区间 `[start_char, end_char)`。Token 数只通过 `token_count` 单独记录，不能作为 Fragment 的定位单位。原子 Block 可以整块引用，普通文本 Block 可以引用局部范围。

### Heading 与 Sparse 的注意事项

如果每个 Chunk 都把 section heading 拼进 `sparse_text`:

```text
KEEPALIVED
KEEPALIVED
KEEPALIVED
...
```

会导致该词 `df` 很高、IDF 很低，失去区分度。

建议:

```text
heading
    进入 full_text / dense_text / LLM metadata

sparse_text
    默认只保留正文高信息词

如果确实需要标题权重
    -> 使用 BM25F 字段权重
    -> 不要让标题文本重复进入普通 BM25 文档正文
```

### Chunk Builder 的规则

```text
1. 以顶层 section 作为默认硬边界
2. 按 token 目标长度累积 Block
3. 不切断 code/formula/table/image 等原子 Block
4. 在文本 Block 内部寻找语义切点
5. overlap 通过回退 Block 或句子实现
6. 从 Block 继承 page/section/confidence/provenance
7. 为同一个 chunk 生成 full/dense/sparse 三类文本投影
8. 最终输出 LlamaIndex TextNode / IndexReadyChunk
```

因此最终分工是:

```text
Parser
    负责尽量正确地把格式转成 Block

Block Cleaner
    负责修复和规范化 Block

Chunk Builder
    负责把 Block 组织成检索单元

Index Quality Gate
    负责决定 Chunk 是否进入 dense/sparse

RAG pipeline
    负责同一个 chunk_id 的 dense/sparse/RRF/rerank/generation
```

### 与固定长度 + 语义切分的关系

Block-aware ChunkBuilder 不是替代固定长度和语义切分, 而是把两者
放进更外层的结构约束中:

```text
顶层 section
    作为默认硬边界

code / formula / table / image
    作为原子 Block, 默认不切断

text / list / paragraph
    允许参与 token 累积和语义切分

heading
    默认和后续正文一起进入同一个 Chunk
```

执行顺序:

```text
1. 按 section 分组
2. 按 Block 顺序累积 token
3. 在接近 800 token 时确定硬上限
4. 如果当前 Block 是可分割文本:
       在边界窗口内寻找语义断点
5. 如果当前 Block 是 code/formula/table/image:
       不切断, 直接放入对应 Chunk
6. 通过回退句子或 Block 生成 overlap
7. 继承 Block metadata, 生成 IndexReadyChunk
```

需要特别注意的冲突:

```text
原子 Block 大于 chunk_size
    -> 允许成为 oversized chunk, 并标记
    -> 或者使用 code-aware / formula-aware 的子切分器

heading 和正文被切散
    -> 强制 heading 跟随后续正文

overlap 跨过多个 Block
    -> 记录 overlap_block_ids, 避免重复 metadata 污染

低质量 OCR Block 跨多个 Chunk
    -> 每个 Chunk 独立做 Index Quality Gate
```

最终原则:

```text
Block 给出结构边界和原子性;
token 目标给出长度约束;
语义距离只负责文本内部的边界微调;
同一个 Chunk 继续提供 dense_text / sparse_text / full_text。
```

### Block-native Chunking 策略

当解析结果已经变成 Block 后, 顶层策略不应该继续是:

```text
整篇文本 -> 固定长度 -> 语义微调
```

而应该改成:

```text
Document
  -> section
  -> Block 序列
  -> 结构分组
  -> token 目标约束
  -> 只在长文本 Block 内做语义切分
  -> Leaf Chunk
  -> Parent Section Node
```

推荐采用 **Block-aware Hierarchical Chunking**:

```text
Parent node
    顶层 section / 章节
    保存完整 section 上下文
    主要用于：
        结果扩展
        引用
        送给 LLM 的更大上下文

Leaf node
    实际进入 dense/sparse 的检索 chunk
    由 heading + paragraph + code + formula + image 组合产生
```

示例:

```text
Parent P1 = 第 2 章 KEEPALIVED 概述

Leaf C01 = 2.1 + 基本说明
Leaf C02 = 2.1 配置示例 + code
Leaf C03 = 2.2 + 调度说明 + formula
Leaf C04 = 2.2 表格 + image caption
```

关系:

```text
P1 -> C01
P1 -> C02
P1 -> C03
P1 -> C04
```

检索阶段:

```text
先用 Leaf Chunk 做 dense/sparse 匹配
    -> RRF
    -> rerank
命中 Leaf 后按需扩展到 Parent Section
    -> 送给 LLM 做最终回答
```

### Block-native Chunk Builder 规则

```text
1. section 作为 Parent 边界
2. heading 进入 Leaf 的开头或 metadata
3. paragraph/list 可以合并或切分
4. code/formula/table/image 作为原子 Block
5. 接近 token 上限时优先在文本 Block 内切
6. overlap 使用尾块或尾句回退
7. 不把 Parent 和 Leaf 混成同一个检索点
8. Parent 可以只存摘要、标题路径和子节点关系
9. Leaf 是 dense/sparse 的唯一检索单位
```

### 和旧策略的关系

旧策略:

```text
固定长度 + overlap + 语义微调
```

新策略:

```text
结构优先 Block 分组
  + token 上限
  + 原子 Block 保护
  + 长文本内部语义微调
  + Parent/Leaf 两级结构
```

所以旧策略不是删除, 而是降级为:

```text
TextBlock 内部切分器
```

### 推荐实施顺序

```text
阶段 1
    Block-aware token packing
    section hard boundary
    atomic block 不切断
    text block 保持旧语义切分

阶段 2
    Parent/Leaf hierarchical nodes
    Leaf 检索 + Parent 扩展

阶段 3
    字段化 dense_text / sparse_text
    OCR quality / index decision

阶段 4
    多模态图片节点
```

---

## 二点二、Chunk 的形式化定义与 Block 分组依据

### Chunk 的定义

Chunk 不是“若干个 Block 的拼接结果”，而是:

```text
从一个 DocumentArtifact 中选出的、
有顺序的 Block fragment 集合，
用于作为一个统一检索单元。
```

形式化表示:

```text
Chunk = (
    chunk_id,
    artifact_id,
    scope_block_id,
    section_block_id,
    ordered_fragments,
    full_text,
    dense_text,
    sparse_text,
    quality,
    index_decision
)
```

其中:

```text
scope_block_id
    最外层硬边界, 默认对应顶层章节

section_block_id
    Chunk 的首个结构节点, 用于引用和 section 元数据

ordered_fragments
    有序 fragment 列表
    每个 fragment 都是 (block_id, start_char, end_char)
```

### Fragment 定义

```json
{
  "block_id": "b13",
  "start_char": 0,
  "end_char": 42
}
```

区间使用半开区间:

```text
[start_char, end_char)
```

原子 Block 通常引用完整范围:

```json
{
  "block_id": "b15",
  "start_char": 0,
  "end_char": 31
}
```

普通文本 Block 可以只引用一部分:

```json
{
  "block_id": "b13",
  "start_char": 80,
  "end_char": 180
}
```

### 字符偏移单位与坐标系

`start_char` 和 `end_char` 是 Fragment 的来源定位字段，统一使用字符偏移，不使用 token 偏移。

```text
单位:
    Unicode code point
    Python 中对应 str 下标

基准文本:
    CleanDocumentArtifact 中该 Block 的规范化 block.text

区间:
    [start_char, end_char)
    start_char 包含，end_char 不包含

截取公式:
    fragment_text = block.text[start_char:end_char]
```

因此必须满足:

```text
0 <= start_char < end_char <= len(block.text)
fragment.text == block.text[start_char:end_char]
```

Token 只作为预算和统计信息，不能替代字符偏移:

```text
token_count
    当前 Fragment 或 Chunk 的 token 数

start_token / end_token
    可选辅助字段
    依赖具体 tokenizer
    不能用于跨 embedding 模型稳定定位原文
```

同时区分三种互不混用的坐标:

```text
start_char / end_char
    Block 内部字符范围，用于 Fragment 溯源和重建

start_byte / end_byte
    UTF-8 等编码下的字节范围

source_locations
    PDF page / bbox / HTML DOM path / 原始 URL 等外部来源位置
```

跨语言实现时需要固定规则。这里的字符指 Unicode code point，对应 Python `len(str)` 和字符串切片；JavaScript 使用 UTF-16 index，遇到 BMP 之外字符时需要做索引转换。

### Block 分组的硬约束

这些条件必须满足:

```text
1. 所有 fragment 来自同一个 DocumentArtifact
2. fragments 按文档顺序排列
3. 同一个 fragment 内不交叉、不倒退
4. code/formula/table/image 默认作为原子 Block
5. 默认不跨越顶层 section 硬边界
6. 不出现空 Chunk
7. 必须能还原 full_text 和 block provenance
```

允许但有标记的例外:

```text
oversized_atomic_block
    code/formula/table 超过 token 上限

cross_scope_chunk
    跨顶层章节边界, 仅在明确策略允许时出现

overlap_fragment
    与相邻 Chunk 重叠的 fragment
```

### Block 分组的软目标

在满足硬约束后, 选择最优分组:

```text
最大化:
    主题一致性
    自包含程度
    section 完整性
    标题与正文的关联性

最小化:
    token 长度偏差
    边界语义跳变
    原子块破坏
    metadata 混杂
    无意义 overlap
```

可以用一个简化的目标函数理解:

```text
group_score =
    topic_coherence
  + self_containedness
  + heading_attachment
  - size_penalty
  - semantic_jump_penalty
  - atomic_break_penalty
  - cross_scope_penalty
```

### Chunk 边界的选择顺序

```text
1. 先找 section 硬边界
2. 再按 Block 顺序累积 token
3. 接近目标 token 上限时生成候选切点
4. 优先选择 Block 边界
5. 如果只能在文本 Block 内部切:
       在最后 window_size 范围内寻找语义断点
6. 绝不从 code/formula/table/image 中间切
7. 用尾部句子或 Block 生成 overlap
8. 重新计算 Chunk 的 full/dense/sparse 文本
9. 再做 Index Quality Gate
```

### 边界候选的优先级

```text
最高:
    顶层 section 结束

很高:
    heading 之前

高:
    paragraph/list 之间的自然边界

中:
    长 paragraph 内部的语义断点

禁止:
    code/formula/table/image 中间
```

### Chunk 的质量判断

质量判断必须在 Chunk 形成后进行:

```text
block_quality
    |
    v
chunk_quality
    |
    v
index_decision
```

因为同一个 Chunk 可能包含:

```text
高质量正文
低质量 OCR 图片文字
完整代码
损坏公式
```

所以最终要决定的是:

```text
这个 Chunk 是否进入 dense
这个 Chunk 是否进入 sparse
sparse vector 是否需要降权
是否需要人工复核
```

### 一句话定义

```text
Chunk 是：
在一个结构作用域内，
按文档顺序选择的 Block fragment 集合；
它必须保持代码/公式/表格/图片的原子性，
满足长度和 overlap 约束，
并在形成后通过 Index Quality Gate，
最终成为同一 chunk_id 下的 dense/sparse 检索单位。
```

### 为什么使用两层 Quality Gate

```text
Raw Parse Quality Gate
    在解析器刚输出 DocumentArtifact 后执行
    用于判断:
        是否改用 fallback parser
        是否需要重新 OCR
        页面是否解析失败
        文本层是否不可信

Block Cleaner
    用于:
        去 HTML boilerplate
        合并/拆分错误 block
        保留 code / formula / image
        做保守 OCR 纠错
        规范空白和标题层级

Index Quality Gate
    在 block 清洗后执行
    用于判断:
        哪些 chunk 可以进入 dense
        哪些 chunk 可以进入 sparse
        哪些 chunk 需要降权或人工复核
```

这样才能同时满足两个目标:

```text
解析质量差时，先尝试换 parser 或重新 OCR
清洗完成后，再决定是否进入 dense/sparse 索引
```

---

## 三、Route 判断顺序

不要只依赖文件扩展名。按下面顺序判断:

```text
1. 来源类型
    url / local_file / raw_bytes

2. HTTP Content-Type
    text/html / application/pdf / application/json / octet-stream

3. Magic bytes
    %PDF
    <!doctype html
    PK (Office zip)

4. 扩展名
    .html / .htm / .pdf / .md / .docx / .pptx / .xlsx

5. 内容嗅探
    HTML 标签密度
    PDF trailer
    Markdown 标题结构

6. PDF 画像
    是否有文本层
    扫描页比例
    中英文语言
    公式密度
    代码/配置块密度
```

输出:

```text
RouteDecision
    source_type
    mime_type
    confidence
    parser
    fallback_parsers
    reason
```

---

## 四、路由矩阵

| 输入 | 首选解析器 | 备选 | 重点 |
|---|---|---|---|
| URL 博客/文章 | Trafilatura | Readability | 正文、标题、作者、时间 |
| URL 代码文档 | Trafilatura + BeautifulSoup | Docling HTML | 保留 pre/code |
| 本地 Markdown | Markdown passthrough | Pandoc | 规范化，不重解析 |
| 中文 PDF / 扫描件 | MinerU | PaddleOCR / PyMuPDF | 中文 OCR、版面 |
| 英文通用 PDF | Docling | Marker / PyMuPDF | 阅读顺序、表格 |
| 英文论文/公式 | Docling | MinerU / GROBID | LaTeX、章节、引用 |
| 简单文本层 PDF | PyMuPDF | Docling | 低成本兜底 |
| Office | MarkItDown | Pandoc | Markdown 转换 |

---

## 五、统一 DocumentArtifact

所有 parser 的最终输出统一为:

```json
{
  "artifact_id": "doc_...",
  "source_uri": "https://...",
  "source_type": "url|pdf|scan|html|markdown|office",
  "mime_type": "text/html",
  "parser": "trafilatura",
  "parser_version": "x.y.z",
  "title": "...",
  "author": null,
  "published_at": null,
  "language": "zh|en|mixed",
  "metadata": {},
  "blocks": []
}
```

### DocumentBlock

```json
{
  "block_id": "b001",
  "type": "text|heading|code|formula|table|image|list",
  "text": "...",
  "markdown": "...",
  "code_language": null,
  "latex": null,
  "image_uri": null,
  "image_alt": null,
  "section_path": [],
  "page": null,
  "bbox": null,
  "order": 0,
  "ocr_confidence": null,
  "quality_flags": []
}
```

必须保留:

```text
block type
原始文本
正文 Markdown
代码语言
公式 LaTeX
图片路径
section path
page/bbox
parser + version
OCR 质量
```

---

## 六、清洗流程

### 6.1 HTML

```text
HTML
  -> 去 script/style/nav/footer/ad/cookie
  -> 提取正文
  -> 提取 pre/code
  -> 提取图片和 caption
  -> 规范标题层级
  -> Markdown
```

代码块规则:

```text
保留缩进
保留换行
保留语言标识
不使用 HTML 正文空白归一化
```

### 6.2 PDF

```text
PDF
  -> 判断文本层
  -> 直接文本抽取 OR OCR
  -> 识别 heading/body/code/formula/table/image
  -> 规范阅读顺序
  -> Markdown + blocks
```

### 6.3 扫描件

```text
扫描页
  -> 渲染页图
  -> OCR
  -> 置信度
  -> 规则清洗
  -> 质量门控
```

### 6.4 Markdown

```text
Markdown
  -> 解析成 block
  -> 保留 code fence / formula / image
  -> 标题层级规范化
  -> 再生成 Markdown
```

不要对已经是 Markdown 的文件做 HTML 去噪。

### 6.5 Office

```text
DOCX/PPTX/XLSX
  -> MarkItDown/Pandoc
  -> Markdown + table blocks
  -> 再进入统一清洗
```

---

## 七、代码块策略

代码必须作为独立 block:

```text
type = code
code_language = nginx/javascript/python/...
text = 原始代码
```

代码不能经过:

```text
普通正文空白合并
停用词过滤
词形归一化
HTML boilerplate cleaner
```

HTML 中优先提取:

```html
<pre><code class="language-nginx">...</code></pre>
```

PDF 中的代码:

```text
1. 优先信任 Docling/MinerU code block
2. 相似缩进行合并
3. 保留分号和花括号
4. 不把配置行误判成标题
```

---

## 八、数学公式策略

公式分 inline/display:

```text
inline formula
    $...$

display formula
    $$...$$
```

公式 block 保存:

```text
type = formula
latex = ...
text = OCR 原文(可选)
```

可选解析器:

```text
MinerU formula recognition
Docling formula enrichment
GROBID(TEI)
```

公式不能参与普通 HTML 空白/标点清洗。

---

## 九、图片策略

图片有三种角色:

```text
1. 装饰性图片
    丢掉，不索引

2. 信息性图片
    保留 image_uri + caption + OCR/描述

3. 扫描页面图
    保留原图，作为重新 OCR 的原始证据
```

图片 block:

```json
{
  "type": "image",
  "image_uri": "assets/img_001.png",
  "image_alt": "...",
  "caption": "...",
  "ocr_text": "...",
  "ocr_confidence": 0.82,
  "bbox": [0, 0, 800, 600],
  "page": 7
}
```

第一版图片处理:

```text
图片保留本地文件
图片 OCR 文本进入正文
caption/alt 进入正文
不把 base64 图片塞进 Markdown 文本
```

后续图片检索:

```text
图像 embedding
图文联合 embedding
caption 文本 embedding
```

---

## 十、OCR 质量门控

### 质量指标

```text
文本覆盖率
平均 OCR 置信度
乱码率
字典命中率
语言一致性
公式/代码识别率
多引擎一致性
```

### 路由策略

```text
高质量
    dense + sparse

中等质量
    dense + sparse 降权
    sparse 只使用高置信度 block

低质量
    dense only
    不进入 sparse
    保留原图和 raw OCR
    标记人工复核
```

### 重要约束

```text
低质量 OCR 不能直接污染 BM25 词表。
```

### 如何判断低质量 OCR

不能只看 OCR engine 给出的一个 confidence 值。更可靠的方法是组合多个信号:

```text
1. OCR token/line confidence
2. 低置信度 token 比例
3. 乱码字符比例
4. 语言/字典命中率
5. 文本覆盖率和版面完整度
6. 多 OCR 引擎一致性
7. 代码/公式结构是否可解析
8. 领域词典和术语命中率
```

粗略评分:

```text
block_score =
    w1 * mean_confidence
  + w2 * p10_confidence
  + w3 * dictionary_hit_rate
  + w4 * language_consistency
  + w5 * engine_agreement
  + w6 * layout_coverage
  - w7 * garbled_ratio
  - w8 * low_confidence_token_ratio
```

其中 `p10_confidence` 表示最差 10% 行的置信度, 用来防止少量严重错误被平均值掩盖。

硬失败标志:

```text
code_parse_failed
formula_parse_failed
text_layer_missing
page_render_failed
reading_order_broken
```

建议第一版阈值:

```text
high       score >= 0.85
medium     0.60 <= score < 0.85
low        0.35 <= score < 0.60
reject     score < 0.35 或存在 hard_fail
```

阈值必须通过人工抽样校准, 不能当成跨语料通用常数。

页面级和 Block 级的动作不同:

```text
页面级低质量
    -> fallback parser / re-OCR / VLM transcription

Block 级低质量
    -> 只影响该 Block 和包含它的 Chunk 的索引准入

Chunk 级低质量
    -> 决定 dense/sparse 是否建立以及 sparse_weight
```

---

## 十一、Chunk 字段投影

同一个 chunk 可以有多种文本投影:

```text
chunk_id = C17

full_text
    完整正文、代码、公式、图片 caption

dense_text
    适合语义 embedding 的清洗文本

sparse_text
    只保留高置信度、适合词项匹配的文本

llm_text
    full_text 或按照 block 类型组织的上下文

payload
    chunk_id
    source
    page
    section
    parser
    ocr_confidence
    block_types
    image_uris
```

RRF 和 rerank 必须使用同一个 chunk_id。

---

## 十二、计划分阶段实施

### Phase 0: 接口和基础路由

```text
定义 Source / RouteDecision / DocumentBlock / Artifact
实现本地 Markdown passthrough
实现 HTML -> Trafilatura
实现 PDF -> MinerU/Docling adapter
实现 parser registry
```

### Phase 1: 代码和公式

```text
HTML pre/code -> code block
PDF code block -> code block
公式 -> latex block
Markdown code fence/display formula -> block
```

### Phase 2: OCR 质量

```text
文本层检测
OCR confidence
低质量 block 标记
dense_text / sparse_text 投影
质量报告
```

### Phase 3: 图片

```text
图片提取和去重
caption / OCR 文本
image_uri 保存
后续多模态 embedding
```

### Phase 4: 评测

```text
解析覆盖率
代码保真率
公式保真率
OCR 质量分布
dense/sparse Recall@k
```

---

## 十三、目录规划

```text
pipeline/ingest/
├── models.py
├── router.py
├── source.py
├── quality.py
├── parsers/
│   ├── html_trafilatura.py
│   ├── pdf_mineru.py
│   ├── pdf_docling.py
│   ├── markdown.py
│   └── office.py
└── cleaners/
    ├── html.py
    ├── pdf.py
    ├── markdown.py
    └── blocks.py
```

解析器统一接口:

```python
class Parser:
    def supports(self, route) -> bool: ...
    def parse(self, source) -> DocumentArtifact: ...
```

这样格式路由器和解析器可以独立升级。

---

## 十四、第一版验收标准

```text
1. PDF 扫描件能够识别文本，并输出置信度
2. HTML 代码块能够保留
3. 公式能够保留 LaTeX
4. 图片不会丢失
5. 低质量 OCR 不进入 sparse
6. 所有 block 有 source/page/section/parser
7. 同一 chunk 的 dense/sparse 共享 chunk_id
8. 后续能把这些 block 继续交给 LlamaIndex NodeParser
```
