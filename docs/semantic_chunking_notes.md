# LangChain 分块专栏

> 本文档只讲 **LangChain 这一套的落地细节**: 抽象、参数、我们的封装、实测数据。
>
> **与框架无关的原理**(分块要解决什么问题、五步流程、断点判定方法、窗口的两种角色、
> 工程结论、策略全景、相关策略) 全部移到了 `rag_chunking_concepts.md`。
> 建议先读通用文档, 再看本文的落地细节。
>
> LlamaIndex 的对应实现见 `llamaindex_chunking_notes.md`。

---

## 一、LangChain 在分块上的抽象

### 1.1 TextSplitter 接口

LangChain 的分块抽象是 `TextSplitter`, 继承自 `BaseDocumentTransformer`。

```text
langchain_text_splitters.base.TextSplitter
  └─ langchain_core.documents.transformers.BaseDocumentTransformer
       └─ abc.ABC
```

四个核心方法:

```python
split_text(self, text)                      -> list[str]
create_documents(self, texts, metadatas)    -> list[Document]
split_documents(self, documents)            -> list[Document]
transform_documents(self, documents)        -> Sequence[Document]
```

`split_documents()` 是 `split_text()` 的文档级包装: 它对每个 `Document` 调 `split_text`,
再把原文档的 metadata 复制到切出来的每一块上。

### 1.2 可用的 splitter 清单

`langchain-text-splitters` 里实际导出的类:

```text
CharacterTextSplitter                    RecursiveCharacterTextSplitter  ← 最常用
TokenTextSplitter                        SentenceTransformersTokenTextSplitter
MarkdownHeaderTextSplitter               MarkdownTextSplitter
ExperimentalMarkdownSyntaxTextSplitter
HTMLHeaderTextSplitter                   HTMLSectionSplitter
HTMLSemanticPreservingSplitter           RecursiveJsonSplitter
LatexTextSplitter                        PythonCodeTextSplitter
JSFrameworkTextSplitter                  NLTKTextSplitter
SpacyTextSplitter                        KonlpyTextSplitter(韩语)
```

外加 `langchain-experimental` 里的:

```text
SemanticChunker        ← 本文重点
```

### 1.3 Document 结构

LangChain 的 `Document` 很简单, 只有两个字段:

```python
doc.page_content    # 正文
doc.metadata        # 元数据 dict
```

**关键: metadata 默认不参与嵌入。**

`VectorStore.add_documents()` 内部做的是:

```python
embed_documents([doc.page_content for doc in documents])
```

**只嵌 `page_content`, metadata 只是随文档一起存下来。** 所以 LangChain 里不会出现
"元数据意外混进向量"的问题 —— 也正因为如此, 它没有"排除列表"这种机制。

要理解这句话为什么重要, 对比一下 LlamaIndex:

```text
LangChain Document   metadata 默认不进嵌入
LlamaIndex Node      metadata 默认【会】拼进嵌入文本,
                     必须用 excluded_embed_metadata_keys 显式排除
```

这个差异直接决定了"单句嵌入 + 窗口元数据"这种模式能不能原生实现 ——
LlamaIndex 有 `Node` 的排除列表才做得出 SentenceWindow, LangChain 需要在应用层自己拼。

### 1.4 和 LlamaIndex Node 的完整对照

| 能力 | LangChain `Document` | LlamaIndex `Node` |
|---|---|---|
| 正文 | `page_content` | `text` / `get_content()` |
| 元数据 | `metadata` | `metadata` |
| 元数据进嵌入 | **默认不进** | **默认进, 可排除** |
| 前后节点关系 | 无 | `relationships`(NEXT/PREV/PARENT/CHILD) |
| 内容可替换 | 无(直接赋值) | `set_content()` |
| 唯一 ID | 无内置 | `node_id` |
| 分块抽象 | `TextSplitter` | `NodeParser` |
| 批量切分 | `split_documents()` | `get_nodes_from_documents()` |

多出来的 `excluded_*_metadata_keys`、`relationships`、`set_content()` 这三样,
正是 SentenceWindow 那套机制能在 LlamaIndex 里原生跑通、而在 LangChain 里得自己搭的原因。

---

## 二、SemanticChunker 的核心参数

| 参数 | 作用 | 默认值 |
|---|---|---|
| `embeddings` | 嵌入函数(**必填**), 需要实现 `embed_documents` / `embed_query` | — |
| `buffer_size` | 窗口半径。**可以是 0**(退化成单句嵌入) | 1 |
| `breakpoint_threshold_type` | 断点判定方法 | `percentile` |
| `breakpoint_threshold_amount` | 阈值松紧 | 95 / 3 / 1.5 / 95 |
| `number_of_chunks` | 直接指定块数(走反向插值) | None |
| `sentence_split_regex` | 句子切分正则 | `(?<=[.?!])\s+` |
| `min_chunk_size` | 小于此长度的块会被合并掉 | None |
| `add_start_index` | 是否在 metadata 里记录起始位置 | False |

四个方法的默认参数(`BREAKPOINT_DEFAULTS`):

```text
percentile           95
standard_deviation   3
interquartile        1.5
gradient             95
```

**它没有 `chunk_overlap` 参数** —— 语义分块的输出是划分, 不是滑动切分。

---

## 三、我们的封装: 自定义了什么

这一节要说清楚"哪些是 LangChain 的, 哪些是我们加的", 避免把两边的经验混为一谈。

### 3.1 分块算法: 完全没改

`SemanticChunker` 整个类就是 LangChain 提供的。窗口拼装、嵌入、算距离、找断点、成块
—— 这些逻辑一行都没动, 我们只是实例化它。

### 3.2 句子切分: 只替换了一个正则字符串

```python
CN_SENTENCE_SPLIT_REGEX = r"(?<=[。！？；])\s*(?=\S)"    # 我们写的

chunker = SemanticChunker(
    embeddings,
    buffer_size=buffer_size,
    breakpoint_threshold_type=threshold_type,
    breakpoint_threshold_amount=amount,
    sentence_split_regex=CN_SENTENCE_SPLIT_REGEX,        # 传进去
)
```

LangChain 内部怎么用它:

```python
def _get_single_sentences_list(self, text: str) -> List[str]:
    return re.split(self.sentence_split_regex, text)
```

**我们只提供了一个正则字符串, 切分动作还是库在执行。**

### 3.3 和 LlamaIndex 的封装方式差别很大

```text
LangChain 版    替换【参数】—— 传一个正则字符串
                切分动作: re.split(我们的正则, text)      ← 库执行

LlamaIndex 版   替换【整个函数】—— 传一个 callable
                切分动作: 完全由我们的 zh_splitter 接管
```

```python
# LangChain: 只能给正则
"sentence_split_regex": CN_SENTENCE_SPLIT_REGEX

# LlamaIndex: 给整个函数
def zh_splitter(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？；])", text) if s.strip()]

"sentence_splitter": zh_splitter
```

**能力边界不一样**:

- LangChain 给正则 -> 只能控制"在哪切", 切完的**后处理**(去空白、过滤空片段)动不了
- LlamaIndex 给函数 -> 整个切分行为都能接管

这也解释了我们为什么在 LangChain 那边要写 `(?=\S)` 这种绕的技巧 —— 因为没有后处理的机会,
只能在正则里用"后面还有非空白内容"这个条件变通。LlamaIndex 那边函数里一句 `if s.strip()` 就解决了。

### 3.4 嵌入函数: 我们实现了两个

两个都实现 LangChain 的 `Embeddings` 接口(只要 `embed_documents` 和 `embed_query`):

| 类 | 性质 |
|---|---|
| `HashingEmbeddings` | 手写字符 n-gram 哈希, 零依赖零成本。**不是语义模型**, 只用于离线验证链路 |
| `QwenVLEmbeddings` | 阿里云百炼 `qwen3-vl-embedding` 的封装, 走多模态向量原生接口 |

`QwenVLEmbeddings` 内部做了三件事: 按内容缓存(避免重复嵌入相同窗口)、分批(每批 10 条)、
批量失败时自动降级为逐条调用。

### 3.5 分析脚手架: 生产不会这么写

`main()` 里的四种方法对比、参数扫描, 以及 `diagnose()` 函数, 都是**学习工具**。

`diagnose()` 调的是库的私有方法:

```python
chunker._get_single_sentences_list(text)
chunker._calculate_sentence_distances(sentences)
```

下划线开头是私有约定, 库升级时可能改名。**生产代码不要依赖它们。**

---

## 四、实测数据

语料 A: 10 句 / 251 字。语料 B: PDF 解析出的 274 句 / 17123 字。
嵌入模型: 百炼 `qwen3-vl-embedding`(1024 维)。

### 4.1 语料 A(10 句)

**距离序列**(`buffer_size=1`):

```text
d(0,1)=0.0342   d(1,2)=0.1676   d(2,3)=0.1182   d(3,4)=0.1064
d(4,5)=0.1700   d(5,6)=0.0611   d(6,7)=0.1075   d(7,8)=0.0741   d(8,9)=0.0752

阈值(percentile, 95) = 0.1690
越线: 只有 d(4,5)=0.1700  ->  切在 S5|S6
```

**四种方法对比**:

```text
percentile            阈值 0.1690   ->  2 块   切 S5|S6
standard_deviation    阈值 0.2320   ->  1 块   一刀没切
interquartile         阈值 0.1678   ->  2 块   切 S5|S6
gradient              阈值 0.0969   ->  2 块   切 S1|S2  (假断点)
```

注意 gradient 的阈值 0.0969 **作用在梯度数组上**, 不能和另外三个比大小。

**buffer_size 的影响**:

```text
buffer_size   窗口宽度   距离范围          阈值      切点
──────────────────────────────────────────────────────────
     0         1 句     0.2248 ~ 0.4747   0.4692   S5|S6
     1         3 句     0.0342 ~ 0.1700   0.1690   S5|S6
     2         5 句     0.0099 ~ 0.1372   0.1324   S6|S7
     3         7 句     0.0099 ~ 0.0914   0.0850   S4|S5
```

**阈值松紧的影响**(`buffer_size=1`):

```text
amount=50  ->  5 个 chunk
amount=80  ->  3 个 chunk
amount=90  ->  2 个 chunk
amount=95  ->  2 个 chunk
amount=99  ->  2 个 chunk
```

### 4.2 语料 B(274 句)

**四种方法终于分化**:

```text
percentile            ->  15 块
standard_deviation    ->   5 块
interquartile         ->  11 块
gradient              ->  15 块
```

10 句时 percentile 和 interquartile 结果完全一样, 274 句时变成 15 vs 11 ——
**样本量决定了这些统计方法有没有区分度。**

**块大小失控**(同一套参数):

```text
amount=50  -> 137 个 chunk   字数 min=6     max=2901   均值=125
amount=80  ->  56 个 chunk   字数 min=19    max=2949   均值=309
amount=90  ->  29 个 chunk   字数 min=36    max=3304   均值=598
amount=95  ->  15 个 chunk   字数 min=50    max=4426   均值=1158
amount=99  ->   4 个 chunk   字数 min=3264  max=6015   均值=4348
```

**number_of_chunks 直接指定块数**:

```text
number_of_chunks=3  ->   3 个 chunk   字数 min=4074  max=7305
number_of_chunks=5  ->   5 个 chunk   字数 min=670   max=6015
number_of_chunks=8  ->   8 个 chunk   字数 min=50    max=5009
```

注意它的实现是**反向插值**: 按块数反推一个百分位阈值。所以它并不保证精确切出指定块数,
只是在逼近。

---

## 五、LangChain 特有的坑

这几条是**这个实现的选择**, 换框架就不成立, 不能当通用规律记:

**1. gradient 模式的下标语义变了**

`_calculate_breakpoint_threshold()` 在 gradient 分支返回的是**梯度数组**, 而
`split_text()` 拿它的下标去切句子:

```python
distance_gradient = np.gradient(distances, range(0, len(distances)))
return np.percentile(distance_gradient, amount), distance_gradient
```

所以切点落在"梯度最大"的位置, 不是"距离最大"的位置。

**2. interquartile 用均值而不是 Q3**

```python
q1, q3 = np.percentile(distances, [25, 75])
iqr = q3 - q1
return np.mean(distances) + amount * iqr        # 注意是 mean, 不是 q3
```

标准箱线图规则是 `Q3 + 1.5×IQR`。库里用 `mean`, 在分布左偏时会给出**更低**的阈值
(更容易切)。对照其他框架时数值会对不上。

**3. 默认句子切分正则只认英文**

```python
sentence_split_regex: str = r"(?<=[.?!])\s+"
```

**4. 输出没有 overlap** —— 没有 `chunk_overlap` 参数, 输出是划分而非滑动切分。

**5. 9 个距离值时 percentile 永远只切 1 刀**

```text
9 个点, amount=95
-> 位置 = 0.95 × (9-1) = 7.6
-> 在排序后的第 7 和第 8 个值之间插值
-> 阈值紧贴最大值
-> 只有最大值越线
```

这是数学必然, 不是 bug。想让切分更细, 只能降 amount。

---

## 六、LangChain 定位与可迁移性

### 6.1 哪些是 LangChain 原生能力

| 能力 | 类 / 函数 | 来源包 |
|---|---|---|
| 语义分块 | `SemanticChunker` | langchain-experimental |
| 递归字符分块 | `RecursiveCharacterTextSplitter` | langchain-text-splitters |
| 嵌入接口 | `Embeddings` | langchain-core |
| 文档对象 | `Document` | langchain-core |

### 6.2 哪些是我们补的

| 组件 | 位置 | 性质 |
|---|---|---|
| 中文句子切分正则 | `CN_SENTENCE_SPLIT_REGEX` | 参数覆盖, 不是新算法 |
| 本地哈希嵌入 | `HashingEmbeddings` | 教学替身 |
| 百炼多模态嵌入 | `QwenVLEmbeddings` | 接口实现 |
| PDF 解析与清洗 | `pipeline/pdf_loader.py` | 前置预处理 |
| 四方法对比 / 参数扫描 | `main()` | 分析脚手架 |
| `diagnose()` | 同上 | 调用了库的私有方法 |

### 6.3 时效风险

安装 `langchain-experimental` 时就会看到:

```text
DeprecationWarning: `langchain-experimental` is being sunset
and is no longer actively maintained.
```

`SemanticChunker` 本体只有约 60 行, 依赖的 `combine_sentences` 和
`calculate_cosine_distances` 各几十行。**如果这个包停止维护, 把这三段抄进自己的代码库
是可行且推荐的做法。**

这也是为什么要把"概念层"和"实现层"分开写 —— 概念层是资产, 实现层是可替换的。

### 6.4 迁移到其他框架

LlamaIndex 的 `SemanticSplitterNodeParser` 是同一套逻辑, 参数名不同:

```text
本文(LangChain)                          LlamaIndex
──────────────────────────────────────────────────────────
buffer_size                              buffer_size
breakpoint_threshold_amount              breakpoint_percentile_threshold
embeddings                               embed_model
sentence_split_regex                     自定义 sentence_splitter 函数
```

概念上可以直接对应过来, 细节见 `llamaindex_chunking_notes.md`。

---

## 七、代码与文件

```text
semantic_chunker_demo.py    语义分块主程序
                            --backend qwen / local / openai
                            --pdf / --md 指定语料
pdf_loader.py               MinerU 调用 + Markdown 清洗
sentence_window_demo.py     SentenceWindow 检索对照实验
check_dashscope_key.py      百炼 API Key 诊断
pipeline/corpus/                     PDF 解析出的中间 Markdown
```

运行:

```powershell
# 需要环境变量 DASHSCOPE_API_KEY
python semantic_chunker_demo.py                          # 内置 10 句示例
python semantic_chunker_demo.py --md pipeline/corpus/extracted.md # 用已有 Markdown
python semantic_chunker_demo.py --pdf 文件.pdf --pdf-pages 1-20
python semantic_chunker_demo.py --backend local          # 离线(纯词法, 仅验证逻辑)
```

MinerU CLI 安装:

```powershell
uv tool install mineru-open-api
```

注意 `flash-extract` 模式会把文档上传到 mineru.net 处理, 限 10MB / 20 页;
敏感材料应改用本地部署的 MinerU。

