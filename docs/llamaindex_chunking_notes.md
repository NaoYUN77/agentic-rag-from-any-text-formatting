# LlamaIndex 分块机制学习与工程总结

> ⚠️ **2026-09-21 状态说明**：本文记录的 LlamaIndex 分块机制（Node Parser 原理、
> SentenceWindowNodeParser、metadata 替换等）作为**框架知识仍然有效**。
> 但文中提到的**本项目实现文件已全部删除** —— `llamaindex_fixed_semantic_splitter.py`、
> `llamaindex_fixed_semantic_demo.py`、`rebuild_llamaindex_corpus.py` 属于已废弃的
> "固定切片 + 语义切片"链路（见 `CHANGELOG.md` 的 Removed 段）。
> 当前生产切块器是 `pipeline/ingest/chunker.py`（固定大小 + 结构边界，无 overlap）。

> 本文档记录 LlamaIndex 特有的分块(Node Parser)机制的学习与工程实践。
> 版本: llama-index-core (本地已安装), 文档来源: developers.llamaindex.ai
>
> **与框架无关的原理**(分块要解决什么问题、五步流程、断点判定、窗口的两种角色、
> 工程结论、策略全景) 在 `rag_chunking_concepts.md`。
> LangChain 的对应实现见 `semantic_chunking_notes.md`。

---

## 一、为什么单独研究 LlamaIndex 的分块

我们前面用 LangChain 的 `SemanticChunker` 做了语义分块。LlamaIndex 解决的是同一类问题, 但抽象层次不同:

```text
LangChain    分词器 -> 文本块(text chunks)
LlamaIndex   Document -> Node -> 更细的粒度
```

**Node 是 LlamaIndex 的核心抽象**, 它比"文本块"多带了几样东西:

| 字段 | 作用 |
|---|---|
| `text` / content | 节点正文(默认参与嵌入的就是它) |
| `metadata` | 任意键值, 可存窗口、标题、来源 |
| `relationships` | 和前后节点、父节点的关系 |
| `excluded_embed_metadata_keys` | 哪些 metadata 不参与嵌入 |
| `excluded_llm_metadata_keys` | 哪些 metadata 不送给 LLM |

后两个字段是 LlamaIndex 相对独特的设计 —— **它允许一段文本"用于匹配时"和"送给 LLM 时"呈现不同内容**, 这正是 SentenceWindow 能成立的前提。

### 1.1 metadata 到底会不会进嵌入? 实测

这个问题必须实测, 因为**它和 LangChain 的行为相反**。

LlamaIndex 的 `Node.get_content()` 支持四种模式:

```python
MetadataMode.ALL      # metadata + 正文, 全带上
MetadataMode.EMBED    # 用于嵌入: 带上除 excluded_embed_metadata_keys 之外的 metadata
MetadataMode.LLM      # 用于送 LLM: 带上除 excluded_llm_metadata_keys 之外的
MetadataMode.NONE     # 只有正文
```

实测(取 12 句语料中间的某个 node):

```text
A. 默认状态(excluded 里已有 window / original_text)
   node.get_content(MetadataMode.EMBED)
   -> '第二句是目标句子。'

B. 手动清空 node.excluded_embed_metadata_keys = []
   node.get_content(MetadataMode.EMBED)
   -> 'window: 第一句内容在这里。 第二句是目标句子。 第三句作为后文。
       original_text: 第二句是目标句子。

       第二句是目标句子。'
```

**清空排除列表之后, 窗口被完整拼进了嵌入文本。**

### 1.2 那两行 excluded 代码就是这个机制的开关

回头看 `SentenceWindowNodeParser` 里曾经看起来不起眼的代码:

```python
node.excluded_embed_metadata_keys.extend(
    [self.window_metadata_key, self.original_text_metadata_key]
)
```

**它就是"单句嵌入"和"窗口嵌入"之间的开关:**

```text
有这两行    ->  嵌入的只有单句        ->  真正的 SentenceWindow
删掉这两行  ->  窗口被拼进嵌入文本    ->  退化成窗口嵌入(即 LangChain 那种做法)
```

也就是说, **LlamaIndex 的 Node 结构本身同时支持两种模式**, 靠 `excluded_embed_metadata_keys` 切换。
而 LangChain 的 `Document` 只有"单句嵌入"这一种可能 —— 因为 metadata 压根进不了嵌入。

### 1.3 完整对照

| 能力 | LangChain `Document` | LlamaIndex `Node` |
|---|---|---|
| 正文 | `page_content` | `text` / `get_content()` |
| 元数据 | `metadata` | `metadata` |
| **元数据进嵌入** | **默认不进** | **默认进, 可用排除列表关掉** |
| 元数据进 LLM | 默认不进 | 默认进, 可用排除列表关掉 |
| 前后节点关系 | 无 | `relationships`(NEXT/PREV/PARENT/CHILD) |
| 内容可替换 | 无(直接赋值) | `set_content()` |
| 唯一 ID | 无内置 | `node_id` |
| 分块抽象 | `TextSplitter` | `NodeParser` |
| 批量切分 | `split_documents()` | `get_nodes_from_documents()` |

**为什么这个差异是决定性的:**

多出来的 `excluded_*_metadata_keys`、`relationships`、`set_content()` 这三样,
正是 SentenceWindow 能在 LlamaIndex 里原生跑通、而在 LangChain 里必须在应用层自己搭的原因。

---

## 二、Node Parser 全景

官方文档把 node parser 分成三类。

### 2.1 基于文件的解析器

按内容类型切分, 自动识别结构:

```text
SimpleFileNodeParser    自动按文件类型选用最合适的 parser
HTMLNodeParser          用 beautifulsoup 解析, 默认标签:
                        p, h1-h6, li, b, i, u, section
JSONNodeParser          解析 JSON
MarkdownNodeParser      按 Markdown 标题层级切
```

官方推荐组合:

```text
FlatFileReader + SimpleFileNodeParser
  -> 自动按类型选 parser
  -> 再串一个"基于文本的 parser"来控制实际长度
```

**"文件解析器 + 文本切分器"串联**是个值得借鉴的模式 —— 先用结构切出逻辑块, 再按长度细化。这和我们讨论过的"结构分块 -> 语义分块"是同一个思路。

### 2.2 文本切分器

```text
CodeSplitter                按语法结构切代码(chunk_lines / overlap / max_chars)
LangchainNodeParser         把任意 LangChain splitter 包装成 node parser
Chunker                     包装 chonkie 的多种切分策略
SentenceSplitter            尊重句子边界 + chunk_size + chunk_overlap  (默认推荐)
SentenceWindowNodeParser    单句成 node + 窗口存 metadata          ← 本文重点
SemanticSplitterNodeParser  语义分块(对应 LangChain 的 SemanticChunker)
TokenTextSplitter           纯按 token 数切
```

### 2.3 基于关系的解析器

```text
HierarchicalNodeParser  一次切成多个层级(chunk_sizes=[2048, 512, 128])
                        每个 node 持有父节点引用
                        配合 AutoMergingRetriever 使用
```

`AutoMergingRetriever` 的机制很有意思: **当检索命中的子节点数量超过阈值时, 自动向上替换成父节点**, 把更完整的上下文交给 LLM。

这解决的是"检索粒度"和"上下文粒度"的矛盾 —— 和我们讨论的 SentenceWindow 是同一个问题的两种解法:

```text
SentenceWindow   横向扩: 命中小句 -> 左右各取 N 句
Hierarchical     纵向扩: 命中多个小片 -> 合并成父块
```

---

## 三、SentenceWindowNodeParser 详解

### 3.1 官方定义

> The `SentenceWindowNodeParser` is similar to other node parsers, except that it splits all documents into individual sentences. The resulting nodes also contain the surrounding "window" of sentences around each node in the metadata. **Note that this metadata will not be visible to the LLM or embedding model.** This is most useful for generating embeddings that have a very specific scope. Then, combined with a `MetadataReplacementNodePostProcessor`, you can replace the sentence with its surrounding context before sending the node to the LLM.

**(官方文档, developers.llamaindex.ai)**

三句话概括了它的全部设计:

```text
1. 切成单句     每句一个 node
2. 窗口进 metadata  前后 N 句存起来, 但不参与嵌入
3. 检索后替换    用 postprocessor 把单句换成完整窗口再送 LLM
```

### 3.2 核心参数

官方示例:

```python
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)
```

参数清单:

| 参数 | 默认值(源码) | 说明 |
|---|---|---|
| `window_size` | **3** | 前后各取几句。**约束 gt=0, 不允许设 0** |
| `window_metadata_key` | `"window"` | 窗口文本存到哪个 metadata 键 |
| `original_text_metadata_key` | `"original_text"` | 原句存到哪个键 |
| `sentence_splitter` | NLTK punkt | 句子切分函数, 可替换 |
| `include_metadata` | True | 是否保留文档元数据 |
| `include_prev_next_rel` | True | 是否建立前后节点关系 |

**两个值得注意的点**:

1. **`window_size` 不接受 0。** 源码里是 `gt=0`, 设 0 会直接报校验错误。这一点和 LangChain 的 `buffer_size=0`(合法, 退化成单句)不同 —— 说明它的设计前提就是"一定要有窗口"。
2. **官方文档示例里写的是 `original_text_metadata_key="original_sentence"`, 但源码默认值是 `"original_text"`。** 两者不一致, 以源码为准; 如果你显式传了 `"original_sentence"`, 那 postprocessor 取数时也得用同一个键。

### 3.3 源码机制

窗口构造(`build_window_nodes_from_documents` 的片段):

```python
for i, node in enumerate(nodes):
    window_nodes = nodes[
        max(0, i - self.window_size) :
        min(i + self.window_size + 1, len(nodes))
    ]

    node.metadata[self.window_metadata_key] = " ".join(
        [n.text for n in window_nodes]
    )
    node.metadata[self.original_text_metadata_key] = node.text

    # exclude window metadata from embed and llm
    node.excluded_embed_metadata_keys.extend(
        [self.window_metadata_key, self.original_text_metadata_key]
    )
    node.excluded_llm_metadata_keys.extend(
        [self.window_metadata_key, self.original_text_metadata_key]
    )
```

逐点说明:

```text
max(0, i - window_size)                    左边界夹紧, 防止负数下标
min(i + window_size + 1, len(nodes))       右边界夹紧
" ".join([n.text for n in window_nodes])   窗口文本直接拼原句
```

**最关键的是最后那两行 `excluded_*_metadata_keys.extend(...)`:**

它把 `window` 和 `original_text` 两个键**同时排除出嵌入和 LLM 的可见范围**。这样做的效果是:

```text
嵌入时  ->  只用 node.text(单句), 窗口不参与  ->  匹配精确
送 LLM 时 ->  默认也不带窗口
            要靠 MetadataReplacementPostProcessor 显式替换才带上
```

换句话说, **"排除"是默认状态, "替换"是显式动作**。这个设计避免了"不小心把窗口一起嵌进去导致向量被稀释"。

### 3.4 节点结构实测

输入 6 句中文, `window_size=1`, 观察生成的 node:

```text
node[i].text              = 第 i 句原文
node[i].metadata["window"]        = 第 i-1 句 + 第 i 句 + 第 i+1 句
node[i].metadata["original_text"] = 第 i 句原文
node[i].excluded_embed_metadata_keys = ["window", "original_text"]
node[i].excluded_llm_metadata_keys   = ["window", "original_text"]
```

---

## 四、MetadataReplacementPostProcessor

### 4.1 源码

```python
class MetadataReplacementPostProcessor(BaseNodePostprocessor):
    target_metadata_key: str

    def _postprocess_nodes(self, nodes, query_bundle=None):
        for n in nodes:
            n.node.set_content(
                n.node.metadata.get(
                    self.target_metadata_key,
                    n.node.get_content(metadata_mode=MetadataMode.NONE),
                )
            )
        return nodes
```

### 4.2 三个要点

1. **它直接改 node 的内容** —— 调的是 `set_content()`, 不是临时拼接。检索结果被就地替换。
2. **取了兜底值** —— 如果 metadata 里没有目标键, 就保持原内容不变, 不会报错。
3. **执行时机在检索之后** —— 它是 postprocessor, 所以嵌入和相似度计算早就完成了, 此时替换不影响匹配结果。

### 4.3 用法

```python
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

query_engine = index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
```

注意 `target_metadata_key` 必须和建索引时的 `window_metadata_key` 一致。

---

## 五、实测: 中文场景的坑

### 5.1 默认切分器对中文完全失效

`SentenceWindowNodeParser` 的默认 `sentence_splitter` 是 NLTK 的 punkt 分词器。实测:

```text
输入:  "RAG 的第一步是从知识库检索文档。检索质量决定最终效果。
        向量是文本的语义编码。分片粒度影响检索精度。
        块太大会稀释语义。块太小会丢上下文。"
        (6 句, 69 字)

window_size=1 的默认设置
输出:  1 个 node

node[0].text = 整段文字(6 句全在一起)
```

**6 句话切出 1 个 node**, 等于窗口机制彻底失效。

这个问题和 LangChain 那边一模一样 —— 我们当时用默认正则 `(?<=[.?!])\s+` 也是整篇被当成 1 句。

### 5.2 官方自己也承认了

LlamaIndex 官方文档在 `SemanticSplitterNodeParser` 一节明确列出两条 caveat:

> - The regex primarily works for English sentences
> - You may have to tune the breakpoint percentile threshold.

**"正则主要适用于英文句子"** —— 官方对中文场景的坑是有明确提示的。

### 5.3 对策: 自定义 sentence_splitter

`SentenceWindowNodeParser` 的 `sentence_splitter` 是个可替换的 `Callable[[str], List[str]]`, 直接传中文切分函数即可:

```python
import re

def zh_splitter(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？；])\s*(?=\S)", text) if s.strip()]

parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    sentence_splitter=zh_splitter,
)
```

这也正是我们前面在 LangChain 里踩过、已经修好的那个正则。

---

## 六、和 LangChain 方案的对照

| 维度 | LangChain `SemanticChunker` | LlamaIndex `SentenceWindowNodeParser` |
|---|---|---|
| 切分依据 | 语义跳变(相邻向量距离) | 单句边界 |
| 嵌入对象 | 窗口(上下文拼进文本再嵌入) | **单句**(窗口只进 metadata) |
| 窗口用途 | 决定切点 | 检索后扩展上下文 |
| 输出粒度 | 若干句组成的块 | 一句一个 node |
| 块间重叠 | 无 | 天然"重叠"(每句都在邻居的窗口里) |
| 中文支持 | 默认正则失效, 需替换 | 默认分词器失效, 需替换 |
| 参数 | `buffer_size` 可为 0 | `window_size` 必须 > 0 |
| 是否需要 postprocessor | 不需要 | 需要 `MetadataReplacementPostProcessor` |

**核心差异一句话:**

```text
LangChain 的语义分块, 窗口是"嵌入的输入"
LlamaIndex 的窗口机制, 窗口是"检索的输出"
```

前者用窗口找边界, 后者用窗口补上下文。两者解决的不是同一个问题, 可以叠加使用。

---

## 七、Demo 实测

脚本: `pipeline/llamaindex_window_demo.py` (语料 12 句 / 393 字, `window_size=2`)

### 7.1 默认切分器 vs 中文切分器

```text
默认 NLTK punkt     ->  1  个 node     (窗口机制彻底失效)
中文切分器          ->  12 个 node     (每句一个)
```

### 7.2 节点结构实测

以 `node[4]` 为例:

```text
原句        : 它支持多种调度算法，比如轮询、最少连接和源地址哈希。
window_size : 2
窗口覆盖     : node[2] ~ node[6]  (共 5 句)

metadata["window"] =
    当主路由器失效时，备份路由器会在几秒内接管虚拟 IP，服务几乎不中断。
    HAProxy 则工作在应用层，为 TCP 和 HTTP 流量做负载均衡。
    它支持多种调度算法，比如轮询、最少连接和源地址哈希。
    HAProxy 的健康检查可以精确到 HTTP 状态码，因此比四层检查更可靠。
    把 Keepalived 和 HAProxy 组合使用，可以同时获得高可用和负载均衡。

excluded_embed_metadata_keys = ["window", "original_text"]
excluded_llm_metadata_keys   = ["window", "original_text"]
```

窗口宽度 = `2 × window_size + 1 = 5` 句, 首尾自动变窄。

### 7.3 检索实测(真实嵌入)

嵌入模型: 百炼 `qwen3-vl-embedding`, 1024 维。`similarity_top_k=2`。

```text
查询: Keepalived 用什么协议同步虚拟 IP
  0.7885  它通过 VRRP 协议在主动和备份路由器之间同步虚拟 IP 地址。   ← 精确命中
  0.7506  Keepalived 是运行在 LVS 路由器上的守护进程，负责健康检查与故障转移。

查询: HAProxy 支持哪些调度算法
  0.8328  它支持多种调度算法，比如轮询、最少连接和源地址哈希。       ← 精确命中
  0.6924  把 Keepalived 和 HAProxy 组合使用，可以同时获得高可用和负载均衡。

查询: 怎么避免虚拟 IP 冲突
  0.8207  部署时要先规划好虚拟 IP，避免与现有网段冲突。             ← 精确命中
  0.7237  监控方面要重点关注虚拟 IP 的归属变化和健康检查的失败次数。
```

**三个查询全部精确命中目标句**, 而且 top-1 和第二名的分差在 0.04 ~ 0.14 之间, 区分度明显。这正是"单句嵌入"的价值 —— 向量指向非常明确。

### 7.4 MetadataReplacement 前后对比

查询 `"监控时要关注哪些指标"`:

```text
--- 替换前(送给 LLM 的是单句) ---
  监控方面要重点关注虚拟 IP 的归属变化和健康检查的失败次数。
  日志里如果频繁出现状态切换，通常说明健康检查阈值设置得太敏感。

--- 替换后(送给 LLM 的是完整窗口) ---
  配置文件修改后必须重启服务，新的路由规则才会生效。 生产环境建议保留至少
  一台备份节点，并定期演练切换流程。 监控方面要重点关注虚拟 IP 的归属变化和
  健康检查的失败次数。 日志里如果频繁出现状态切换，通常说明健康检查阈值设置得
  太敏感。

  生产环境建议保留至少一台备份节点，并定期演练切换流程。 监控方面要重点关注
  虚拟 IP 的归属变化和健康检查的失败次数。 日志里如果频繁出现状态切换，通常
  说明健康检查阈值设置得太敏感。
```

**关键点**: 替换是**就地**进行的(`set_content()` 改写 node 本身), 且发生在检索之后 —— 相似度计算早已完成, 替换不影响匹配结果, 只影响最终送给 LLM 的内容。

### 7.5 一句话总结这套机制

```text
匹配时  用单句   ->  向量指向明确, 检索精确
阅读时  用窗口   ->  LLM 拿到完整上下文
中间的切换  由 MetadataReplacementPostProcessor 显式完成
```

---

## 八、LlamaIndex 固定长度 + overlap + 语义微调

除了 `SentenceWindowNodeParser`, 我们还实现了一个自定义 LlamaIndex
`NodeParser`, 用来复现当前项目的 chunk 策略:

```text
chunk_size      = 800 token
chunk_overlap   = 400 token
window_size     = 400 token
```

切分流程:

```text
Markdown
  -> 按顶层章节聚合 Document
  -> 每句计算 embedding
  -> 在 section 内按 token 累积到接近 800
  -> 在末尾 400 token 窗口内找语义距离最大的断点
  -> 生成 TextNode
  -> 下一块回退约 400 token 作为 overlap
```

实现:

```text
pipeline/llamaindex_fixed_semantic_splitter.py
pipeline/llamaindex_fixed_semantic_demo.py
```

Red Hat 预览:

```text
documents       2
nodes           42
tokens min/avg/max  378 / 590.0 / 763
chars min/avg/max   431 / 756.1 / 1162
overlap 示例        378 ~ 398 tokens
```

全量重建:

```text
Red Hat chunks   42
NGINX chunks     312
总 chunks        354
dense points     354
sparse points    354
```

它和 `SentenceWindowNodeParser` 的区别:

```text
SentenceWindowNodeParser
    每个句子一个 node
    窗口只放 metadata
    适合精确匹配后再补上下文

FixedSemanticNodeParser
    每块约 800 token
    每块带约 400 token overlap
    语义距离只用来移动块边界
    更适合已有固定长度 + overlap 的 RAG 基线
```

这是**LlamaIndex 的 NodeParser 抽象**, 不是 LangChain `TextSplitter`。
输出是标准 `TextNode`, 后续可以继续使用 LlamaIndex 的关系、metadata 和 postprocessor。

---

## 九、待办

- [x] 用真实语料跑 `SentenceWindowNodeParser`, 输出 node 结构
- [x] 对比自定义中文切分器 vs 默认切分器的结果差异
- [x] 搭最小检索闭环: 建索引 -> 检索 -> MetadataReplacement 前后对比
- [x] 实现固定长度 + overlap + 边界语义微调的 LlamaIndex NodeParser
- [ ] 在全量 Red Hat + NGINX 语料上重建索引并评估该切分器
- [ ] 探索 `HierarchicalNodeParser` + `AutoMergingRetriever`
- [ ] 探索 `SemanticSplitterNodeParser`, 和 LangChain 版做同语料对照
- [ ] 测试 `window_size` 取 1 / 3 / 5 对检索结果的影响



