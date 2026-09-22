# 问题 05: metadata 质量差, 过滤和引用都不可靠

**优先级: P1**　**状态: 部分已修复；残留 1 个已定位的问题（3 篇文档无结构）**

> ⚠️ 本文「现象」「根因」写于旧 LlamaIndex 管线（`section` 从脏标题句里截 40 字）。
> 那个根因在当前管线里**已不存在** —— 现在用 `section_path` 数组 + `headings`。
> 当前状态见下一节。

## 2026-09-22: 用当前索引重测

### 引用所需字段的覆盖率（Qdrant payload，113 chunks）

| 字段 | 覆盖率 |
|---|---|
| `file_name` / `url` / `source_uri` / `mime_type` | **100%** |
| `title` | 94% |
| `section` / `section_path` | **88%** |
| `headings` | 84% |
| `page` | **不适用** —— 语料 10 篇全是 HTML/Markdown，本来就没有页 |

`section_path` 深度分布：深度 1 = 48 个、深度 2 = 40 个、深度 3 = 10 个、
深度 4 = 2 个，另有 **13 个深度 0**。

> ⚠️ 注意 `chunks.jsonl`（瘦身 dump）**不含** `page`/`file_name`/`url`/`headings`，
> 查这些字段要看 Qdrant payload 或 `artifact.json`，否则会误判成"全缺"（踩过）。

### 那 13 个没有 `section_path` 的块，分两类

```text
6 篇 anthropic 的首段（c0）  —— 标题之前的引言段，内容正常，只是没有所属章节
2 篇 openai 的全部块（7 个）  —— 整篇 0 标题
```

前者正常；**后者是真问题**。

### 真问题：3 篇 openai 文档丢了全部结构

| 文档 | parser | 字符数 | **标题数** |
|---|---|---:|---:|
| 7 篇 anthropic | `html_readability` | 8907~39362 | 7~19 |
| `openai_agents_api` | **`markdown`** | 7846 | **0** |
| `openai_model_misalignment` | **`markdown`** | 12804 | **0** |
| `openai_scaling_storage` | `markdown` | 19667 | 9 |

**根因**（`rebuild_phase0_corpus.py` 的注释里已写明）：3 篇 openai 文档
**抓不到源 HTML**，退而用旧产物的 `document.md` → 走 `markdown` 解析器 →
其中 2 篇的 markdown 里**一个 `#` 都没有**。

**影响**：这 2 篇（占语料 16% 的 chunk）**没有 `section_path`** ——
按章节过滤、按章节引用都无从谈起；`title` 也是 `None`。

### 质量门此前**抓不到这个**（已修）

判据 2/3 都要求「**有标题才检查**」（`len(headings) >= 5` / `>= 3`），
于是 **0 标题的文档反而静默通过**。已新增判据 4：

```text
long_document_without_headings   字符数 >= 2000 且 0 标题
```

阈值依据：全部**有**标题的文档都 ≥ 8907 字符，而**无**标题的是 7846 / 12804
—— 取 2000 留足余量，不会误伤短文（一段话的便签本来就该没标题）。

**实测零误报、精确命中**：

```text
7 篇 anthropic        score 0.9957~1.0000   ok
openai_agents_api     score 0.8900  ⚠️ long_document_without_headings
openai_model_misalig  score 0.8900  ⚠️ long_document_without_headings
openai_scaling_storage score 0.9959  ok（有 9 个标题）
```

> 说明：**没有**让结构 flag 去压低 `status`。那会改变 `decide_index` 的
> `sparse_weight`（1.0 → 0.6），进而影响检索指标，需要重跑评估才能动。
> 本轮目标是让问题**可见**（flag + score 0.95 → 0.89），这一层已经做到。

### 仍未处理

- **2 篇 openai 文档的结构无法恢复** —— 源 HTML 抓不到。
  实测（2026-09-22）：重新跑 `fetch_phase0_sources.py --only openai_agents_api`，
  openai.com 返回的是 **Cloudflare 挑战页**（`<title>请稍候…</title>`，11414 字节），
  不是正文。**换抓取方式（如带 cookie/浏览器指纹）或换掉这 2 篇语料。**

  > 对照：用 WebFetch 取同一个 URL 能拿到完整正文，且**标题结构完好**
  > （1 个 h1 + 7 个 h2 + 3 个 h3）。所以**源页面是有结构的，是我们没抓到**。

- ⚠️ **顺带修了一个抓取守卫的漏洞**（`fetch_phase0_sources.py`）：
  原先只用 `len(html) < 5000` 判断抓取是否成功，而**挑战页有 1 万多字节**，
  轻松过关 → 挑战页会被当成正文写进 `experiments/_sources/`，
  后续解析出一篇**没有标题、没有正文**的"文档"却毫无告警。
  已新增 `_looks_like_challenge()`（在开头 30KB 里找
  `just a moment` / `请稍候` / `enable javascript` / `cf-challenge` 等特征），
  写盘前拦截，并给出可操作的失败信息。回归测试 `tests/test_fetch_guard.py`（6 例）。

---

## 现象

入库时生成的 metadata:

```text
块0: {
  'file_name': 'redhat_p1-20.md',
  'page': 1,
  'section': 'Red Hat Enterprise Linux 7  ## 负载平衡器管理  ',     ← 脏
  'chunk_index': 0,
  'char_count': 2138,
  'n_sentences': 1
}
块2: {
  'section': '目录  1.1. KEEPALIVED 3 1.2. HAPOXY 3 1.3.',           ← 脏
  ...
}
```

数据来源: `pipeline/experiments/10_pipeline_full.txt`

## 根因

### 根因 A: section 从"句子的开头"取, 但标题句是脏的

```python
if t.startswith("#"):
    cur = t.lstrip("#").strip().replace("\n", " ")[:40]
```

**问题在于: 标题句本身就包含了标题 + 正文。**

```text
S4 = "## 1.1. KEEPALIVED  keepalived 守护进程在主动和被动 LVS 路由器上运行。..."
     ↑ 标题                ↑ 标题后面的正文也粘在同一"句"里
```

`[:40]` 截断后拿到的是"标题 + 正文开头"的混合体。

**这和问题 01 是同一个根因的两面** —— 标题和正文粘在一起, 既污染正文块, 又污染 metadata。

### 根因 B: page 是估算的

```python
CHARS_PER_PAGE = 900
page = 累计字符数 // CHARS_PER_PAGE + 1
```

**这是硬编码的估算**, 依据是"17123 字 / 20 页 ≈ 856 字/页"。但:

```text
每页字数不均匀(有图表的页少, 纯文字页多)
MinerU 的 Markdown 输出里没有显式的分页标记
所以只能估算, 而且必然不准
```

### 根因 C: 没有利用 MinerU 能提供的信息

MinerU 实际是**按页解析**的, 它知道每段内容来自第几页。但我们的 `pipeline/pdf_loader.py` 只保存了拼好的 Markdown, **丢掉了分页信息**。

## 影响

```text
过滤不可靠     section 里混着正文, MatchText 容易误匹配或漏匹配
引用不准确     返回给用户"来源: 第 N 页"可能是错的
排序辅助失效   本可以用 section 做加权(同章节的块优先), 现在不敢用
```

## 解决方向

### 方向 1: 标题独立提取(先解决 A)

分句**之前**, 先把标题行单独摘出来:

```text
输入    "## 1.1. KEEPALIVED\nkeepalived 守护进程在..."

处理    1. 识别行首的 "#+ "
        2. 把标题内容存为当前 section
        3. 把标题行从正文里剥掉(或拆成独立的一句)

输出    section = "1.1. KEEPALIVED"
        正文    = "keepalived 守护进程在..."
```

**这一步同时解决了问题 01 的"Markdown 标记污染"。**

### 方向 2: 保留分页信息(解决 B 和 C)

改 `pdf_loader.extract()`: 不用 MinerU 拼好的整篇 Markdown, 而是**按页解析**, 每页单独保存, 带上页码。

```text
方式 A   逐页调用 MinerU(--pages N-N)
         成本高(每页一次 API 调用)

方式 B   解析 MinerU 的输出目录结构(如果它保留了分页)
         需要先确认 MinerU 的输出格式

方式 C   用 PyMuPDF 拿到每页的文本, 和 MinerU 的输出做对齐
         复杂但不用额外 API 调用
```

**需要先调研 MinerU 的输出里有没有分页信息。**

### 方向 3: section 做成层级路径

```text
现在     "1.1. KEEPALIVED"
改成     "第 2 章 KEEPALIVED 概述 > 1.1. KEEPALIVED"
```

**这样既支持精确匹配(整串), 也支持前缀匹配(某章下的所有节)。**

**OpenAI 的 `attributes` 用的就是这个思路** —— 短、结构化、可过滤。

## 待验证

```text
[ ] MinerU 的 Markdown 输出里到底有没有分页信息?
[ ] 标题行的识别规则? (只认 "#+ " 还是也要认加粗行?)
[ ] section 层级路径怎么构建(需要维护一个标题栈)
[ ] page 如果拿不到准确的, 用"章节"代替是不是更好?
```

