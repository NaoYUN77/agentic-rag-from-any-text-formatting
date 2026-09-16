# 问题 05: metadata 质量差, 过滤和引用都不可靠

**优先级: P1**　**状态: 待解决**

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

