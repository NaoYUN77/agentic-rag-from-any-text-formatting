# 问题 11: heading 层级抽取不准, section_path 前缀被污染

**优先级: P1**　**状态: PDF 与 URL/HTML 路径均已修复 (2026-09-20)**

> 分层结论:
> - **PDF 路径**已用 outline 实测标定修好（见下节）
> - **URL/HTML 路径**的根因**不是**"readability 剥掉 h1"这么单一 ——
>   还有一个**独立的栈比较 bug**（见「第三个成因」），已一并修好。
>   readability 缺根的问题仍在，但它不是 section_path 失真的主因。

## 第三个成因（2026-09-20 发现）: heading_stack 拿"栈深度"比"层级"

**这是本 issue 之前完全没识别出的成因**，也是造成 `section_path` 大面积
失真的直接原因。

### bug 形态

`html.py` 与 `markdown.py` 的弹栈条件写的是：

```python
while len(heading_stack) >= level:    # ← 拿"栈深度"和"heading 层级"相比
    heading_stack.pop()
heading_stack.append(title)
```

**两种量纲相比**。栈长 1、level 2 时 `1 >= 2` 为假 → 不弹栈
→ **新的同级标题被 append 成前一个同级标题的子节点**。

### 实测症状（`anthropic_agent_skills`）

```text
b0006 lv2 The anatomy of a skill         section_path=[]
b0016 lv3 Skills and the context window  section_path=['The anatomy of a skill']
b0021 lv3 Skills and code execution      section_path=[..., 'Skills and the context window']  ← 同级却嵌套
b0026 lv2 Developing and evaluating...   section_path=['The anatomy of a skill']               ← 同级却嵌套
b0038 lv2 Acknowledgements               section_path=['The anatomy of a skill']               ← 同级却嵌套
```

这**正是本 issue 开头描述的"A note on simply using a longer prompt 成了整篇
文档的章节根"** 的机制 —— 不是 readability 缺根，而是同级标题层层嵌套，
导致第一个出现的标题变成所有后续同级标题的祖先。

### 修法（与 pdf_common 同一套）

栈改存 `(level, title)` 二元组，弹栈条件改为：

```python
while heading_stack and heading_stack[-1][0] >= level:
    heading_stack.pop()
heading_stack.append((level, title))
```

### 实测效果

```text
section_path 在 80/117 chunks (68%) 上变了
parents 16 -> 54 ;  不同顶层 scope 9 -> 41
```

**直接修掉本 issue 的症状**：

```text
buggy: ['A note on simply using a longer prompt', 'Introducing Contextual Retrieval', ...]
fixed: ['Introducing Contextual Retrieval', ...]
```

回归测试：`pipeline/tests/test_heading_stack.py`（5 例，覆盖 html / markdown /
两者一致性；注入旧 bug 后 3 例失败）。

### ★ 但检索指标**完全不变**（必须记住）

800 + buggy parser vs 800 + fixed parser：5 个阶段 × 4 个指标**逐位相同**。

验证：两版的 **chunk 全文集合完全相同**（117 个，SHA256 逐一相同）。
原因：**chunk 边界只取决于"是不是 heading"，不取决于 `section_path` 的值**
→ 同样 117 个 chunk、同样 `dense_text` → 同样向量 → 同样召回。

> **所以本修复的价值在 metadata 质量 / 引用展示 / Parent expansion，
> 不在这套检索指标上。** 不要拿检索指标去论证它值不值得修，
> 也不要以为"section_path 修好了检索就该变好"。

## PDF 路径已修复: 字号层级用 outline 实测标定（2026-09-20）

上面「方案一归一化 level」被证伪是因为它只做**平移**，保序所以不改变结果。
但本文档 PDF 路径的问题**不是平移能解决的** —— 是**字号排名 ≠ 结构深度**。
修复思路来自 RAGFlow 的 `title_frequency` / `most_level`（用文档自身观测到的
层级分布来定层级），落地为 **outline 实测标定**。

### 根因（比"缺根"更具体）

`_heading_size_levels` 返回的是**字号排名**（"在本文档所有标题字号里排第几"），
不是结构深度。字号档数 > 真实层级数时必然错位：

```text
字号排名阶梯 (7 档):  26.9->lv1 23.0->lv2 16.3->lv3 15.4->lv4
                      14.4->lv5 13.4->lv6 12.5->lv7
outline 实测标定 (4 级): 16.3->lv1  14.4->lv2  12.5->lv3  11.5->lv4
```

于是前置页的 13.4pt 标题（配置 Keepalived / 法律通告 / 摘要）被判成 **lv6**，
比所有正文章节（最深 lv4）都深。后果不只是路径语义错，还包括
`_render_block()` 会输出 `###### 法律通告` 而不是 `# 法律通告`。

### 修法

新增 `_calibrate_font_levels(size_levels, observations)`：

1. 对每个字号，取它在 outline 中**实测出现过的层级**（多数票）作为标定值；
2. 没有观测的字号，按字号大小找最近的已标定字号插值；
3. 完全没有 outline 时返回原排名阶梯 —— **退化为旧行为，不会更差**。

标定结果（真实文档）：

```text
排名阶梯 : 26.9->1 23.0->2 16.3->3 15.4->4 14.4->5 13.4->6 12.5->7
标定后   : 26.9->1 23.0->1 16.3->1 15.4->1 14.4->2 13.4->1 12.5->3
观测样本 : 16.3->{1:4}  14.4->{2:17}  13.4->{1:1}  12.5->{3:21}
```

### 效果（真实文档，chunk_tokens=800/400）

| 指标 | 标定前 | 标定后 |
|---|---:|---:|
| heading level 分布 | `{1:11, 2:32, 3:21, 4:6, **6:3**}` | `{1:15, 2:31, 3:21, 4:6}` |
| 顶层 scope 数 | 11 | **13** |
| parents / chunks | 10 / 88 | **14 / 90** |
| 摘要正文 section_path | `[Red Hat..., 负载平衡器管理, 摘要]` | `[摘要]` |

**副作用需要说明**：前置页标题从"嵌套在书名下"变为"顶层独立"。
判断依据是 **outline 自身把前置页（目录）放在 lv1**，所以标定是忠于
文档约定的；且 parents 从 10 增到 14，对问题 10 的分组退化是正向的。

### 一个重要的反例: RAGFlow 的 `lvl <= most_level` 截断不能照搬

实测本文档 `most_level` = **6**（13.4pt 前置页命中最多，50 行），
若照搬 RAGFlow 的 `if lvl <= most_level` 截断，会把 **lv7 的 21 个真实
x.y.z 章节全部截掉**。原因是该文档的字号阶梯与结构层级**非单调**：

```text
16.3->lv1  14.4->lv2  13.4->lv1(目录)  12.5->lv3
                        ^^^ 非单调: 前置页用大字号但层级浅
```

所以本项目只借「用观测分布定层级」的**思想**，不照搬截断规则。

### 补充: 页码曾被误当标题（已由 _is_header_footer 挡住）

排查中发现 13.4pt 有 50 行命中，其中 46 行是**页码**（`1`..`41`）。
但它们在 `_is_header_footer` 阶段已被过滤，最终 0 个页码进入 blocks，
也没有污染 section_path（实测含纯数字层的 block = 0/463）。
这条记录保留在此，因为它是"字号启发式很脆弱"的直接证据。

### 补充: outline 匹配加了一层二元组 Jaccard 兜底

原实现是 `(_normalize(text), page)` **精确键匹配**，OCR/换行把标题改写一点
就静默漏掉，那个标题只能退回字号启发式（而字号启发式正是最容易错的部分）。

新增 `_OutlineMatcher`：**精确优先，失败才走字符二元组 Jaccard 模糊匹配**
（同页 ±1 范围）。阈值取 **0.85** 而非 RAGFlow 的 0.8，实测依据：

```text
长标题单字符 OCR 错 (keepalived -> keepalive)    0.941  ← 应匹配
短标题单字符 OCR 错 (Keepalived -> Keepalive)    0.850  ← 应匹配
纯空白差异 (归一化后)                             1.000  ← 精确层已覆盖
标题追加后缀 (...调度算法 vs ...调度算法概述)     0.833  ← 应拒绝!
语义无关 (法律通告 vs 摘要)                       0.000
```

0.8 会把「追加后缀」这种**不同章节**误判为同一标题；0.85 恰好挡住 0.833，
同时仍接受单字符 OCR 错。宁可漏匹配（退回字号启发式）也不要错匹配。

**诚实说明**: 实测本文档 67/67 条 outline **全部精确命中，模糊层零触发**。
所以这一层是**鲁棒性保险，不是当前收益来源** —— 它不会让本文档的指标变好，
只是让别的（OCR 噪声更大的）PDF 不至于因为一个字符之差丢掉整条 outline。

## 现象

**每篇文档的第一个标题几乎都不是 lv1**,导致 `section_path` 的第一层前缀错误。

实测 10 篇语料中 8 篇含标题的文档:

```text
doc_1d0b6b72bdf2  lv2  'What are agents?'                        ← 应为 lv1
doc_1d3b6982165d  lv1  'Rapidly scaling online storage...'       ← 唯一正常的
doc_38a690a6ba57  lv3  'Benefits of a multi-agent system'        ← 应为 lv1
doc_5c3e74c436ef  lv3  'A note on simply using a longer prompt'  ← 应为 lv1
doc_723b16a48c5d  lv2  'Context engineering vs. prompt...'       ← 应为 lv1
doc_978f022cd39f  lv2  'Introduction'                            ← 应为 lv1
doc_af05975673c5  lv2  'The anatomy of a skill'                  ← 应为 lv1
doc_da52be595f3e  lv2  '**Excessive token consumption...'        ← 应为 lv1
```

heading level 总体分布:

```text
lv1 :  1 个    ← 只有 1 个 lv1 标题!
lv2 : 39 个
lv3 : 48 个
lv4 :  7 个
```

**一篇文档的正文标题里有 1 个 lv1、39 个 lv2、48 个 lv3** —— 这不符合正常文档的层级分布。

## 根因

HTML 解析走的是 `readability_structured`(10 篇中 7 篇),而 readability 的工作方式是
**从 DOM 里评分抽取"正文节点"**,会把网站的导航栏 / 页头 / 面包屑剥掉。

这些导航结构里通常包含页面的 `h1`(文章主标题),剥离后:

```text
原始 HTML:
  <h1>Contextual Retrieval in AI Systems</h1>     ← 被 readability 当导航剥掉
  <h2>...</h2>
  <h3>A note on simply using a longer prompt</h3>  ← 正文里第一个保留的标题

markdown.py 解析时:
  heading_stack = []
  # 第一个遇到的 heading 是 lv3 → 直接 append 到空栈
  heading_stack = ['A note on simply using a longer prompt']
  section_path = heading_stack[:-1] = []           ← 标题自身不含自己
  # 后续所有块: section_path = ['A note on simply using a longer prompt', ...]
```

于是**这个 lv3 标题成了整篇文档所有块的 section_path 第一层前缀**。

实测影响范围(`doc_5c3e74c436ef`,78 blocks):

```text
71 个 block 挂在 'A note on simply using a longer prompt' 之下
```

## 影响

```text
① section_path 语义错误
   一个"补充说明"性质的 lv3 小节, 变成了整篇文档的章节根
   → 按 section 过滤 / 展示时完全误导

② 直接导致问题 10(parent 分组退化)
   _scope() 取 section_path[0] → 所有块的第一层都是同一个错误前缀
   → 10 篇文档 16 个 parent, p0002 变成"整篇文档"的兜底组

③ 引用展示错误
   答案引用里若展示章节路径, 会显示成 "A note on ... > Introducing Contextual Retrieval"
   读者会以为 "Introducing Contextual Retrieval" 是某段补充说明的子节

④ 阻断 Contextual Retrieval
   章节上下文注入的前提是章节归属可信, 现在前缀本身就是错的
```

**这是 09/10/11 三个问题中唯一需要优先修的根因。** 10 是它的下游表现。

## 解决方向

```text
方案一(推荐): 归一化 heading level
   - 解析完整篇文档后, 找出【最小的 level】, 把所有 heading 的 level 平移到 lv1 起
   - 例: 文档里只出现 lv2/lv3/lv4 → lv2 视为 lv1, lv3 视为 lv2, 依次类推
   - 优点: 不改 parser, 在 markdown.py 收尾时做一次归一化; 恢复真实相对层级
   - 难度: 低; 风险: 如果文档确实从 lv2 开始(少见), 会误判

方案二: 从原始 HTML 的 <title> / <h1> 补回根标题
   - html.py 在 parse 后, 从 HTML 里取 <title> 或唯一 h1 作为文档根
   - 把它作为 section_path 的第 0 层前缀
   - 优点: 语义最正确; 缺点: readability 剥离后不一定还拿得到

方案三: 用 section_path 的【相对深度】而非绝对 level
   - 下游不关心 level 数字, 只关心"谁包含谁"
   - 在 heading_stack 里按【出现顺序 + 栈深度】推导包含关系, 忽略 level 数字
   - 难度: 中; 但更鲁棒(不依赖 level 数字是否可信)

方案一 + 方案三 结合最稳: 先归一化 level, 再用栈深度兜底。
```

## 追加实测: 方案一与"用文本编号"都被证伪（2026-09-18）

对两类语料做了跨格式验证，**推翻了上面两个方案的前提**。

### 验证一: 文本编号能否作为层级来源? —— 不行

原本设想「heading level 不可信, 那就用正文里的编号（`2.3.1.` → 深度 3）」。
实测：

```text
PDF (Red Hat 手册, 60 个 heading)
  带数字编号      22 个 (37%)
  "第 N 章" 形式   3 个
  无任何编号      35 个 (58%)
  编号深度分布: 深度1=1 / 深度2=10 / 深度3=10 / 深度4=1

URL 语料 (10 篇博客, 95 个 heading)
  带数字编号       0 个 (0%)
```

**URL 语料完全没有编号，PDF 也只有 37%。** 该方案只在 PDF 的部分标题上有效，
不能作为通用方案。**已放弃。**

### 验证二: 归一化 level 也解决不了核心问题

对 `doc_5c3e74c436ef` 做手工推演（min level = 2, 平移后 lv2→lv1, lv3→lv2）：

```text
平移前: b0005 'A note...' lv3, b0009 'A primer on RAG' lv2
平移后: b0005 'A note...' lv2, b0009 'A primer on RAG' lv1

heading_stack 推演结果（平移前后完全一致）:
  b0006+ 的 section_path = ['A note on simply using a longer prompt']
  b0010+ 的 section_path = ['A primer on RAG: scaling to larger knowledge bases']
```

**归一化不改变任何结果** —— 因为 heading_stack 只依赖 level 的**相对大小**，
而平移是保序的。问题不在绝对数值，在于**真实的根标题（h1）被 readability 剥掉了**，
导致第一个出现的 heading 是中途层级的。

### 真正的根因（修正后的判断）

```text
URL:  readability 剥掉网站导航的 h1/h2
      → 正文第一个 heading 是 lv2/lv3
      → 层级本身是"对的"，只是缺了根

PDF:  MinerU 把层级压平（lv3/lv4 全部输出成 lv2）
      → 层级信息在抽取阶段就丢了，无法恢复
```

**两种输入的失效机理不同，且都无法靠后处理修好。**

### 结论: section_path 应降级为"尽力而为的提示"

```text
① 结构层级在两类输入上都不完全可信
   URL 缺根 / PDF 压平 → 没有通用的修复方案

② 因此 section_path 不能当作可靠的结构事实
   只能当提示用

③ 任何【依赖 section_path 的功能】都要重新评估其基础
   - Parent expansion（按 section 聚合上下文）
   - Contextual Retrieval（注入章节上下文）
   - 按 section 过滤
   这些功能的前提是"章节归属可信", 而这个前提不成立

④ 补充证据: openai_agents_api 与 openai_model_misalignment
   这两篇【一个 heading 都没有】(0 个)
   → 它们的全部 chunk 的 section_path 都是空的
   → 这解释了为什么 123 个 chunk 里有 17 个 section_path 为空
```

**建议把 issue 10（parent 分组）的优先级下调**：
在 section_path 本身不可信的前提下，优化它的分组粒度收益有限。
先解决"结构从哪来"的问题，再谈分组。

## 关联

- `issues/10_parent_grouping_degraded.md` —— **直接下游, 修好 11 才能修 10**
- `issues/07_pdf_section_noise.md` —— 同属"section 不可信"问题族
  (07 是 PDF 伪标题, 08 是位置错乱, 11 是层级错位)
- `docs/block_structure_and_blockification.md` 第 3.3 节、第 8 节结论 1

## 待验证

```text
[ ] 归一化 level 后, section_path 分布是否变合理?
    (期望: 每篇文档出现多个不同的第一层前缀)
[ ] parent 数会从 16 变成多少? 是否接近"每篇 4~8 个"
[ ] readability 剥离的 h1, 能否从原始 HTML 的 <title> 可靠还原?
    (抽查 10 篇, 对比 <title> 与正文首个 heading 的语义关系)
[ ] trafilatura 那 3 篇(OpenAI)是否有同样问题?
    (实测它们首个 heading 是 lv2, 确认是否同一根因)
```
