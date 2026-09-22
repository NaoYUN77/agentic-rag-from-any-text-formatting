# 问题 12: PDF 链路内容有损, 且质量门完全没察觉

**优先级: P1**　**状态: 已全部处理 (2026-09-22)** —— 6 处损失 5 处修复（1 处未复现），
外加质量门补上「结构保真度」维度

> ⚠️ **阅读须知**: 本文「现象」及「六处损失」写于 **2026-09-18**，当时走的是
> `pdf_mineru`（PDF → Markdown → Block）。**该解析器已于 2026-09-21 删除**，
> 换成本地 `pdf_pdfplumber`。所以下面那些数据反映的是**已被移除的链路**，
> 保留作为问题发现的历史记录。当前状态见下一节。

## 2026-09-22: 用当前管线重新实测 —— 5/6 已修复

对同一份 PDF（Red Hat 负载均衡管理指南，前 20 页）用**当前的 `pdf_pdfplumber`
路径**重新实测，逐条核对原文的六处损失：

| # | 损失 | 现状 | 证据 |
|---|---|---|---|
| ① | 代码/配置块丢失 | **未复现** | `keepalived.conf` 5→5、`systemctl` 5→5、`virtual_ipaddress` 1→1，逐字保留。原文里 `virtual_server`/`real_server`/`TCP_CHECK` 在 raw 里就是 **0 次** —— 前 20 页本就没有这些内容，不是被丢掉 |
| ② | 页码丢失 | ✅ **已修** | 126/126 block 带 `page`，范围 1~20 |
| ③ | 目录降级成假正文 | ✅ **已修（本次）** | 见下 |
| ④ | heading 层级错乱 | ✅ **已修** | `{1:9, 2:10, 3:12, 4:1}` —— 4 个层级；metadata 带 `from_outline`，走的是 PDF 原生大纲 |
| ⑤ | shell 命令被当 heading | ✅ **已修** | **0 个**（原文：8 个 parent 里 5 个 scope 是 shell 命令） |
| ⑥ | title 取错 | ✅ **已修** | `Red Hat Enterprise Linux 7 负载平衡器管理`（原文：`第 1 章 LOAD BALANCER 概述`） |
| ⑦ | 非标题块 section_path 为空 | ✅ **已修** | 0/94（0.0%）；原文为 17/123 chunks (13.8%) |

②④⑤⑥⑦ 是 2026-09-19～21 那批修复（outline 实测标定 + `heading_stack` 栈比较
bug + 页码透传）顺带解决的，**不是本次改动**。

### ③ 目录的修复（本次）

**根因**：`ingest/cleaner.py` **完全没有目录规则**。原文提到的
`_TOC_LEADER` / `_TOC_ENTRY` 属于**已被删除的旧 `pdf_loader.py`**，
所以目录块一路进了索引。

实测该目录块：

```text
b0012  type=text  page=5  section_path=['目录']
       5300 字符 / 39 行
       其中 31 行是标准 TOC 条目（1.1. KEEPALIVED 3 / A.1. 先决条件 38 …）
```

**修法**（原文「方案四」）：在 `cleaner.py` 里加目录识别并丢弃。

```python
_TOC_ENTRY = re.compile(r"^\s*[0-9A-Z]+(?:\.[0-9]+)+\.?\s+\S.*?\s+\d+\s*$")
_TOC_MIN_LINES = 5
_TOC_RATIO = 0.5
```

三个设计要点：

1. **要求 `(?:\.[0-9]+)+` 至少一层** —— 避免把有序列表（`1. 第一步`）误判成目录。
   这是最容易踩的坑：单级编号 + 结尾数字的有序列表长得和目录很像。
2. **要求结尾页码** —— 进一步收紧，正文里的编号行几乎不会带尾随页码。
3. **丢内容必须留痕** —— `clean_artifact()` 会往 `artifact.warnings` 写
   `dropped_toc_blocks: b0012`，否则以后没人知道少了什么、为什么少。

**实测效果**：

```text
blocks       126 -> 123（其余 2 个是既有的去重/空块规则）
warnings     ['dropped_toc_blocks: b0012']
目录条目残留  0（"1.1. KEEPALIVED 3" / "A.1. 先决条件 38" 均不再出现）
正文完好      keepalived.conf 5 次、systemctl 5 次、virtual_ipaddress 1 次（不变）
chunks       32 个，无一含目录条目
```

**误报验证**：对全部 126 个 block 跑检测器 —— 只有 b0012 命中（ratio 0.79），
**其余 125 个全部 < 0.2**。阈值 0.5 有 2.5 倍余量，零误报。

**URL 语料不受影响**：10 篇 HTML 语料的 chunk_id 序列**逐位一致**
（113 chunks / 105 parents / 34,662 token 不变）—— 目录修复只作用于 PDF。

回归测试：`tests/test_cleaner.py`（7 例，含「有序列表不得误判」的护栏）。

### 质量门的内容保真度检查（2026-09-22 已实施）

本文开头指出的**独立缺陷**已修复：

> 质量门只看 token 数与长度分布，不看内容保真度。一本配置手册丢了全部
> 配置示例，质量分仍是 0.99。

`ingest/quality.py` 新增 `_structural_fidelity()`，补上了此前**完全没有**的
「结构保真度」维度。issues 11/12 的三次回归当时**一个都没被发现**：

| 回归 | 新增判据 | flag |
|---|---|---|
| PDF 页码完全丢失 | `page_coverage == 0`（仅 PDF） | `pdf_without_page_numbers` |
| heading 层级被压平 | 标题 ≥5 且层级数 ≤1 | `flat_heading_levels` |
| `section_path` 大面积为空 | 标题 ≥3 且正文块覆盖率 = 0 | `section_path_all_empty` |

三个设计要点：

1. **判据都取「有前提才检查」** —— 没 heading 的文档不该因为「层级单一」被扣分，
   非 PDF 不该因为「没有页码」被扣分。否则会制造一堆假警报。
2. **只做小扣分**（每个 flag −0.06）—— 目的是**可见** + 把状态压到 medium
   提示人工看一眼，不是把文档一棒打死。结构坏了内容往往还在，仍值得进索引。
3. **必须写 `reasons`** —— 光扣分不解释等于又制造一个黑盒。

**实测零误报**：

```text
真实 PDF（前 20 页）  score=0.9902  flags=[]  page_coverage=1.0
                      distinct_heading_levels=4  section_path_coverage=1.0
7 篇 URL 语料         全部 ok，无结构 flag（score 0.996~1.000）
```

回归测试 `tests/test_quality.py`（8 例）：既钉「该报的要报」，
也钉「不该报的别报」（无 heading 文档、非 PDF、标题数不足都不触发）。

> 说明：原文「方案一」提的是「检查代码围栏/表格占位符比例」。
> 实测在当前的 `pdf_pdfplumber` 路径上 ① 已不复现（配置示例逐字保留），
> 所以改为检查**真正发生过的那三类结构回归**，而不是一个当前不会触发的判据。

---

## 现象（2026-09-18，`pdf_mineru` 链路，已废弃）

首次对 PDF 实测（Red Hat 负载均衡管理指南, 前 20 页）：

```text
命令  format_router_demo.py --file <pdf> --pdf-pages 1-20
route pdf_mineru (magic bytes 正确识别)
blocks 226 (heading 60 / text 158 / list 8)
parents 8
chunks 61
quality high 0.9894      ← 打了高分
```

**链路跑得通，路由正确，但内容有 6 处严重损失，质量门一个都没报。**

这暴露了一个独立缺陷：**质量门只看 token 数与长度分布，不看内容保真度**。
一本配置手册丢了全部配置示例，质量分仍是 0.99。

## 六处损失（全部实测）

### ① 代码/配置块几乎全部丢失 —— 最严重

```text
virtual_server  0 次
real_server     0 次
delay_loop      0 次
TCP_CHECK       0 次
raw md 中的代码围栏 ```  0 个
```

这是一本讲配置的书（`keepalived.conf` / `haproxy.cfg`），**配置示例全没了**。

**根因在 MinerU，不在本项目管线。** `flash-extract` 是免鉴权的快速模式，
官方文档明确写「Images, Tables, and Formulas are replaced with placeholders」。
要保真必须用 `extract`（需鉴权）或换 Docling。

### ② 页码完全丢失

Markdown 输出中**没有任何页边界标记**，无法回答"这段内容在第几页"。

页码只以残渣形式存活：在目录块的文本里（`1.1. KEEPALIVED 3`）。

**这是结构性缺陷**：`PDF → Markdown → Block` 的中转方式**在原理上就保不住页码**，
因为 Markdown 没有页的概念。要页码必须在抽取阶段注入页标记。

### ③ 目录降级成一段"假正文"

```text
b0028  type=text  len=656
内容   1.1. KEEPALIVED 3
       1.2. HAPOXY 3
       2.1. 基本 KEEPALIVED 负载平衡器配置 4
       ...

落在 chunk doc_3f1f8371240f_c0009 (486 token)
section_path = ['Red Hat Enterprise Linux 7', '目录']
```

**它进了索引。** 检索时完全可能匹配到目录而不是正文。

### ④ 标题层级错乱

```text
level 分布: lv1=7 / lv2=53 / lv3=0 / lv4=0
14 个 heading 的编号深度与 level 不一致

  "2.3.1."   真实深度 3  但 level=2
  "2.4.2.1." 真实深度 4  但 level=2
  "## 第3章为KEEPALIVED设置负载均衡器先决条件"  level=2
     → 变成了「第 2 章」的子节点（严重错乱）
```

后果：`section_path` 丢中间层。`2.3.1.` 的 section_path 只有
`['第 2 章 KEEPALIVED 概述']`，**`2.3.` 那一层被吃掉了**。

### ⑤ shell 命令被误判为 heading（问题 07 在 PDF 上复现）

parent scope 里出现了：

```text
p0004 scope=ip addr add 192.168.76.24 dev eth0
p0005 scope=systemctl start firewalld
p0006 scope=systemctl enable firewalld
p0007 scope=firewall-cmd --permanent --direct --add-
p0008 scope=firewall-cmd --reload
```

**8 个 parent 里有 5 个的 scope 是 shell 命令**，而不是章节名。
这些命令成了 `section_path[0]`，直接污染分组。

### ⑥ title 取错

```text
title = "第 1 章 LOAD BALANCER 概述"
应为    "Red Hat Enterprise Linux 7 / 负载平衡器管理"
```

原因：`title` 取了第一个 lv1 heading，而文档真正的标题是 lv2。

## `clean()` 的实测行为（附带发现）

对 `pdf_loader.clean()` 逐行实测：

```text
✓ 删目录点线行     _TOC_LEADER  生效
✓ 删孤立页码       _PAGE_NUMBER 生效
✗ 目录条目文字未删  _TOC_ENTRY   不生效
```

`_TOC_ENTRY = re.compile(r"^.{0,80}\.{3,}\s*\d+\s*$")` 要求 **3 个连续点**，
但实际格式是 `1.1. KEEPALIVED 3`（无点线）。

**结果比"整个目录删掉"更糟**：点线删了、条目留着，
变成一段看起来像正文的噪声进入索引。

## 影响

```text
① 配置类文档的 PDF 链路基本不可用
   这本书的核心价值就是配置示例, 而示例全丢了

② 检索可能命中目录
   c0009 是纯目录内容, 486 token, 会参与召回

③ 页码无法用于引用
   引用只能给到 chunk_id, 无法给"第 N 页"
   对技术手册这类场景, 页码是刚需

④ parent 分组被 shell 命令污染
   8 个 parent 里 5 个 scope 是命令, 分组失去语义
```

## 解决方向

```text
方案一(推荐先做): 在质量门里加"内容保真度"检查
   - 检查抽取产物里是否出现代码围栏、表格占位符比例
   - 代码块占比异常低 → 标记 parser 降级
   - 难度: 低; 收益: 让问题【可见】而不是被 0.99 掩盖

方案二: 换抽取模式
   - flash-extract → extract(需鉴权), 保真度更高
   - 或接入 Docling(docling_optional.py 目前是空壳)
   - 难度: 中; 需要评估鉴权成本与效果

方案三: 抽取阶段注入页标记
   - 在 MinerU 输出里按页插 `<!-- page: N -->`
   - 需要 MinerU 支持分页输出, 或按页调用后拼接
   - 难度: 中; 这是保住页码的唯一途径

方案四: 目录识别与隔离
   - 识别"目录"标题后的连续条目块, 标为 type=toc 而非 text
   - 可选: 不进索引, 或降权
   - 难度: 低; 收益明确
```

## 追加实测: 本地 PyMuPDF 路线远优于 MinerU（2026-09-18）

对 MinerU `extract` 与本地库做了能力对比实测。

### MinerU 精确模式被鉴权卡住

```text
$ mineru-open-api extract <pdf> -f json
Error: no API token found. Run 'mineru-open-api auth' to configure your token
```

`auth` 是交互式命令，本机无 token 配置 → **该路线当前不可用**。

### 本地 PDF 库：本机原本一个都没装

```text
fitz (PyMuPDF)  未安装
pdfplumber      未安装
pypdf / PyPDF2  未安装
docling         未安装
```

在隔离环境装上 PyMuPDF 1.28.2 + pdfplumber 0.11.10 后实测。

### 关键发现一: 该 PDF 有原生大纲（67 条书签）

```text
层级  标题                                    页码
  1    目录                                     5
  1    第 1 章 LOAD BALANCER 概述                 7
  2    1.1. KEEPALIVED                          7
  2    1.2. HAPOXY                              7
  3    2.3.1. keepalived Scheduling Algorithms   11
  4    2.4.2.1. 直接路由和 ARP 限制                14
  1    第 3 章 为 KEEPALIVED 设置负载均衡器先决条件    16
```

**这是 MinerU 完全丢掉的东西** —— 层级 1/2/3/4 全对，而且**带页码**。

对比 MinerU 的输出：`lv1=7 / lv2=53 / lv3=0 / lv4=0`，且 `第3章` 被压成 lv2。
**PyMuPDF 从大纲拿到的层级是权威的，不需要推断。**

### 关键发现二: 页数从 20 变成 51

```text
flash-extract 上限 20 页 → 之前只拿到 39% 的内容
PyMuPDF 读到 51 页   → 完整
```

### 关键发现三: 元数据里有真标题

```text
title: 'Red Hat Enterprise Linux 7 负载均衡器管理'
```

解决了问题 ⑥（title 取成"第 1 章"）。

### 关键发现四: 字号可区分标题与正文

```text
正文:  9.6 / 10.6 / 11.5 pt
标题:  13.4 / 14.4 / 15.4 / 16.3 / 23.0 / 26.9 pt
```

### 关键发现五: 页码在文本层可直接取

每页页脚 `y≈818` 处有纯数字（1,2,3,4,5,6,7,8...），可直接读取。

### PoC: PDF 直接产 Block（不经 Markdown）

脚本 `experiments/pdf_test/poc_pdf_to_blocks.py`，实测字段填充率：

```text
字段          PoC 填充率     之前(经 Markdown)
page          1576 / 1576    0 / 226
bbox          1576 / 1576    0 / 226
font_size     1576 / 1576    拿不到
level         83 个 heading  0（且层级错误）
```

章节页码验证（MinerU 完全给不了）：

```text
page=7   lv1  第 1 章 LOAD BALANCER 概述
page=8   lv1  第 2 章 KEEPALIVED 概述
page=16  lv1  第 3 章 为 KEEPALIVED 设置负载均衡器先决条件
page=29  lv1  第 4 章 使用 KEEPALIVED 初始负载均衡器配置
```

### PoC 的四个已知缺陷（需在正式实现中解决）

```text
① 段落未合并: 产出 1576 个 block（MinerU 只有 226）
   原因是按【行】输出, 一个段落被拆成多行
   → 需按 bbox 的 y 间距 + 行距判断段落边界

② 层级只到 lv2: 分布 {1:16, 2:67}, 丢了大纲里的 lv3/lv4
   原因是标题精确匹配 (text, page) 失败, 落到字号兜底逻辑
   → 需改为模糊匹配（归一化空格/全半角）

③ 页眉混入: b0005 "Red Hat Enterprise Linux 7 负载均衡器管理"(p3) 被误判为 lv1
   → 需按 y 坐标区域过滤页眉页脚

④ 代码块无法识别: 该 PDF 由 wkhtmltopdf 生成, 代码块【不用等宽字体】
   → 字体特征失效, 需要别的判据（缩进/背景色/内容模式）
```

### 能力对比结论

```text
能力            MinerU flash   MinerU extract   PyMuPDF(本地)
页数上限        20             600              无限制
鉴权            免             需 token         免
离线            ✗              ✗                ✓
体积            —              —                约 20MB
页码            ✗              ?                ✓
bbox            ✗              ?                ✓
原生大纲        ✗              ?                ✓ 67 条
字号            ✗              ?                ✓
标题层级        ✗ 压平          ?                ✓ 权威
表格结构        ✗ 占位符        ✓                ✗ 弱
版面分析        ✗              ✓ ML             ✗
代码块          ✗              ✓                ✗ 本 PDF 不行
```

**结论: 结构字段上本地 PyMuPDF 完胜，且离线免费无鉴权。**
短板是表格结构与版面分析 —— 这两项才需要 MinerU extract 或 Docling。

**因此推荐路线: PyMuPDF 为主 + MinerU 可选补充（而非二选一）。**

## 关联


- `issues/07_pdf_section_noise.md` —— shell 命令误判为标题, 本次在 PDF 上复现
- `issues/11_heading_level_extraction.md` —— 层级问题, 本次证明 PDF 更严重
- `docs/eval_harness_design.md` —— 质量门只看长度不看保真度, 是评估缺口的另一面

## 附带发现: PDF 的回退链是空的

`router.py:26-34` 给 PDF 配了回退链：

```python
parser="pdf_mineru",
fallbacks=["pdf_docling", "pdf_pymupdf"],
```

但实测这两个回退**都是空壳**：

```text
pdf_docling  (docling_optional.py, 15 行)  → raise NotImplementedError
pdf_pymupdf  (pdf_pymupdf.py,      13 行)  → raise NotImplementedError
```

所以 PDF 链路**实际上没有本地回退**：MinerU 一旦失败（网络、超 20 页、超 10MB、
CLI 未安装），整个 PDF 就无路可走，只能报错。

对比 HTML：`html_trafilatura` → `html_readability` 两个名字**指向同一个函数
`parse_html`**，而真正的择优逻辑在该函数内部（按 heading 数比），
所以 HTML 的"回退"其实也不是链式的 —— 它是**一次调用内部的并行择优**。

**结论：目前只有 HTML 一条路有实质性的双路保障，PDF 的回退是名义上的。**

## 待验证

```text
[ ] extract(鉴权模式) 能否保住代码块和页码? 成本多少?
[ ] Docling 在同一 PDF 上的表现如何?
[ ] 质量门加"保真度检查"的具体判据是什么?
    (代码围栏数 / 表格占位符比例 / 目录块比例)
[ ] PDF 是否应该走独立链路而不进主索引?
```
