# 问题 07: PDF 代码块被误判为标题, section 噪声严重

**优先级: P1**　**状态: 已解决** (2026-09-19)

## 解决记录 (2026-09-19)

在修 `min(idx+1, 6)` 标题压平 bug 时, 一并验证了本问题的表现。

**修复内容** (`pipeline/ingest/parsers/pdf_common.py`):

1. `_heading_size_levels` 去掉 `min(idx + 1, 6)` 硬上限, 改为单调递增
   (上限提到 `max_level=10`)。旧实现会把第 7 档及以后的字号压进同一级。
2. `heading_stack` 从 `List[str]` 改为 `List[Tuple[int, str]]`,
   弹栈条件从 `while len(stack) >= level` 改为
   `while stack and stack[-1][0] >= level`。
   旧逻辑把"栈深度"与"字号排名"两种量纲相比, 导致同 rank 标题层层嵌套。

**验证结果** (Red Hat 负载平衡器手册, pdfplumber 路径):

```text
修复前 (mineru 产物, 旧逻辑):
  顶层 scope 里出现代码行:
    firewall-cmd --reload                     12
    ip addr add 192.168.76.24 dev eth0         9
    systemctl start firewalld                  2

修复后 (pdfplumber 路径, 新逻辑):
  无 section_path 的非标题块 = 0
  顶层 scope 全部是真实章节:
    Red Hat Enterprise Linux 7 / 目录 / 第 1~5 章 / 附录 A / 附录 B / 索引
```

**关键断言**: 摘要正文 (b0010) 的 section_path
从 `[..., '法律通告', '摘要']` (错) 恢复为 `[..., '摘要']` (对)。

**回归测试**: `pipeline/tests/test_pdf_section_path.py` (7 例)。
已用"注入旧 bug"方式验证测试确实能捕获回归 —— 注入后 4 例失败,
其中包含对真实 PDF 的断言。

**遗留**: 本问题原始描述针对的是 MinerU 输出的 Markdown 伪标题
(`preprocess.is_heading()`), 与 `pdf_common` 的字号/outline 路径
是**两条独立链路**。本次修的是后者。前者是否仍需强化
标题合法性规则 (长度/符号/配置语法), 尚未定论。

## 现象

NGINX PDF 增量入库后, 部分 payload.section 变成:

```text
comment
name1:password1
name2:password2:comment
name3:password3 > 解决方案
```

原本这些内容属于代码块或配置示例, 不是章节标题。

## 根因

MinerU 输出 Markdown 时会对部分代码行加 `#` 或形成伪标题结构。
当前 `preprocess.is_heading()` 只能排除少量 shell 提示符,
无法覆盖 NGINX 配置、SAML 配置、变量示例等代码块。

## 影响

```text
section 路径不可信
按 section 过滤失效
引用展示出现配置代码
metadata 进入 index_text 时会污染 BM25 词项
```

## 解决方向

```text
1. 解析阶段识别代码围栏和配置块
2. 标题合法性增加长度、符号、配置语法规则
3. 代码块内容不进入 section 栈
4. 对 PDF 解析结果做独立结构 QA
5. 为不同 PDF 编写 parser-specific 清理测试
```

## 待验证

```text
[ ] 修复后 Nginx chunk 的 section 是否恢复为真实章节
[ ] section 过滤是否比当前更稳定
[ ] 修复 metadata 后 BM25 排序是否改善
[ ] 引用展示是否不再出现配置代码
```
