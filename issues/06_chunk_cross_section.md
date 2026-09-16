# 问题 06: 块跨章节, 导致主题混淆

**优先级: P0**　**状态: 已修复**

## 现象

查询关于 Keepalived, top-2 却出现了 HAProxy 的内容。

定位到具体块:

```text
idx 1  615字  section='第 1 章 LOAD BALANCER 概述 > 1.2. HAPOXY'
              HAProxy×7   Keepalived×3
              ↑ 标的是 1.2 节, 但内容混进了后面的章节
```

数据来源: `pipeline/experiments/11_pipeline_preprocessed.txt`

## 根因

**分块器只看长度和语义距离, 完全不看章节边界。**

```python
# 修复前的逻辑
while end < n and (size + len(sents[end]) <= max_chars or end == start):
    吃满到 max_chars
    ↓
在边界窗口里找语义距离最大的位置切
    ↓
但候选范围可以跨过章节边界
```

**结果**: 一个块里同时装着 `## 1.2 HAPOXY` 的结尾、`## 1.3` 的全部、`# 第 2 章` 的开头。

**为什么这是问题**:

```text
向量稀释     一个向量要同时表达两个章节的语义 -> 两边都不像
召回混淆     query 匹配到其中任意一个主题, 都会召回这个块
元数据失真   section 只标了第一个主题, 过滤时会把整块当成那个主题
```

## 修复

**以顶层章节为硬边界, 章内再按长度+语义分块。**

```python
def chunk_by_section(sents, sections, vecs, max_chars, window_chars):
    tops = [(s or "").split(" > ")[0].strip() for s in sections]
    # 按顶层章节切成连续区间
    while start < n:
        end = start + 1
        while end < n and tops[end] == tops[start]:
            end += 1
        # 每个区间内部独立分块
        sub = chunk_by_window(sents[start:end], vecs[start:end], ...)
        ...
```

**为什么用"顶层章节"而不是"每个小节"**: 这本手册有 39 个小节, 每节独立成块会产生 39+ 个碎片。用一级章节(第 N 章)作边界, 块数从 21 涨到 22, 基本不变。

## 效果

```text
                      修复前        修复后
chunk 数               24            22
字数 min / max         431 / 2138    208 / 800
块跨章                 是            否
```

**验证**: 查询"如何配置 keepalived 的故障转移", top-3 全部落在 Keepalived 相关章节, 不再出现纯 HAProxy 的块。

## 连带发现

修复过程中暴露了预处理的两个 bug:

```text
① shell root 提示符被当成标题
   "# systemctl start firewalld"  -> 被当成一级标题, 重置章节栈
   -> 已修: 加了 looks_like_shell() 判别

② 列表项被当成标题
   "## 1. 确保 firewalld 正在运行。"  -> 被当成二级标题
   -> 未修, 但影响小(只影响那个小节的命名)
```

## 待验证

```text
[ ] "顶层章节作边界"是不是最优? 换成"二级章节作边界"块数会涨多少?
[ ] 跨章但主题连续的段落(比如一章的结尾和下一章的开头在讲同一件事)被切开, 会不会损失?
[ ] 块最小 208 字, 是否需要 min_chars 下限合并?
```
