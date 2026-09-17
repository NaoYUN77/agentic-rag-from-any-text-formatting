# 问题 08: HTML 补充代码块位置错乱, section 归属丢失

**优先级: P2**　**状态: 潜伏问题(当前未触发)**

## 现象

`html.py:111-123` 的"代码块补充"逻辑把 `<pre>` 里的代码追加到 `artifact.blocks` **尾部**,导致:

1. `section_path = []`(空)—— 完全没有章节归属
2. `parent_id = None` —— 挂在任何 heading 之外
3. 物理位置漂移到文档末尾,而不是它原本所在的章节

实测数据:

```text
doc_5c3e74c436ef (Contextual Retrieval, 78 blocks)
  b0076 text  sec=['A note...', 'Acknowledgements']    ← 正常块
  b0077 code  sec=[]   meta={'source':'html_code_supplement'}
  b0078 code  sec=[]   meta={'source':'html_code_supplement'}

doc_58b1b15bbd49 (Introducing the Agents API, 24 blocks)
  b0022/b0023/b0024 code sec=[] meta={'source':'html_code_supplement'}  (24 块中补了 3 块)
```

`b0077` 的实际内容是 `Contextual Retrieval` 一文里 Prompt Caching 的 `<document>` 模板代码,
本应属于 "Using Prompt Caching to reduce the costs of Contextual Retrieval" 一节,
现在却被放到了 "Acknowledgements" 之后。

## 根因

```python
# html.py:104  parse_markdown_text() 先跑完, sections 是基于 Markdown 行序推导的
artifact = parse_markdown_text(markdown, source, parser=parser)

# html.py:112  然后才从原始 HTML 里补代码块, 直接 append 到尾部
for language, code in _extract_code_blocks(html):
    if code in existing_code:
        continue
    artifact.blocks.append(DocumentBlock(
        block_id=f"b{len(artifact.blocks) + 1:04d}",
        order=len(artifact.blocks) + 1,
        # section_path / parent_id 都没设 → 默认 [] / None
    ))
```

**两个独立的缺陷**:

### 缺陷 A: 用 `len(blocks)` 而非 `order` 生成编号

`markdown.py:37` 的 `next_id()` 用 `order` 计数器;这里用 `len(artifact.blocks)`。
当两者相等时结果一致(当前 10 篇语料恰好如此),但当 cleaner 过滤掉任何块时就会脱钩。

已实测到脱钩实例(`experiments/router_phase0_url/artifact.json`,该产物经 cleaner 过滤):

```text
blocks[0] = b0003 / order=1      ← b0001、b0002 被 cleaner 吃掉
blocks[-1] = b0054 / order=52    ← 52 个块, 编号却是 b0003~b0054
```

修复方向:改为复用 `next_id()` 或统一用 `order`。

### 缺陷 B: 补进的块没有章节归属

`section_path` 为空 → `chunker._group_blocks()`(`chunker.py:201`)按 `section_path[0]` 分组时:

```python
@staticmethod
def _scope(block, fallback):
    if block.section_path:
        return block.section_path[0]
    return fallback or "文档开头"      # ← 空 section_path 落进这里
```

结果:这些代码块被归入 "文档开头" 组,造成**跨章节漂移**——
一个属于第 4 小节的代码块,和文档开头的引言段落分到了同一个 parent。

## 影响

当前影响有限(10 篇语料共补 5 块,且都是代码),但有三个实际后果:

```text
① parent 分组污染   代码块被塞进"文档开头"parent, 该 parent 的主题被稀释
② 位置检索失效      代码块在 chunk 序列里的位置与原文不符, 无法用于"按位置定位"
③ section 过滤失效  按 section_path 过滤时, 这些块永远采不到
```

对后续 **Contextual Retrieval** 是明确障碍:如果章节归属本身不可信,
"给 chunk 注入章节上下文" 就失去了依据。

## 解决方向

```text
方案一(最小改动): 补进块时继承相邻块的 section_path / parent_id
   - 找它前面最近的一个有 section_path 的块, 复制过来
   - 代价: 位置仍然在尾部, 归组会漂移, 但至少 section 不为空
   - 难度: 低

方案二(推荐): 在 parser 阶段就把代码块插到正确位置
   - 解析 Markdown 时记录每个 `<pre>` 在文档里的锚点(如前后文片段)
   - 补齐时按锚点插入, 并复用 heading_stack 推导 section_path
   - 难度: 中

方案三: 改用 markdownify 全程处理, 不依赖 trafilatura 的代码块保留
   - 排查为什么 readability/trafilatura 会丢代码块, 从源头解决
   - 难度: 中高, 且会改变现有 parser 分支行为
```

## 待验证

```text
[ ] 补进块的 section_path 为空, 是否真的造成了 retrieval 质量下降?
    (可用 evaluate harness 对比: 把补充块过滤掉 vs 保留)
[ ] 缺陷 A 在 cleaner 真正过滤时会不会污染 index?
    (当前 10 篇语料零脱钩, 属于潜伏问题)
[ ] 各 parser 丢代码块的具体触发条件是什么?
    (为什么 doc_da52be595f3e 有 12 个 code 块而没触发补充?)
```

## 关联

- `docs/block_structure_and_blockification.md` 第 3.4 节、第 8 节结论 3
- `issues/07_pdf_section_noise.md` —— 同属 "section 不可信" 问题族,但根因不同(07 是 PDF 伪标题,08 是位置错乱)
