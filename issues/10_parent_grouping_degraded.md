# 问题 10: parent 分组退化, 章节级上下文实际不可用

**优先级: P1**　**状态: 部分解决 —— 根因之一已修, 但 `_scope()` 设计问题仍在**

## 2026-09-20（第二次）: URL/HTML 路径的栈 bug 修好后, parents 16 -> 54

`issues/11` 新发现的**第三个成因**（`heading_stack` 拿"栈深度"比"层级"，
导致同级标题层层嵌套）修好后，10 篇 URL 语料的实测：

```text
                    修前    修后
parents              16     54
不同顶层 scope        9     41
section_path 变化     —     80/117 chunks (68%)
```

**这直接解释了本 issue 开头的「10 篇里 10 篇都是 parents ≤ 2」**：
根因不只是 `_scope()` 只取 `section_path[0]`，而是
**`section_path[0]` 本身就被这个栈 bug 污染了** ——
同级标题互相嵌套使顶层前缀塌缩成极少数几个值。

### 但这**没有完全解决**本 issue

- `_scope()` 仍然只取 `section_path[0]`；在"第一层只有一个值"的文档上
  依然会退化（本 issue 开头的分析仍成立）
- 而且**检索指标没有变化**（chunk 边界未变，只是 metadata 变好）
  → parent 分组变好**不等于**检索变好，本 issue 的待验证项依然待验证

## 2026-09-20（第一次）: PDF 路径 parent 数 10 -> 14（改善，但未解决）

修 `issues/11` 的 PDF 路径（字号层级用 outline 实测标定）后，
同一篇 PDF 的分组粒度有实质改善：

```text
                    标定前   标定后
顶层 scope 数         11      13
parents / chunks    10 / 88  14 / 90
摘要正文 section_path  [Red Hat..., 负载平衡器管理, 摘要]   [摘要]
```

原因: 前置页标题原先被字号排名判成 lv6 并挂在书名下，
标定后按 outline 约定落到 lv1，成为独立 scope。

**但这没有解决本问题。** 本问题的核心是 `_scope()` 只取
`section_path[0]`，在"整篇文档第一层只有一个值"时必然退化 ——
上面 10 篇 URL 语料正是这种情况，而它们**没有 outline 可用**，
所以不适用本次的标定方案。本问题仍需按下面的方案一/二独立处理。

## 现象

`ParentNode` 的设计目的是「单 parent 挂多 child, 检索到 leaf 后返回 parent 的章节上下文」。
但实测 parent 分组**几乎退化成"整篇文档一个 group"**。

实测数据(`index_artifacts/phase0_combined/chunks.jsonl`,123 chunks / 10 篇):

```text
总 parent 数   : 16    ← 10 篇文档只有 16 个 parent
section_path 为空: 17 chunks (13.8%)

每篇文档的 parent 数:
  doc_1d0b6b72bdf2  parents=2  chunks=19
  doc_1d3b6982165d  parents=1  chunks=11   ← 整篇只有 1 个 parent
  doc_38a690a6ba57  parents=2  chunks=11
  doc_4a29a797f7d7  parents=1  chunks=6
  doc_58b1b15bbd49  parents=1  chunks=5
  doc_5c3e74c436ef  parents=2  chunks=17
  doc_723b16a48c5d  parents=2  chunks=11
  doc_978f022cd39f  parents=1  chunks=22
  doc_af05975673c5  parents=2  chunks=8
  doc_da52be595f3e  parents=2  chunks=13
```

**10 篇里 10 篇都是 parents ≤ 2**。parent 的 chunk 数分布:

```text
doc_978f022cd39f_p0002   22 chunks   ← 单 parent 挂了 22 个 chunk
doc_1d0b6b72bdf2_p0002   18 chunks
doc_5c3e74c436ef_p0002   15 chunks
doc_da52be595f3e_p0002   12 chunks
doc_1d3b6982165d_p0001   11 chunks
...
```

## 根因

`chunker.py:200-212` 的分组逻辑:

```python
@staticmethod
def _scope(block: DocumentBlock, fallback: str) -> str:
    if block.section_path:
        return block.section_path[0]        # ← 只取【第一层】
    return fallback or "文档开头"

def _group_blocks(self, artifact):
    groups = OrderedDict()
    fallback = artifact.title or "文档开头"
    for block in sorted(artifact.blocks, key=lambda b: b.order):
        scope = self._scope(block, fallback)
        groups.setdefault(scope, []).append(block)
    return groups
```

**问题在于 `section_path[0]` 只取顶级标题**。而实测每篇文档的 `section_path` 第一层,
基本只有一个值(见问题 11: 第一篇文档的标题层级不是 lv1),于是:

```text
doc_978f022cd39f_p0002 的真实 section_path 组合:
  3 x  Introduction / The structure of an evaluation
  2 x  Introduction / Going from zero to one: a roadmap to great evals / Design the eval harness and graders
  1 x  Introduction
  1 x  Introduction / Why build evaluations?
  1 x  Introduction / How to evaluate AI agents
  1 x  Introduction / How to evaluate AI agents / Types of graders for agents
  ...
```

**22 个 chunk 覆盖了文档的每一个子章节**,只因为它们的第一层都是 `Introduction`。
`p0002` 不是"Introduction 这一节",而是**整篇文档正文**;`p0001` 则是文档开头的引言段落。

也就是说:

```text
p0001 = 文档开头(无 section_path 的块)
p0002 = 剩下的全部
```

## 影响

```text
① Parent expansion 失去意义
   设计意图是"用 parent 补章节上下文", 但 parent 现在≈整篇文档
   → 展开 parent 等于把整篇文档塞进 context, 无法聚焦
   → 这也是 issue 中"Parent expansion 未实现"的一个隐性依赖:
     不是没实现, 而是【当前的 parent 粒度下实现了也没用】

② Contextual Retrieval 的前置障碍
   后续要做的 Anthropic 风格 Contextual Retrieval, 核心是"给 chunk 注入
   它所在的章节上下文"。章节归属本身退化成"整篇", 注入无有效信息
   → 必须先修 parent 粒度, 否则 Contextual Retrieval 的实验结论不可信

③ section 过滤能力丧失
   按 parent / section 做过滤或加权时, 只有"开头"和"其余"两档, 区分度≈0
```

## 解决方向

```text
方案一(推荐): 分组时用【完整 section_path】而非 section_path[0]
   - 把 _scope 改为返回 "/".join(section_path) 或取【最细粒度】的一层
   - 代价: parent 数量会大幅增加(接近每个 heading 一个 parent)
   - 需同时考虑: 过细也不利于 "返回章节上下文", 可能要取"第 2 层"
   - 难度: 低(改 1 个函数)

方案二: 引入层级阈值
   - 取 section_path 的【某个深度】(如第 1 层有多个值时用第 2 层)
   - 本质是动态选择合适粒度, 更贴合"章节上下文"语义
   - 难度: 中

方案三: 放弃 parent, 改用"相邻 chunk 拼接"
   - 检索到 leaf 后取前后各 N 个 chunk 作为上下文, 不依赖 section 结构
   - 优点: 绕开 section 质量问题; 缺点: 上下文可能跨主题
   - 难度: 低; 但与 Parent/Child 的设计意图冲突
```

## 追加发现: lv1 标题全部塌缩成同一组, 产出"章节目录"垃圾 chunk（2026-09-18）

在 PDF 链路上实测到一个本问题的新表现, 而且暴露了 Block 模型的一个信息缺失。

### 现象

`experiments/pdf_test/phase0_pdfplumber/chunks.jsonl` 的 `c0001`:

```text
chunk_id   doc_b73448ca88c6_c0001
tokens     135
fragments  b0001, b0011, b0013, b0021, b0073, b0206, b0263, b0334, b0405, b0441, b0443

dense_text:
  # Red Hat Enterprise Linux 7
  # 目录
  # 第 1 章 LOAD BALANCER 概述
  # 第 2 章 KEEPALIVED 概述
  # 第 3 章 为 KEEPALIVED 设置负载均衡器先决条件
  # 第 4 章 使用 KEEPALIVED 初始负载均衡器配置
  # 第 5 章 HAPROXY 配置
  # 附录 A. 示例配置：...
  # 附录 B. 修订历史记录
  # 索引
  # 索引

sparse_text: ''   ← 空
```

**这是一个"章节目录"chunk: 只有 11 个标题拼在一起, 没有任何正文。**
它会进入 dense 索引(有向量), 但不进 sparse(标题不在 SPARSE_TYPES)。

### 根因: heading 的 section_path 不含自己, 导致 _scope 无法按自身章节分组

两个设计事实叠加:

```text
① markdown.py:134   heading 块的 section_path = heading_stack[:-1]  ← 不含自己
                    所以所有 lv1 标题的 section_path 都是 []

② chunker.py:201    _scope(block) 在 section_path 为空时返回 fallback
                    fallback = artifact.title
```

于是**所有 11 个 lv1 标题的 scope 都等于文档标题**, 全部落进同一个 parent
(`p0001`), 被 chunker 累积成 1 个 chunk。

实测(51 页 PDF):

```text
section_path 为空的 block = 11 个, 全部是 lv1 heading
   b0001 p1  Red Hat Enterprise Linux 7
   b0011 p5  目录
   b0013 p7  第 1 章 LOAD BALANCER 概述
   b0021 p8  第 2 章 KEEPALIVED 概述
   b0073 p16 第 3 章 ...
   b0206 p29 第 4 章 ...
   b0263 p36 第 5 章 ...
   b0334 p42 附录 A. ...
   b0405 p48 附录 B. ...
   b0441 p49 索引
   b0443 p49 索引      ← 重复标题
```

### 这暴露的模型缺陷

```text
对【正文块】:  section_path = 完整祖先链   ← 有用
对【标题块】:  section_path = 祖先链(不含自己) ← 无法按自身章节分组

结果: chunker 拿不到"这个标题属于哪一节"的信息,
      只能退化成按文档标题分组。
```

**这是 `_scope` 之外的第二层原因** —— 即使把 `_scope` 改成用完整路径,
lv1 标题的 section_path 仍是空, 依然会塌缩到 fallback。

### 影响

```text
① 产生无意义的 dense 向量
   11 个标题的拼接向量, 与任何真实问题都不相似, 但会占用索引位

② 该 chunk 不进 sparse
   标题不在 SPARSE_TYPES → sparse_text 空 → 只进 dense
   (这也解释了此前"123 dense / 123 sparse 有 6 个差额"的现象)

③ 数量小但性质明确
   实测 88 个 chunk 中只有 1 个是纯标题, 占比 1.1%
   但它是"设计缺陷的显式体现", 不是随机噪声
```

### 建议的修法

```text
修法 A(推荐): 让标题块的 section_path 含自己
   - markdown.py 里 heading 的 section_path 改为 list(heading_stack)(含自己)
   - 风险: 会改变现有 8 条不变式中的一条, 需回归
   - 收益: _scope 能正确按章节分组; 同时修掉本问题

修法 B: chunker 显式识别纯标题 chunk 并丢弃
   - emit 时若 current 全是 heading → 不产出 chunk
   - 难度: 低; 但只是掩盖症状

修法 C: _scope 对 heading 类型特殊处理
   - 若 block.type == "heading", 用 block.text 作为 scope
   - 难度: 低; 但不改模型, 属打补丁
```

**推荐 A**, 因为它同时解决"标题无法定位自身章节"这个模型缺陷,
而 B/C 都只是绕过。

**推荐先做方案一**,因为它同时降低了对问题 11(heading 层级不准)的依赖 ——
但**根因仍在问题 11**:如果 `section_path` 的第一层是错的,换用完整路径也只是
"在错误前缀下切得更细",无法还原真实章节层级。

**依赖关系:问题 10 的正确修复依赖问题 11 先解决。**

## 关联

- `issues/11_heading_level_extraction.md` —— **根因, 必须先修**
- `docs/block_structure_and_blockification.md` 第 8 节结论 2
- `docs/plan_vs_implementation.md` 偏差 A

## 待验证

```text
[ ] 改成完整 section_path 后, parent 数从 16 变成多少? 分布是否合理?
[ ] parent expansion 展开后的 context 长度是多少? 会不会撑爆 prompt?
[ ] 在评估集上, "展开 parent" vs "不展开" 的 recall / 答案质量差多少?
    (这也是验证 Parent expansion 值不值得做的关键实验)
```
