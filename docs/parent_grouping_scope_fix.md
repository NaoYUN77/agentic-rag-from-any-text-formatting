# Parent 分组修复：`_scope()` 改用完整 `section_path`

日期：2026-09-21
对应 issue：`issues/10_parent_grouping_degraded.md`
状态：**已修复（有 heading 的文档）**；无 heading 文档仍待处理

---

## 一、改了什么

### 改动位置

`pipeline/ingest/chunker.py` 的 `BlockAwareHierarchicalChunkBuilder._scope()`。
**一个函数，约 6 行逻辑。**

### 改动前后

```python
# 改前：只取第一层
@staticmethod
def _scope(block, fallback):
    if block.section_path:
        return block.section_path[0]        # ← 只取顶级标题
    if block.type == "heading":
        return block.text
    return fallback or "文档开头"
```

```python
# 改后：用完整路径；标题块要把自己补到路径末尾
@staticmethod
def _scope(block, fallback):
    if block.type == "heading":
        # 标题块的 section_path 是【祖先链, 不含自己】→ 补上自己，
        # 才能与自己章节下的正文块同组
        path = list(block.section_path) + [block.text]
        return " > ".join(str(x) for x in path)
    if block.section_path:
        return " > ".join(str(x) for x in block.section_path)
    return fallback or "文档开头"
```

### 为什么标题要「补上自己」

这是容易改错的地方。`DocumentBlock.section_path` 对标题块和对正文块的语义**不一样**：

```text
第 1 章（lv1 标题）      section_path = []            ← 祖先链，不含自己
第 1 章下的正文块        section_path = ['第 1 章']    ← 含自身章节

第 1 章 > 1.1 节（lv2 标题）  section_path = ['第 1 章']
1.1 节下的正文块             section_path = ['第 1 章', '1.1 节']
```

所以标题块的「自己所属的路径」= `section_path + [自己]`。不做这一步，标题会和自己的正文分家——实测会让 lv1 标题塌缩成一个「章节目录」chunk（只有标题、没有正文、`sparse_text` 为空）。

---

## 二、为什么要改（两种退化机制）

`_scope()` 的返回值决定 `_group_blocks()` 怎么分桶，每个桶产出一个 `ParentNode`。
原实现只取 `section_path[0]`，实测在**两类文档**上退化：

### 类型 A：顶层只有一个标题

`openai_scaling_storage` 有 9 个章节，但它们**全都挂在同一个 lv1 标题下**：

```text
chunks 的 section_path[0] 分布：
   'Rapidly scaling online storage to serve over 1 billion ChatGPT users'  11
```

9 个章节的第一层全是同一个值 → **整篇只剩 1 个 parent**（11 个 chunk 全挂它下面）。

### 类型 B：完全没有 heading

`openai_agents_api` / `openai_model_misalignment` 没有任何 heading 块
→ `section_path` 全为空 → 全部落到同一个 `fallback`
→ **整篇 1 个 parent**。

### 后果

```text
① Parent expansion 失去意义
   parent ≈ 整篇文档 → 展开 parent 等于把整篇塞进 context，无法聚焦

② Contextual Retrieval 的前置障碍
   "给 chunk 注入所在章节上下文" —— 章节归属退化成"整篇"，注入无有效信息

③ section 过滤能力丧失
   只有"文档开头"和"其余"两档，区分度 ≈ 0
```

---

## 三、优化了什么（实测）

同一批 10 篇语料、同一 parser、`chunk_tokens=800`、无 overlap：

### 分组粒度

| 指标 | 改前 | 改后 | 变化 |
|---|---:|---:|---|
| **parents** | 54 | **103** | **+91%** |
| parent token p50 | 396 | **204** | −48% |
| parent token p90 | 1773 | **745** | −58% |
| parent token max | 3788 | **2378** | −37% |
| **单 parent 最大 chunk 数** | **11** | **4** | **−64%** |
| > 1500 token 的 parent | 7 (13%) | **2 (2%)** | −71% |
| > 2000 token 的 parent | 4 (7%) | **1 (1%)** | −75% |

### 逐篇 parents

| 文档 | 改前 | 改后 |
|---|---:|---:|
| **openai_scaling_storage** | **1** | **9** ← 类型 A 修好 |
| anthropic_building_effective_agents | 9 | 19 |
| anthropic_demystifying_evals | 8 | 19 |
| anthropic_contextual_retrieval | 8 | 16 |
| anthropic_code_execution_mcp | 5 | 13 |
| anthropic_agent_skills | 5 | 8 |
| anthropic_context_engineering | 7 | 8 |
| anthropic_multi_agent_research | 9 | 9 |
| openai_agents_api | 1 | 1 ← 类型 B，未修 |
| openai_model_misalignment | 1 | 1 ← 类型 B，未修 |
| **合计** | **54** | **103** |

### ★ 检索指标完全不受影响（关键结论）

| 指标 | 改前 | 改后 |
|---|---|---|
| chunks | 113 | **113** |
| 总 token | 34,662 | **34,662** |
| `full_text` 不同的 chunk | — | **0** |
| `sparse_text` 不同的 chunk | — | **0** |
| 全文集合 SHA256 | `88a644004646ca12` | **`88a644004646ca12`** |

**chunk 内容逐字不变** → 向量不变 → 召回不变 → **无需重跑评估**。

原因：chunk 边界由「遇到 heading 就收口」这条规则决定，而新分组恰好与
heading 边界重合，所以没有多切一刀。

> 这一点很重要：**本修复是纯 metadata 改进**。
> 不要拿检索指标去论证它值不值得修，也不要指望它让检索变好。
> 它的价值在 Parent expansion / Contextual Retrieval / section 过滤
> —— 而这些能力目前都还没实现。

---

## 四、还没解决的（类型 B）

**完全无 heading 的文档仍退化成 1 个 parent**（`openai_agents_api` 3 chunks /
1773 token、`openai_model_misalignment` 4 chunks）。

这类文档**没有可用的结构信号**，`_scope()` 无法改善。可选方向：

```text
方案 A：给 parent 加 token 尺寸上限
        超限的组按顺序拆成多个 parent（如每个 ≤ 1200 token）
        优点：同时能兜住"某个章节本身过大"的情况
        缺点：拆出来的 parent 不再是语义完整的章节

方案 B：无结构文档不产出 parent
        让 parent expansion 回退到"相邻 chunk 拼接"
        优点：不制造假章节
        缺点：需要另一套扩展逻辑

方案 C：保持现状
        承认无结构文档拿不到章节级上下文
```

**建议 A**：它是唯一能同时覆盖「无结构文档」和「超大章节」的规则，
且不改变有结构文档的行为（那些 parent 已在 2378 token 以内）。

---

## 五、验证方式

```bash
cd pipeline

# 1. dry-run 看分组（不调 API）
python -m rebuild_phase0_index --backend qwen --out-dir rebuilt_scope_v1 --dry-run

# 2. 对比 chunk 内容是否逐字不变
#    （全文集合 SHA256 相同 -> 无需重跑评估）

# 3. 重建索引
export DASHSCOPE_API_KEY="$(cat .dashscope_key)"
python -m rebuild_phase0_index --backend qwen \
  --out-dir rebuilt_scope \
  --artifact-dir index_artifacts/phase0_scope \
  --dense-collection phase0_scope_dense \
  --sparse-collection phase0_scope_sparse

# 4. 回归测试
python -m pytest tests/test_format_router.py -q
```

回归测试：`tests/test_format_router.py` 新增
`test_nested_sections_do_not_collapse_into_one_parent`。
**注入旧 `_scope()` 后该测试失败**（实测旧实现 1 个 parent、新实现 3 个），
确认它是有效的护栏。

产物：`index_artifacts/phase0_scope`，collections `phase0_scope_dense` / `phase0_scope_sparse`
（113 chunks / **103 parents**）。
