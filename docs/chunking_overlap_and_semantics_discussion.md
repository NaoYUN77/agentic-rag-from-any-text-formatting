# Block → Chunk 组织方式 / overlap 取舍 / content 间语义

日期：2026-09-20
状态：**已落地**（2026-09-21 按本文结论执行：取消 overlap、删除滑动窗口参数）

> **执行结果（2026-09-21）**
>
> 本文第二节的结论已实施：
>
> | 项 | 取消前 | 取消后 | 变化 |
> |---|---|---|---|
> | chunks | 117 | **113** | −4 |
> | 总 token | 40,069 | **34,662** | **−5,407 (−13.5%)** |
> | dense / sparse / rrf hit@5 | 0.941 / 0.765 / 0.971 | **完全一致** | 持平 |
> | rrf recall@20、MRR@20 | 1.000 / 0.801 | **完全一致** | 持平 |
>
> 减少的 token 数（5,407）与本文独立测得的 overlap token 数**逐位相同**；
> 各文档片段覆盖字符数不变（无内容丢失）。
> → **overlap 对检索质量没有可测收益，纯属成本**，这个判断被数据证实了。
>
> 同时删除了 `window_tokens`（从未参与计算的死参数）。
> **第三节的语义方向（方案 A/B/C）仍未实施**，属待办。
>
> 下方保留原始分析，供追溯当时的推理依据。

> 本文回答三个问题：
> 1. block 里的 content 到底怎么组织成 chunk？现在这套规则有没有问题？
> 2. `overlap_tokens=400` 还需要吗？该不该取消？
> 3. **content 之间的语义**该怎么做？

---

## 一、现状：block → chunk 的实际规则

### 1.1 五步流水线

```text
① 分组   _group_blocks()   按 scope 键（section_path[0]）分桶 → 每个桶 = 一个 parent
② 展开   _block_pieces()   每个 block 变成 1..N 个 piece
③ 累积   主循环             贪心塞 piece 直到超 chunk_tokens(800)
④ 收口   emit()             成 chunk；决定是否携带 overlap
⑤ 定型   _make_chunk()      渲染全文/稀疏文本/位置/fragments/quality
```

### 1.2 每步的关键约束

| 步骤 | 规则 | 代码位置 |
|---|---|---|
| 分组 | 同 `section_path[0]` 归一组；heading 无祖先时用**自身文本**当 scope | `_scope()` |
| 原子块 | `{code,formula,table,image}` **永不拆**；超限就独占一个 chunk | `_block_pieces()` |
| 文本块 | 超 `chunk_tokens` 时用 `SentenceSplitter` 按**句子边界**切 | `_split_text_block()` |
| heading | 遇到新 heading 且**已有正文** → 强制 `emit(carry_overlap=False)` | 主循环 L521 |
| 超限 | 先 emit，再从上一块尾部**回填** overlap（预算不足则裁减） | L523-530 |
| 空块 | 纯 heading 分组的 chunk 正文为空 → **丢弃**，不产出空壳 parent | `_make_chunk()` 返回 None |

### 1.3 一个容易忽略的设计：标题不进正文

`strip_headings=True`（默认）时，heading **完全不进 `full_text`**，只存 `metadata.headings`。
实测依据（已在上一轮验证）：

- 标题只占全部 token 的 **1.39%**，剥离成本极低
- 剥离对**检索指标零影响**（逐位相同）
- gold 反查只读 `fragments[].text`，不受影响

所以"chunk 里没有标题、只有正文"是**有意为之**，不是 bug。结构信息靠 `section_path` +
`headings` 在**生成阶段**补回（`build_context` 的"背景:"行）。

---

## 二、overlap 该不该取消？—— 实测数据

### 2.1 全局数字（117 chunks）

| 指标 | 值 |
|---|---|
| 带 overlap 的 chunk | **14 / 117 (12.0%)** |
| overlap 占用 token | **5,407 / 40,069 (13.5%)** |

⚠️ 这个数字**比之前"8/10 篇根本不触发"的印象高得多** —— 之前的判断是按"文档数"口径，
而真实分布是**双峰**的：少数文档几乎 100% 带 overlap，多数文档 0%。

### 2.2 双峰分布（关键发现）

| 文档 | heading 块数 | 带 overlap 数 | overlap 占该文 token |
|---|---|---|---|
| anthropic_agent_skills | 7 | 0 | 0.0% |
| anthropic_building_effective_agents | 18 | 0 | 0.0% |
| anthropic_code_execution_mcp | 12 | 0 | 0.0% |
| anthropic_context_engineering | 7 | 1 | 9.8% |
| anthropic_contextual_retrieval | 15 | 0 | 0.0% |
| anthropic_demystifying_evals | 19 | 0 | 0.0% |
| anthropic_multi_agent_research | 8 | 2 | 14.8% |
| **openai_agents_api** | **0** | **4** | **47.4%** |
| **openai_model_misalignment** | **0** | **5** | **43.2%** |
| anthropic_scaling_storage | 9 | 2 | 17.4% |

### 2.3 机制：overlap 与 heading 是**互补的两套切分信号**

相关性是**精确对应**的：

- **有 heading 的文档** → 章节本身就是作者做的语义切分，heading 强制断句，
  overlap 几乎不触发（0%~1 个）。**靠结构切。**
- **无 heading 的文档** → 没有任何结构信号，只能靠长度贪心 + overlap 兜上下文。
  **靠 overlap 切。**

`openai_agents_api`（0 heading）与 `openai_model_misalignment`（0 heading）正是
之前记录过的那两篇"天然对照实验"文档 —— 它们的 chunk 尺寸最均匀（中位 754/787），
`<100 token` 的碎块为 0，代价就是 **overlap 占到该文 token 的 43%~47%**。

### 2.4 overlap 的真实代价（实测）

以 `openai_agents_api` 为例：

```text
该文总计入 token = 3,373
其中 overlap     = 1,600 (47.4%)
去重后唯一 token  = 1,773
```

**意味着近一半的 token 预算花在重复内容上。** 且重叠文本会**同时进入两个 chunk 的向量**
→ 同一段内容可能被检索命中两次，占用 `candidate_k` 名额、挤掉真正不同的候选。

### 2.5 但也确实在起作用（实测片段）

```text
chunk 1 结尾 : ...compaction logic.
chunk 2 开头 : compaction logic.\n\nThe Agents API helps agents find...
                                    ↑ 上一块结尾被正确携带过来
```

所以 overlap **不是死代码**，它在无结构文档上确实保证了跨块指代的完整性。

### 2.6 结论：不该无脑取消，但也**不该保持 400 这么高**

| 方案 | 判断 |
|---|---|
| 保持 `overlap_tokens=400` | ❌ 无结构文档上重复率达 47%，且污染向量召回 |
| **全部取消（=0）** | ⚠️ 有结构文档无影响（本来就不触发）；**无结构文档会丢跨块上下文** |
| **降到一个合理值** | ✅ 建议方向 |

**推荐：`overlap_tokens` 不作为全局常量，而应与结构信号挂钩：**

- 文档**有 heading 结构** → overlap 设 0（heading 已提供边界语义，overlap 无意义且污染纯度）
- 文档**无 heading 结构** → overlap 保留，但降到 **100~150 token**（够接住"这个参数…"
  这类指代句即可，400 明显过量）

其余论证：overlap 的收益是"跨块指代完整"，而**指代通常只需一句**就能锚定，
400 token 相当于带上了一整段。**实测证据**（`openai_agents_api` 的 4 个 overlap chunk）：

| chunk | overlap 前缀含句子数 | 首句（真正接住指代的那句） |
|---|---|---|
| 1 | 2 | `B, Modal, Oracle, Runloop, and Vercel, to provide…` |
| 2 | 4 | `compaction logic.` ← 只有这半句在接指代 |
| 3 | 4 | `only the relevant results back into context.` |
| 4 | 6 | `in production.` |

即：**400 token 平均携带 4 句，但真正起作用的只有第 1 句。** 100 token 足以覆盖
"上一句 + 半句"，正好落在接指代所需的最小充分量上（本估算建议在 A/B 中一并扫
`100 / 150 / 200` 三个值确认）。

### 2.7 这个结论**可以用现有 qrels 裁决**

好消息：`qrels` 恰恰覆盖了那两篇无 heading / 高 overlap 的文档
（`openai_agents_api` 2 条 gold，`openai_model_misalignment` 3 条 gold），
所以 **overlap=400 vs 100 vs 0 的 A/B 是能跑的**，不需要新标注。

建议做法：加一个 `--overlap-tokens` 扫描（复用现有 index 结构，只改切块参数），
对比 `hit@5 / recall@20`，**尤其看 recall** —— overlap 的作用本来就是"提高召回完整性"。

---

## 三、content 之间的语义怎么做？

这是三个问题里最开放的一个。先厘清**现在缺什么**，再给三层方案。

### 3.1 现状：我们目前**完全没有用语义做任何事**

- 切分边界来自：**结构**（heading）+ **长度**（token 上限）+ **标点**（`SentenceSplitter` 的句子边界）
- `window_tokens=400` 参数被接收但**从未参与计算**（见 `chunker.py` 类 docstring 与 issues/09）
- 规划文档设想的"滑动窗口算 embedding 语义距离、在距离峰值处切分"**从未实现**

所以"content 之间语义"目前是**空白**，不是"做得不好"。

### 3.2 三层方案（按投入/收益排序）

#### 方案 A：把 `window_tokens` 真正用起来 —— 块间语义距离切分

思路（就是当初规划、但没实现的那件事）：

```text
对每个 piece 算 embedding
→ 相邻 piece 的余弦距离构成序列
→ 在距离"峰值"处切分（语义断层 = 话题切换）
→ 距离平坦处可跨 block 合并（即使跨 section）
```

代价与风险（**必须诚实说明**）：

1. **需要为每次切块调用 embedding API** —— 成本从"离线零成本"变成"按 token 付费"。
   当前 40K token 建库 0.02 元，可接受；但重建频率高时要算。
2. **粒度问题**：piece 级别可能是几十个 token，用 embedding 算距离噪声很大
   （已用 `HashingEmbeddings` 离线验证链路可跑，但哈希嵌入**不是语义模型**，
   实测相似度区分度很弱：同一话题 0.4~0.6，话题切换 0.2~0.3，边界模糊）。
   → 真正上线需要**换成 `qwen3.7-text-embedding`**才有效。
3. **它切出来的边界会和 heading 冲突** —— 这是最需要想清楚的：
   作者已经用 heading 标注了话题边界，再叠一层语义切分，两者不一致时听谁的？

**关键判断**：**语义切分应该只在"没有结构信号"的地方做。**
即：有 heading 的文档继续走结构切分；无 heading 的文档（`openai_agents_api` 这类）
才用语义距离。这样两个信号不打架，且正好补上最弱的那块。

#### 方案 B：Contextual Retrieval —— 给 chunk 加 LLM 生成的上下文前缀

这是**另一个容易和方案 A 混淆**的思路（上一轮讨论已区分过）：

- 方案 A 改的是**边界**（在哪切）
- 方案 B 改的是**向量内容**（切完之后，给每个 chunk 前拼一段 LLM 生成的说明：
  "这段来自 X 文档的第 Y 章，讲的是 Z"）

Anthropic 的 Contextual Retrieval 就是这个，官方数据是**检索失败率降 49%**。
我们语料里第一篇 `anthropic_contextual_retrieval` 讲的就是它 —— **算是自证**。

代价：每个 chunk 要过一次 LLM（生成上下文），有 token 成本 + 延迟。
但可以**只在建库时跑一次**，且能用便宜模型（`qwen-flash`）。

**与现有设计的关系**：我们已经用 `metadata.headings` 在**生成阶段**补了背景
（`build_context` 的"背景:"行）。方案 B 是把同样的信息**提前注入到 embedding 输入**里，
让**检索**也受益 —— 两者不冲突，是"检索侧 vs 生成侧"的分工。

#### 方案 C：块间语义关系显式建模 —— 构建 chunk 图

不只是线性相邻，而是把 chunk 之间的语义关系统统显式化：

- **父子**：已有（`ParentNode` → `child_chunk_ids`）
- **相邻**：已有隐式（`order` 字段），但未用于检索扩展
- **语义相似**：**缺**。可离线用 embedding 建 kNN 图

用途：检索命中一个 chunk 后，沿图**扩展邻居**一起送进上下文（GraphRAG / 邻域扩展）。
收益在**多跳问题**上最明显 —— 而我们的 qrels 里正好有 `multi_hop` 类别（4 条）。

代价：图要维护（新增文档要更新），且**扩展会放大上下文长度**（成本上升，上一轮算过
上下文是生成侧成本的主项）。

### 3.3 优先级建议

| 优先级 | 方案 | 理由 |
|---|---|---|
| **P0** | 先做 overlap 的 A/B（第二节） | 成本最低、qrels 已覆盖、能立刻拿到结论 |
| **P1** | 方案 B（Contextual Retrieval） | 收益有公开数据支撑（-49% 失败率），实现独立、不碰边界规则 |
| **P2** | 方案 A 只用**无结构文档** | 补最弱环节，但要先解决"piece 粒度噪声"与 embedding 成本 |
| **P3** | 方案 C（chunk 图） | 收益主要在多跳，4 条 qrels 样本太少，暂不足以验证 |

### 3.4 一句忠告

**这三个方案都需要指标裁决，而当前 qrels 只有 39 条**（其中 `multi_hop` 仅 4 条）。
在做方案 A/C 之前，**先把评估集扩到能分辨差异的规模**，否则会陷入
"改完不知道有没有变好"的循环 —— 这正是之前踩过的坑（无 qrels 时做优化 = 无法验证的优化）。

---

## 四、总结

### 对三个问题的直接回答

**Q1. content 怎么组织成 chunk？现在有没有问题？**
五步规则：分组 → 展开 → 贪心累积 → 收口 → 定型。核心设计是**用作者的 heading 结构
当边界**（免费且高信噪比）。**规则本身没有明显问题**，实测 117 chunks 的 section
纯度为 100%（无跨 section 的 chunk）。已有的小瑕疵是 14.5% 的 chunk < 100 token，
但它们多是 Acknowledgements/附录这类本身无检索价值的章节，独立存在无害。

**Q2. overlap 需不需要取消？**
**不该无脑取消，但 400 明显偏高。** 实测：全局 12% 的 chunk 带 overlap、占 13.5% token；
但在**无 heading 的文档上占到 43%~47%**，近一半预算浪费在重复内容上，还污染向量召回。
→ 建议**按结构信号分治**：有 heading 的文档 overlap=0；无 heading 的降到 100~150。
**这个 A/B 用现有 qrels 就能跑**（那两篇文档正好有 gold）。

**Q3. content 之间的语义怎么做？**
目前**完全空白**（`window_tokens` 是死参数）。三个可选方向：
- **A. 块间语义距离切分**（把 `window_tokens` 用起来）—— 建议**只用于无结构文档**
- **B. Contextual Retrieval**（给 chunk 加 LLM 上下文前缀）—— 收益最有据，**推荐先做**
- **C. chunk 语义图 + 邻域扩展** —— 面向多跳，但样本不足以验证

**共性前提：先扩评估集。** 39 条 qrels 支撑不了这些细粒度对比。

---

## 附：本次实测数据可复现命令

```bash
cd pipeline
# overlap 触发率与占比（全局 + 按文档）
./.venv_rag/Scripts/python.exe - <<'EOF'
import json, glob, os
for p in sorted(glob.glob('rebuilt_qwen37/*/chunks.jsonl')):
    doc=os.path.basename(os.path.dirname(p))
    n=wo=ovt=tot=0
    for line in open(p,encoding='utf-8'):
        c=json.loads(line); n+=1
        t=c.get('token_count') or 0; tot+=t
        ov=c.get('overlap_from_previous') or 0
        if ov>0: wo+=1; ovt+=ov
    print('%-40s chunks=%3d 带ov=%2d ov_tok=%5d (%.1f%%)' % (
        doc,n,wo,ovt, ovt/tot*100 if tot else 0))
EOF
```
