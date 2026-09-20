r"""把 phase0_url_draft.jsonl 从 13 条扩到 ~39 条。

背景
----
docs/eval_harness_design.md Stage 0 要求 30~40 条 query、覆盖 5 个 category、
冻结 dev/test。2026-09-18 的初稿只有 13 条(全 zh、multi_hop 仅 1 条、
4 篇文档完全没覆盖)。本脚本补齐这些缺口。

为什么用脚本生成而不是手写 JSON
--------------------------------
gold 的 text 是主锚点, 必须**逐字**存在于语料 block 里。手抄长句必然
抄错标点/弯引号/空格(实测踩过: 归一化 bug 就是手抄引发的排查)。
所以每条 gold 在这里声明为 (block, 起始 marker, 结束 marker),
由脚本从 artifact.json **原文切片**, 逐字正确由构造保证。

marker 唯一性强制校验: marker 在 block 内必须恰好出现一次,
否则抛错 —— 防止语料更新后切片悄悄漂移。

幂等: 已存在的 query_id 跳过, 可重复运行。

用法
----
    cd pipeline
    .venv_rag/Scripts/python.exe expand_phase0_qrels.py            # 追加缺失条目
    .venv_rag/Scripts/python.exe expand_phase0_qrels.py --dry-run  # 只检查不写
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.qrels import load_qrels, summarize  # noqa: E402

REBUILT = Path("experiments_rebuilt")
QRELS_PATH = Path("eval/qrels/phase0_url_draft.jsonl")
ANNOTATED_AT = "2026-09-19"

# 语料 -> source_uri (取自 experiments_rebuilt/<doc>/artifact.json 的 source_uri 字段)
DOC_URI = {
    "anthropic_agent_skills":
        "https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills",
    "anthropic_building_effective_agents":
        "https://www.anthropic.com/engineering/building-effective-agents",
    "anthropic_code_execution_mcp":
        "https://www.anthropic.com/engineering/code-execution-with-mcp",
    "anthropic_context_engineering":
        "https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents",
    "anthropic_contextual_retrieval":
        "https://www.anthropic.com/engineering/contextual-retrieval?utm_source=chatgpt.com",
    "anthropic_demystifying_evals":
        "https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents",
    "anthropic_multi_agent_research":
        "https://www.anthropic.com/engineering/multi-agent-research-system",
    "openai_agents_api":
        "https://openai.com/index/introducing-the-agents-api/",
    "openai_model_misalignment":
        "https://openai.com/index/model-misalignment-reporting-framework/",
    "openai_scaling_storage":
        "https://openai.com/index/scaling-storage-one-billion-users-part-one/",
}

# 每条 gold: (doc, block_id, start_marker, end_marker)
# 切片 = block.text[start : end + len(end_marker)] —— 含两个 marker 本身。
G = lambda doc, bid, s, e: (doc, bid, s, e)  # noqa: E731

AG = "anthropic_building_effective_agents"
CE = "anthropic_context_engineering"
CR = "anthropic_contextual_retrieval"
CX = "anthropic_code_execution_mcp"
EV = "anthropic_demystifying_evals"
MA = "anthropic_multi_agent_research"
SC = "openai_scaling_storage"
OA = "openai_agents_api"
OM = "openai_model_misalignment"

NEW_ITEMS = [
    # ---- building_effective_agents (此前 0 条) ----
    dict(query_id="ag_01", query="Anthropic 如何区分 workflow 和 agent？",
         query_lang="zh", category="semantic", split="dev",
         notes="两条定义在同一列表块: 预定义代码路径 vs 动态自主控制",
         golds=[G(AG, "b0005", "Workflows are systems", "predefined code paths."),
                G(AG, "b0005", "Agents , on the other hand", "accomplish tasks.")]),
    dict(query_id="ag_02", query="什么时候该选 agents 而不是 workflows？",
         query_lang="zh", category="semantic", split="test",
         notes="可预测性/一致性 vs 灵活性/模型驱动决策",
         golds=[G(AG, "b0009", "workflows offer predictability", "needed at scale.")]),
    dict(query_id="ag_03", query="What does Anthropic recommend when building applications with LLMs?",
         query_lang="en", category="semantic", split="dev",
         notes="simplest solution first, 必要时才加复杂度",
         golds=[G(AG, "b0008", "we recommend finding the simplest solution possible",
                  "increasing complexity when needed.")]),
    dict(query_id="ag_04", query="How does the orchestrator-workers workflow operate?",
         query_lang="en", category="exact", split="test",
         notes="central LLM 分解任务/委派 worker/综合结果",
         golds=[G(AG, "b0043", "In the orchestrator-workers workflow", "synthesizes their results.")]),
    dict(query_id="ag_05", query="orchestrator-workers 适合什么任务？它和 parallelization 的关键区别是什么？",
         query_lang="zh", category="multi_hop", split="dev",
         notes="适用场景与区别在不同句子, 需组合",
         golds=[G(AG, "b0045", "This workflow is well-suited for complex tasks",
                  "predict the subtasks needed"),
                G(AG, "b0045", "the key difference from parallelization", "based on the specific input.")]),

    # ---- context_engineering (此前 1 条) ----
    dict(query_id="ce_02", query="Anthropic 现在倾向用哪句话定义 agent？",
         query_lang="zh", category="semantic", split="test",
         notes="LLMs autonomously using tools in a loop",
         golds=[G(CE, "b0028", "we’ve gravitated towards a simple definition", "using tools in a loop.")]),
    dict(query_id="ce_03", query="structured note-taking（agentic memory）是什么？",
         query_lang="zh", category="exact", split="dev",
         notes="笔记写到上下文窗口外, 之后按需拉回",
         golds=[G(CE, "b0046", "Structured note-taking, or agentic memory",
                  "outside of the context window.")]),
    dict(query_id="ce_04", query="Why isn't waiting for larger context windows a solution?",
         query_lang="en", category="semantic", split="test",
         notes="任何大小的窗口都会有 context pollution",
         golds=[G(CE, "b0039", "Waiting for larger context windows", "relevance concerns")]),
    dict(query_id="ce_05", query="compaction 和 structured note-taking 分别是怎么工作的？",
         query_lang="zh", category="multi_hop", split="test",
         notes="答案分布在两个不同小节",
         golds=[G(CE, "b0041", "Compaction is the practice", "with the summary."),
                G(CE, "b0046", "Structured note-taking, or agentic memory", "at later times.")]),

    # ---- contextual_retrieval (此前 3 条) ----
    dict(query_id="cr_04", query="What is BM25 and when is it particularly effective?",
         query_lang="en", category="exact", split="dev",
         notes="词法匹配; 对唯一标识符/技术术语最有效",
         golds=[G(CR, "b0013", "BM25 (Best Matching 25)", "unique identifiers or technical terms.")]),
    dict(query_id="cr_05", query="知识库小于多少 token 时可以不做 RAG 直接放进 prompt？",
         query_lang="zh", category="exact", split="test",
         notes="200,000 tokens / 约 500 页",
         golds=[G(CR, "b0006", "If your knowledge base is smaller than 200,000 tokens",
                  "no need for RAG")]),

    # ---- code_execution_mcp (此前 0 条) ----
    dict(query_id="cx_01", query="接入的工具数量越来越多时，直接用 MCP 会遇到什么问题？",
         query_lang="zh", category="semantic", split="dev",
         notes="upfront 加载全部工具定义 + 中间结果过上下文, 变慢变贵",
         golds=[G(CX, "b0003", "as the number of connected tools grows",
                  "slows down agents and increases costs.")]),
    dict(query_id="cx_02", query="agent harness 怎么处理导入 Salesforce 这类敏感数据？",
         query_lang="zh", category="mixed", split="test",
         notes="自动 tokenize 敏感数据, 跨行示例",
         golds=[G(CX, "b0046", "the agent harness can tokenize sensitive data",
                  "into Salesforce.")]),
    dict(query_id="cx_03", query="What operational overhead does code execution add over direct tool calls?",
         query_lang="en", category="semantic", split="test",
         notes="沙箱/资源限制/监控等基础设施要求",
         golds=[G(CX, "b0057", "Running agent-generated code requires", "operational overhead")]),

    # ---- demystifying_evals (此前 2 条) ----
    dict(query_id="ev_03", query="为什么不能只看 eval 分数就下结论？",
         query_lang="zh", category="semantic", split="dev",
         notes="要读 transcript, 检查 grading/任务歧义/harness 约束",
         golds=[G(EV, "b0095", "we do not take eval scores at face value",
                  "reads some transcripts.")]),

    # ---- multi_agent_research (此前 2 条) ----
    dict(query_id="ma_03", query="Research 系统的多智能体架构用了什么模式？",
         query_lang="zh", category="mixed", split="dev",
         notes="orchestrator-worker: lead agent + 并行 subagents",
         golds=[G(MA, "b0013", "Our Research system uses a multi-agent architecture",
                  "operate in parallel.")]),

    # ---- scaling_storage (此前 2 条) ----
    dict(query_id="sc_03", query="为什么说并发不等于 CPU 并行？Habitat 哪些职责是 CPU 密集的？",
         query_lang="zh", category="semantic", split="dev",
         notes="asyncio 不解决 GIL; routing/压缩/加密/校验和等是 CPU 重活",
         golds=[G(SC, "b0031", "Asyncio helps Python execute I/O-bound workloads",
                  "provide CPU parallelism."),
                G(SC, "b0031", "Habitat handles many CPU-heavy responsibilities",
                  "and hedging.")]),
    dict(query_id="sc_04", query="LIFO 和 FIFO 在连接复用上各是什么效果？",
         query_lang="zh", category="multi_hop", split="test",
         notes="答案分散在 LIFO/FIFO 两节",
         golds=[G(SC, "b0051", "LIFO encourages more work", "same slower servers."),
                G(SC, "b0055", "FIFO maintains more active connections",
                  "fairly across all servers.")]),

    # ---- openai_agents_api (此前 0 条) ----
    dict(query_id="oa_01", query="Agents API 现在是什么发布状态？有没有额外费用？",
         query_lang="zh", category="exact", split="dev",
         notes="public beta; 无额外费用, 只付 token/工具",
         golds=[G(OA, "b0019", "Agents API is available in public beta",
                  "no additional fees")]),
    dict(query_id="oa_02", query="What do you specify to create a production-ready agent in one API call?",
         query_lang="en", category="mixed", split="test",
         notes="task/model/tools/environment 四要素",
         golds=[G(OA, "b0003", "you can create a production-ready agent",
                  "tools, and environment")]),

    # ---- openai_model_misalignment (此前 0 条) ----
    dict(query_id="om_01", query="这次一共发布了几份模型失准行为的报告？",
         query_lang="zh", category="exact", split="dev",
         notes="six reports",
         golds=[G(OM, "b0011", "we’re publishing six reports", "evaluation of our models")]),
    dict(query_id="om_02", query="Who can flag a misalignment example for investigation?",
         query_lang="en", category="semantic", split="test",
         notes="任何 OpenAI 员工都可发起",
         golds=[G(OM, "b0013", "Any OpenAI employee may flag", "public disclosure.")]),
    dict(query_id="om_03", query="被标记的失准案例会被分到哪三条轨道？",
         query_lang="zh", category="exact", split="dev",
         notes="Ready for Disclosure / Minor / Larger (Slow Track)",
         golds=[G(OM, "b0015", "assigned to one of three tracks", "(“Slow Track”).")]),

    # ---- unanswerable 补充(候选均已 grep 验证语料无对应内容) ----
    dict(query_id="un_03", query="Agents API 的速率限制（RPM/QPS）是多少？",
         query_lang="zh", category="unanswerable", split="test",
         notes="已验证: 全语料无 rate limit/RPM/QPS 字样", golds=[]),
    dict(query_id="un_04", query="How much does it cost to fine-tune Claude Sonnet 4.5?",
         query_lang="en", category="unanswerable", split="test",
         notes="已验证: 全语料无 fine-tune 定价内容", golds=[]),
    dict(query_id="un_05", query="Habitat 的存储节点用的是什么型号的 SSD 磁盘？",
         query_lang="zh", category="unanswerable", split="dev",
         notes="已验证: 全语料无 SSD/磁盘型号字样; 注意不要换成云平台问题(语料含 Azure)",
         golds=[]),
]


def load_block(doc: str, block_id: str, cache: dict) -> dict:
    if doc not in cache:
        art = json.loads((REBUILT / doc / "artifact.json").read_text(encoding="utf-8"))
        cache[doc] = {b["block_id"]: b for b in art["blocks"]}
    return cache[doc][block_id]


def slice_span(block_text: str, start_marker: str, end_marker: str,
               doc: str, block_id: str) -> tuple[str, list[int]]:
    """按 marker 从原文切出逐字片段。marker 必须在 block 内唯一出现。"""
    for name, marker in (("start", start_marker), ("end", end_marker)):
        n = block_text.count(marker)
        if n != 1:
            raise SystemExit(
                f"[FAIL] {doc}/{block_id} 的 {name} marker 出现 {n} 次(要求 1 次): "
                f"{marker[:60]!r}")
    i = block_text.index(start_marker)
    j = block_text.index(end_marker, i)
    span = block_text[i: j + len(end_marker)]
    if not (40 <= len(span) <= 220):
        raise SystemExit(
            f"[FAIL] {doc}/{block_id} 切片 {len(span)} 字符, 超出 40~220: {span[:80]!r}")
    return span, [i, i + len(span)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="只检查与预览, 不写文件")
    args = ap.parse_args()

    existing = load_qrels(QRELS_PATH)
    have = {it.query_id for it in existing}
    todo = [it for it in NEW_ITEMS if it["query_id"] not in have]
    print(f"现有 {len(existing)} 条, 本次新增 {len(todo)} 条 "
          f"(跳过已存在 {len(NEW_ITEMS) - len(todo)})")

    cache: dict = {}
    lines = []
    for item in todo:
        gold_dicts = []
        for doc, bid, s_marker, e_marker in item["golds"]:
            block = load_block(doc, bid, cache)
            text, span = slice_span(block["text"], s_marker, e_marker, doc, bid)
            gold_dicts.append({
                "text": text,
                "source_uri": DOC_URI[doc],
                "block_id": bid,
                "span": span,
                "source_doc": doc,
            })
        lines.append(json.dumps({
            "query_id": item["query_id"],
            "query": item["query"],
            "query_lang": item["query_lang"],
            "gold": gold_dicts,
            "category": item["category"],
            "split": item["split"],
            "notes": item["notes"],
            "annotated_by": "draft",
            "annotated_at": ANNOTATED_AT,
        }, ensure_ascii=False))

    for ln in lines:
        d = json.loads(ln)
        print(f"  {d['query_id']:<6} {d['query_lang']} {d['category']:<12} {d['split']:<4} "
              f"gold={len(d['gold'])}")

    merged = existing + [type(existing[0]).from_dict(json.loads(ln)) for ln in lines]
    print("\n合并后统计:", json.dumps(summarize(merged), ensure_ascii=False))

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return
    if lines:
        with open(QRELS_PATH, "a", encoding="utf-8", newline="\n") as f:
            for ln in lines:
                f.write(ln + "\n")
        print(f"\n已追加 {len(lines)} 条到 {QRELS_PATH}")
    # 写回后整体校验(schema + 重复 id)
    reloaded = load_qrels(QRELS_PATH)
    print(f"重读校验通过: {len(reloaded)} 条")


if __name__ == "__main__":
    main()
