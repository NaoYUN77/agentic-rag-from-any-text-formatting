"""对照验证: query 侧跨语言处理的三种方案。

背景
----
语料是纯英文(10 篇 / 129 chunks), 用户用中文提问。
现状: 中文 query 进 sparse 通道会直接空转, 因为 BM25 词表全是英文词。

本脚本对比三种方案, 判据是「top-k 是否命中期望文档」:

    A  现状      : dense=中文, sparse=中文
    B  分通道翻译: dense=中文, sparse=英文译文
    C  全翻译    : dense=英文译文, sparse=英文译文

不写入任何索引, 只做只读检索。
在进程内直接调用 HybridRetriever, 所以运行前必须停掉 FastAPI(否则 qdrant_data 被锁)。

用法:
    $env:DASHSCOPE_API_KEY = "sk-..."
    .\\.venv_rag\\Scripts\\python.exe eval_zh_crosslingual.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

from hybrid_retriever import HybridRetriever
from qdrant_client import QdrantClient
from semantic_chunker_demo import build_embeddings

ARTIFACT_DIR = Path("index_artifacts/phase0_combined")
QDRANT_PATH = "qdrant_data"
DENSE_COLLECTION = "phase0_dense"
SPARSE_COLLECTION = "phase0_sparse"
CANDIDATE_K = 20
TOP_K = 5

# 语料文档标题 -> 用于判定「检索是否命中了正确的那一篇」
TITLES = {
    "doc_978f022cd39f": "Demystifying evals",
    "doc_1d0b6b72bdf2": "Building Effective Agents",
    "doc_5c3e74c436ef": "Contextual Retrieval",
    "doc_da52be595f3e": "Code execution with MCP",
    "doc_723b16a48c5d": "Context engineering",
    "doc_38a690a6ba57": "Multi-agent research",
    "doc_1d3b6982165d": "Scaling storage 1B users",
    "doc_af05975673c5": "Agent Skills",
    "doc_4a29a797f7d7": "Model misalignment",
    "doc_58b1b15bbd49": "Agents API",
}

# (query_id, 中文 query, 期望命中的文档标题)
CASES = [
    ("zh01", "Contextual Retrieval 能把检索失败率降低多少？", "Contextual Retrieval"),
    ("zh02", "什么是 agent 的上下文工程？", "Context engineering"),
    ("zh03", "工作流和自主智能体有什么区别？", "Building Effective Agents"),
    ("zh04", "什么是 Agent Skill，它由哪些部分组成？", "Agent Skills"),
    ("zh05", "多智能体研究系统常见的失败模式有哪些？", "Multi-agent research"),
    ("zh06", "BM25 的 k1 和 b 参数是做什么的？", "Contextual Retrieval"),
    ("zh07", "为什么代码执行比直接调用工具更省 token？", "Code execution with MCP"),
    ("zh08", "如何评估 AI agent 的能力？", "Demystifying evals"),
    ("zh09", "重排序在检索中起什么作用？", "Contextual Retrieval"),
    ("zh10", "多智能体系统什么时候不该用？", "Building Effective Agents"),
]

TRANSLATE_MODEL = "qwen-turbo"
TRANSLATE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
_CACHE_PATH = Path("_zh_translations_cache.json")


def translate(text: str, api_key: str) -> str:
    """把中文 query 翻成英文检索式。结果落缓存, 保证可复现。"""
    cache = {}
    if _CACHE_PATH.exists():
        cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    if text in cache:
        return cache[text]

    prompt = (
        "Translate the following Chinese question into a concise English "
        "search query. It will be used for BM25 keyword retrieval over "
        "English technical documents about AI agents and RAG. Keep technical "
        "terms in English (e.g. BM25, RAG, token, reranking, embedding, "
        "context engineering). Output ONLY the query, no explanation.\n\n"
        f"Chinese: {text}\nEnglish:"
    )
    r = requests.post(
        TRANSLATE_URL,
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": "application/json"},
        json={
            "model": TRANSLATE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"翻译失败 {r.status_code}: {r.text[:300]}")
    out = r.choices[0] if hasattr(r, "choices") else None
    text_out = r.json()["choices"][0]["message"]["content"].strip()
    text_out = text_out.strip().strip('"').split("\n")[0].strip()

    cache[text] = text_out
    _CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return text_out


def doc_of(chunk_id: str) -> str:
    return TITLES.get(str(chunk_id).rsplit("_c", 1)[0], "?")


def fuse(dense_hits, sparse_hits, limit):
    return HybridRetriever.rrf_fuse([dense_hits, sparse_hits], k=60, limit=limit)


def main() -> int:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[X] 缺少 DASHSCOPE_API_KEY")
        return 1

    client = QdrantClient(path=QDRANT_PATH)
    retriever = HybridRetriever(
        client=client,
        embeddings=build_embeddings("qwen"),
        dense_collection=DENSE_COLLECTION,
        sparse_collection=SPARSE_COLLECTION,
        artifact_dir=ARTIFACT_DIR,
        rrf_k=60,
    )

    print("=" * 92)
    print("query 侧跨语言方案对照  (语料=英文, query=中文)")
    print("=" * 92)
    print(f"文档数={len(TITLES)}  candidate_k={CANDIDATE_K}  top_k={TOP_K}\n")

    plans = {
        "A 现状(dense中文+sparse中文)": ("zh", "zh"),
        "B 分通道(dense中文+sparse译文)": ("zh", "en"),
        "C 全翻译(dense译文+sparse译文)": ("en", "en"),
    }

    summary = {name: {"top1": 0, "hit5": 0, "mrr": 0.0, "sparse_empty": 0,
                      "purity": 0.0}
               for name in plans}
    rows = []

    for qid, zh, expect in CASES:
        en = translate(zh, api_key)
        print("=" * 92)
        print(f"[{qid}] {zh}")
        print(f"       期望={expect}")
        print(f"       译文={en}")

        for name, (dense_src, sparse_src) in plans.items():
            q_dense = zh if dense_src == "zh" else en
            q_sparse = zh if sparse_src == "zh" else en

            d_hits = retriever.dense_search(q_dense, CANDIDATE_K)
            s_hits, known, unknown = retriever.sparse_search(q_sparse, CANDIDATE_K)

            if not s_hits:
                summary[name]["sparse_empty"] += 1
            fused = fuse(d_hits, s_hits, TOP_K)

            docs = [doc_of(h.payload.get("chunk_id")) for h in fused]
            top1_ok = bool(docs) and docs[0] == expect
            try:
                rank = docs.index(expect) + 1
            except ValueError:
                rank = 0
            hit5_ok = rank > 0

            summary[name]["top1"] += int(top1_ok)
            summary[name]["hit5"] += int(hit5_ok)
            if rank:
                summary[name]["mrr"] += 1.0 / rank
            # purity: top-5 中有多少比例落在期望文档里(衡量噪声)
            summary[name]["purity"] += (
                sum(1 for d in docs if d == expect) / len(docs) if docs else 0.0
            )

            rows.append((qid, name, top1_ok, rank, docs[:3]))

            flag = "top1 OK" if top1_ok else (f"top{rank}" if rank else "MISS")
            print(f"   {name:34} {flag:8} "
                  f"dense={len(d_hits):2} sparse={len(s_hits):2} "
                  f"top3={docs[:3]}")
        print()

    # ------------------------------------------------------------------
    print("=" * 92)
    print("汇总")
    print("=" * 92)
    n = len(CASES)
    print(f"{'方案':<36}{'top1命中':>9}{'top5命中':>10}{'MRR':>8}"
          f"{'top5纯度':>10}{'sparse空':>10}")
    print("-" * 92)
    for name in plans:
        s = summary[name]
        print(f"{name:<36}{s['top1']:>7}/{n}{s['hit5']:>8}/{n}"
              f"{s['mrr'] / n:>8.3f}{s['purity'] / n:>10.3f}"
              f"{s['sparse_empty']:>8}/{n}")

    print()
    print("=" * 92)
    print("B vs A 的逐条差异 (分通道翻译是否带来改变)")
    print("=" * 92)
    a = {qid: (ok, r) for qid, nm, ok, r, _ in rows if nm.startswith("A")}
    b = {qid: (ok, r) for qid, nm, ok, r, _ in rows if nm.startswith("B")}
    for qid, zh, expect in CASES:
        ao, ar = a[qid]
        bo, br = b[qid]
        if ao == bo and ar == br:
            continue
        print(f"  [{qid}] {zh}")
        print(f"        A: top1={ao} rank={ar}")
        print(f"        B: top1={bo} rank={br}")

    client.close()
    print()
    print("译文缓存:", _CACHE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
