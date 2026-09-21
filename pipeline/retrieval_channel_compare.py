r"""检索通道对照实验: 纯向量 vs BM25。

目的:
    找出纯向量检索在哪些查询上失效, 并验证 BM25 能否补上。

设计:
    查询分三类
        语义型   换个说法描述同一件事          -> 向量应该赢
        精确型   含配置项名 / 文件路径 / 错误码 -> BM25 应该赢
        混合型   既有语义又有专有名词

用法:
    cd pipeline
    $env:DASHSCOPE_API_KEY = "sk-..."
    .\.venv_rag\Scripts\python.exe retrieval_channel_compare.py
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import jieba
import numpy as np
from qdrant_client import QdrantClient

from embeddings import build_embeddings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QDRANT_PATH = "qdrant_data"
COLLECTION = "redhat"

# 判定"命中"用的关键词: 检索结果里必须出现这个串才算对
QUERIES: List[Tuple[str, str, str]] = [
    # (类型, 查询, 命中判据关键字)
    ("语义", "怎么让服务在主节点挂掉后自动切换", "故障转移"),
    ("语义", "如何避免请求都打到同一台服务器", "负载平衡"),
    ("精确", "virtual_ipaddress 在哪里配置", "virtual_ipaddress"),
    ("精确", "keepalived.conf 里有哪些配置项", "keepalived.conf"),
    ("精确", "invalid ip number count 是什么错误", "invalid ip number count"),
    ("精确", "VRRP_Instance 是什么意思", "VRRP_Instance"),
    ("精确", "arptables 和 iptables 有什么区别", "arptables"),
    ("混合", "keepalived 的 VRRP 版本不匹配会怎样", "VRRP"),
]


def tokenize(text: str) -> List[str]:
    """中文分词 + 小写化, 去掉单字符和标点。"""
    toks = []
    for t in jieba.cut(text):
        t = t.strip().lower()
        if len(t) >= 2 or t.isalnum() and len(t) >= 2:
            toks.append(t)
    return toks


class BM25:
    """标准 BM25 实现(k1=1.2, b=0.75)。"""

    def __init__(self, docs: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs_tokens) / max(self.N, 1)
        self.df: Dict[str, int] = Counter()
        for toks in self.docs_tokens:
            for w in set(toks):
                self.df[w] += 1
        self.tf = [Counter(toks) for toks in self.docs_tokens]

    def idf(self, w: str) -> float:
        df = self.df.get(w, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N, dtype=np.float64)
        for w in q:
            if w not in self.df:
                continue
            idf = self.idf(w)
            for i in range(self.N):
                f = self.tf[i].get(w, 0)
                if not f:
                    continue
                dl = len(self.docs_tokens[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def cos_scores(qv: np.ndarray, mat: np.ndarray) -> np.ndarray:
    qn = qv / (np.linalg.norm(qv) + 1e-12)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mn @ qn


def rrf(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """倒数排名融合。"""
    out: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, 1):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + r)
    return out


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)
    pts, _ = client.scroll(COLLECTION, limit=200, with_payload=True, with_vectors=True)
    pts = sorted(pts, key=lambda p: p.payload.get("chunk_index", 0))
    client.close()

    docs = [p.payload["text"] for p in pts]
    vecs = np.array([p.vector for p in pts], dtype=np.float64)
    sections = [p.payload.get("section", "") for p in pts]

    print("索引: %d 个 chunk" % len(docs))
    print("分词器: jieba    BM25(k1=1.2, b=0.75)")
    print()

    emb = build_embeddings("qwen")
    bm = BM25(docs)

    TOPK = 3
    stats = {"语义": [0, 0, 0], "精确": [0, 0, 0], "混合": [0, 0, 0]}  # [向量, BM25, 融合]

    for kind, query, key in QUERIES:
        qv = np.array(emb.embed_query(query), dtype=np.float64)
        vs = cos_scores(qv, vecs)
        bs = bm.score(query)

        v_rank = list(np.argsort(-vs))
        b_rank = list(np.argsort(-bs))
        fused = rrf([v_rank[:20], b_rank[:20]])
        f_rank = sorted(fused.keys(), key=lambda i: -fused[i])

        def hit(rank):
            return int(any(key.lower() in docs[i].lower() for i in rank[:TOPK]))

        h_v, h_b, h_f = hit(v_rank), hit(b_rank), hit(f_rank)
        stats[kind][0] += h_v
        stats[kind][1] += h_b
        stats[kind][2] += h_f

        print("=" * 78)
        print("[%s] %s" % (kind, query))
        print("      命中判据: 结果里出现过 %r" % key)
        print("-" * 78)
        for name, rank, sc in [("向量", v_rank, vs), ("BM25", b_rank, bs)]:
            mark = "命中" if hit(rank) else "  未中"
            head = "%-5s %s   " % (name, mark)
            print("%s #1 %.4f  %s" % (head, sc[rank[0]], sections[rank[0]][:40]))
            for i in rank[1:TOPK]:
                print("      #%d %.4f  %s" % (list(rank[:TOPK]).index(i)+1, sc[i], sections[i][:40]))
        mark = "命中" if h_f else "  未中"
        print("融合   %s   #1 %s" % (mark, sections[f_rank[0]][:40]))
        print()

    print("=" * 78)
    print("统计(命中 %d 条内记为命中)" % TOPK)
    print("=" * 78)
    print("  %-8s %8s %8s %8s" % ("类型", "纯向量", "BM25", "RRF融合"))
    for k, (a, b_, c) in stats.items():
        n = sum(1 for kk, _, _ in QUERIES if kk == k)
        print("  %-8s %6d/%d %6d/%d %6d/%d" % (k, a, n, b_, n, c, n))
    print()
    print("  注: 语义型该赢在向量, 精确型该赢在 BM25")


if __name__ == "__main__":
    main()

