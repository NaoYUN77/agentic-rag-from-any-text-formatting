r"""Rerank smoke test: 比较 RRF 和 RRF + rerank 的检索指标。

这是一个小样本、人工标注的工程 smoke test, 不是完整评估集。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


API_URL = "http://127.0.0.1:8000/api/search"
QUERY_FILE = Path(__file__).parent / "eval" / "rerank_smoke.jsonl"


def load_queries(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def search(query: str, rerank: bool, top_k: int = 20) -> Dict[str, Any]:
    response = requests.post(
        API_URL,
        json={
            "query": query,
            "mode": "hybrid",
            "top_k": top_k,
            "candidate_k": 20,
            "rrf_k": 60,
            "rerank": rerank,
            "rerank_candidate_k": 20,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def first_relevant_rank(ids: Sequence[int], gold: Sequence[int]) -> int | None:
    gold_set = set(gold)
    for rank, point_id in enumerate(ids, 1):
        if point_id in gold_set:
            return rank
    return None


def recall_at(ids: Sequence[int], gold: Sequence[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ids[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def main() -> None:
    parser = argparse.ArgumentParser(description="RRF vs Rerank smoke 对比")
    parser.add_argument("--qrels", required=True)
    args = parser.parse_args()

    queries = load_queries(Path(args.qrels))
    totals = {
        "pre_recall5": 0.0,
        "post_recall5": 0.0,
        "pre_recall10": 0.0,
        "post_recall10": 0.0,
        "pre_mrr20": 0.0,
        "post_mrr20": 0.0,
    }

    print("=" * 88)
    print("Rerank smoke test: RRF vs RRF + rerank")
    print("=" * 88)

    for row in queries:
        query = row["query"]
        gold = row["gold"]
        pre = search(query, rerank=False)
        post = search(query, rerank=True)
        pre_ids = [h["point_id"] for h in pre["hits"]]
        post_ids = [h["point_id"] for h in post["hits"]]

        pre_rank = first_relevant_rank(pre_ids, gold)
        post_rank = first_relevant_rank(post_ids, gold)
        pre_r5 = recall_at(pre_ids, gold, 5)
        post_r5 = recall_at(post_ids, gold, 5)
        pre_r10 = recall_at(pre_ids, gold, 10)
        post_r10 = recall_at(post_ids, gold, 10)

        totals["pre_recall5"] += pre_r5
        totals["post_recall5"] += post_r5
        totals["pre_recall10"] += pre_r10
        totals["post_recall10"] += post_r10
        totals["pre_mrr20"] += reciprocal_rank(pre_rank)
        totals["post_mrr20"] += reciprocal_rank(post_rank)

        print(f"\n[{row['query_id']}] {query}")
        print(f"gold       : {gold}")
        print(f"RRF ids    : {pre_ids}")
        print(f"rerank ids : {post_ids}")
        print(f"first gold : RRF={pre_rank}  rerank={post_rank}")
        print(f"Recall@5   : {pre_r5:.3f} -> {post_r5:.3f}")
        print(f"Recall@10  : {pre_r10:.3f} -> {post_r10:.3f}")
        print(
            "rerank     : applied=%s model=%s elapsed=%sms rerank=%sms"
            % (
                post.get("rerank_applied"),
                post.get("rerank_model"),
                post.get("elapsed_ms"),
                post.get("rerank_ms"),
            )
        )

    n = max(len(queries), 1)
    print("\n" + "=" * 88)
    print("Aggregate")
    print("=" * 88)
    print(f"Recall@5   : {totals['pre_recall5'] / n:.3f} -> {totals['post_recall5'] / n:.3f}")
    print(f"Recall@10  : {totals['pre_recall10'] / n:.3f} -> {totals['post_recall10'] / n:.3f}")
    print(f"MRR@20     : {totals['pre_mrr20'] / n:.3f} -> {totals['post_mrr20'] / n:.3f}")


if __name__ == "__main__":
    main()
