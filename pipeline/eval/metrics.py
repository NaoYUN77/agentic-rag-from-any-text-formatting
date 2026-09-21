r"""检索指标。

口径约定
--------
gold 是 **chunk_id 的集合**而不是单个 ID —— 因为 overlap 会让同一段原文
出现在多个 chunk 里(实测一个 block 最多进 3 个 chunk)。若 gold 只记一个,
检索命中它的"邻居 chunk"(含同一段文字)会被判为未命中, 指标被系统性低估。

因此每个 query 同时报两个口径:

    hit@k      至少命中一个 gold        —— 宽松, 看"有没有找到"
    recall@k   命中数 / gold 总数       —— 严格, 看"找全没有"

`unanswerable` 类别的 query 没有 gold, 三个指标都返回 None 而不是 0 ——
它们不参与 recall/MRR 的均值(分母不该包含它们)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set


def hit_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> Optional[float]:
    """top-k 里是否至少有一个 gold。"""
    if not gold:
        return None
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def recall_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> Optional[float]:
    """命中数 / gold 总数。"""
    if not gold:
        return None
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> Optional[float]:
    """命中数 / k。注意 k > len(retrieved) 时按实际返回条数算, 避免虚低。"""
    if not gold:
        return None
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def mrr_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> Optional[float]:
    """第一个 gold 的倒数排名; 没命中记 0。"""
    if not gold:
        return None
    for rank, cid in enumerate(retrieved[:k], 1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Set[str], k: int) -> Optional[float]:
    """二值相关性的 nDCG@k。"""
    if not gold:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], 1)
        if cid in gold
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), k) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


# 默认上报的 k 值
DEFAULT_KS = (5, 10, 20)


def score_query(
    retrieved: Sequence[str],
    gold: Set[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Optional[float]]:
    """算一条 query 的全部指标。"""
    out: Dict[str, Optional[float]] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        out[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    # 精确率只在最小的 k 上报, 避免表太长
    k0 = ks[0] if ks else 5
    out[f"precision@{k0}"] = precision_at_k(retrieved, gold, k0)
    return out


def aggregate(
    per_query: Sequence[Dict[str, Optional[float]]],
) -> Dict[str, float]:
    """对多条 query 求均值。

    None(来自 unanswerable)不参与均值 —— 它们没有 gold, 算进来会污染分母。
    若某个指标全是 None, 返回 0.0 并在报告里标注样本数为 0。
    """
    keys: Set[str] = set()
    for row in per_query:
        keys |= set(row.keys())

    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals = [r.get(key) for r in per_query]
        nums = [v for v in vals if v is not None]
        out[key] = round(sum(nums) / len(nums), 4) if nums else 0.0
        out[f"{key}__n"] = float(len(nums))
    return out


def ranking_stage_names() -> List[str]:
    """分阶段评估的五个阶段名, 与 run_eval 对齐。"""
    return ["dense", "sparse", "union", "rrf", "rerank"]
