r"""Dense + sparse 双路召回和 RRF 融合。

    dense  : Qdrant redhat       cosine / Qwen embedding
    sparse : Qdrant redhat_sparse BM25 sparse vector
    fusion : Reciprocal Rank Fusion

这个模块只负责检索, 不负责 HTTP 和页面。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SparseVector

from sparse_pipeline_demo import analyze


@dataclass
class RetrievalHit:
    point_id: int | str
    payload: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    pre_rerank_rank: Optional[int] = None
    matched_terms: Optional[List[str]] = None


def _top_dense(hits: Sequence[RetrievalHit]) -> Optional[float]:
    """dense 排序里 top-1 的相似度。

    ⚠️ 是 **dense 自己的 top-1**，不是「最终排序 top-1 的 dense_score」——
    后者在 rrf / rerank 之后已经换人了。拒答阈值就是在这个量上标定的，
    两边必须用同一个定义，否则阈值对不上。
    """
    if not hits:
        return None
    score = hits[0].dense_score
    return float(score) if score is not None else None


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embeddings: Any,
        dense_collection: str,
        sparse_collection: str,
        artifact_dir: Path,
        rrf_k: int = 60,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.dense_collection = dense_collection
        self.sparse_collection = sparse_collection
        self.rrf_k = rrf_k
        self.k1 = k1
        self.b = b

        vocab_path = artifact_dir / "vocab.json"
        stats_path = artifact_dir / "bm25_stats.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"缺少 sparse 词表: {vocab_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"缺少 BM25 统计: {stats_path}")

        self.vocab: Dict[str, int] = json.loads(
            vocab_path.read_text(encoding="utf-8")
        )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.idf: Dict[str, float] = stats["idf"]
        self.avgdl = float(stats.get("avgdl", 0.0))

    def _filter(self, section: Optional[str]) -> Optional[Filter]:
        if not section:
            return None
        from qdrant_client.models import FieldCondition, MatchText

        return Filter(must=[FieldCondition(
            key="section",
            match=MatchText(text=section),
        )])

    def dense_search(
        self,
        query: str,
        limit: int,
        section: Optional[str] = None,
    ) -> List[RetrievalHit]:
        query_vector = self.embeddings.embed_query(query)
        result = self.client.query_points(
            collection_name=self.dense_collection,
            query=query_vector,
            limit=limit,
            query_filter=self._filter(section),
            with_payload=True,
        )
        return [
            RetrievalHit(
                point_id=point.id,
                payload=dict(point.payload or {}),
                dense_score=float(point.score),
                dense_rank=rank,
            )
            for rank, point in enumerate(result.points, 1)
        ]

    def sparse_query_vector(
        self,
        query: str,
    ) -> Tuple[Optional[SparseVector], List[str], List[str]]:
        counts = Counter(analyze(query))
        indices: List[int] = []
        values: List[float] = []
        known: List[str] = []
        unknown: List[str] = []

        for term, qtf in counts.items():
            term_id = self.vocab.get(term)
            if term_id is None:
                unknown.append(term)
                continue
            known.append(term)
            indices.append(int(term_id))
            values.append(float(self.idf[term]) * qtf)

        if not indices:
            return None, sorted(known), sorted(unknown)
        return SparseVector(indices=indices, values=values), sorted(known), sorted(unknown)

    def sparse_search(
        self,
        query: str,
        limit: int,
        section: Optional[str] = None,
    ) -> Tuple[List[RetrievalHit], List[str], List[str]]:
        query_vector, known, unknown = self.sparse_query_vector(query)
        if query_vector is None:
            return [], known, unknown

        result = self.client.query_points(
            collection_name=self.sparse_collection,
            query=query_vector,
            using="sparse",
            limit=limit,
            query_filter=self._filter(section),
            with_payload=True,
        )
        hits = [
            RetrievalHit(
                point_id=point.id,
                payload=dict(point.payload or {}),
                sparse_score=float(point.score),
                sparse_rank=rank,
                matched_terms=known,
            )
            for rank, point in enumerate(result.points, 1)
        ]
        return hits, known, unknown

    @staticmethod
    def rrf_fuse(
        rank_lists: Sequence[Sequence[RetrievalHit]],
        k: int = 60,
        limit: int = 5,
    ) -> List[RetrievalHit]:
        merged: Dict[int | str, RetrievalHit] = {}

        for rank_list in rank_lists:
            for rank, hit in enumerate(rank_list, 1):
                current = merged.get(hit.point_id)
                if current is None:
                    current = RetrievalHit(
                        point_id=hit.point_id,
                        payload=dict(hit.payload),
                        matched_terms=[],
                    )
                    merged[hit.point_id] = current

                current.payload.update(hit.payload)
                if hit.dense_score is not None:
                    current.dense_score = hit.dense_score
                if hit.sparse_score is not None:
                    current.sparse_score = hit.sparse_score
                if hit.dense_rank is not None:
                    current.dense_rank = hit.dense_rank
                if hit.sparse_rank is not None:
                    current.sparse_rank = hit.sparse_rank
                current.matched_terms = sorted(set(
                    (current.matched_terms or []) + (hit.matched_terms or [])
                ))
                current.rrf_score = (current.rrf_score or 0.0) + 1.0 / (k + rank)

        result = sorted(
            merged.values(),
            key=lambda hit: (
                -(hit.rrf_score or 0.0),
                hit.dense_rank or 10**9,
                hit.sparse_rank or 10**9,
                str(hit.point_id),
            ),
        )
        return result[:limit]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        candidate_k: int = 20,
        rrf_k: Optional[int] = None,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = mode.lower()
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"未知检索模式: {mode}")

        dense_hits: List[RetrievalHit] = []
        sparse_hits: List[RetrievalHit] = []
        known: List[str] = []
        unknown: List[str] = []

        if mode in {"hybrid", "dense"}:
            dense_hits = self.dense_search(query, candidate_k, section=section)

        if mode in {"hybrid", "sparse"}:
            sparse_hits, known, unknown = self.sparse_search(
                query, candidate_k, section=section
            )

        if mode == "dense":
            return {
                "mode": mode,
                "hits": dense_hits[:limit],
                "known_terms": [],
                "unknown_terms": [],
                "dense_top_score": _top_dense(dense_hits),
            }

        if mode == "sparse":
            return {
                "mode": mode,
                "hits": sparse_hits[:limit],
                "known_terms": known,
                "unknown_terms": unknown,
                "dense_top_score": None,
            }

        fused = self.rrf_fuse(
            [dense_hits, sparse_hits],
            k=rrf_k or self.rrf_k,
            limit=limit,
        )
        return {
            "mode": mode,
            "hits": fused,
            "known_terms": known,
            "unknown_terms": unknown,
            "dense_top_score": _top_dense(dense_hits),
        }
