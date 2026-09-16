r"""Reranker: 对召回候选做 query-document 逐对精排。"""

from __future__ import annotations

import os
from typing import List, Sequence

import requests

from hybrid_retriever import RetrievalHit


DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)


class DashScopeReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "gte-rerank-v2",
        api_url: str = DEFAULT_RERANK_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "DashScopeReranker":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY, 无法初始化 reranker")
        return cls(
            api_key=api_key,
            model=os.getenv("QWEN_RERANK_MODEL", "gte-rerank-v2"),
            api_url=os.getenv("QWEN_RERANK_API_URL", DEFAULT_RERANK_URL),
            timeout=float(os.getenv("QWEN_RERANK_TIMEOUT", "60")),
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_k: int,
        max_doc_chars: int = 4000,
    ) -> List[RetrievalHit]:
        if not hits:
            return []

        documents: List[str] = []
        for hit in hits:
            text = str((hit.payload or {}).get("text", "")).strip()
            documents.append(text[:max_doc_chars])

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"query": query, "documents": documents},
                "parameters": {
                    "return_documents": False,
                    "top_n": min(top_k, len(documents)),
                },
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Rerank 接口返回 %s: %s"
                % (response.status_code, response.text[:500])
            )

        results = response.json()["output"]["results"]
        reranked: List[RetrievalHit] = []
        for result in results:
            source = hits[int(result["index"])]
            reranked.append(RetrievalHit(
                point_id=source.point_id,
                payload=dict(source.payload),
                dense_score=source.dense_score,
                sparse_score=source.sparse_score,
                dense_rank=source.dense_rank,
                sparse_rank=source.sparse_rank,
                rrf_score=source.rrf_score,
                rerank_score=float(result["relevance_score"]),
                pre_rerank_rank=int(result["index"]) + 1,
                matched_terms=list(source.matched_terms or []),
            ))
        return reranked
