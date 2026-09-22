r"""Reranker: 对召回候选做 query-document 逐对精排。"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Sequence

import requests

from hybrid_retriever import RetrievalHit


DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"


class DashScopeReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.7-text-rerank",
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
            model=os.getenv("QWEN_RERANK_MODEL", "qwen3.7-text-rerank"),
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


class VoyageReranker:
    """VoyageAI rerank 客户端（`https://api.voyageai.com/v1/rerank`）。

    ⚠️ 与 `DashScopeReranker` 的**协议不同，不能只换 URL**：

    | | DashScope | VoyageAI |
    |---|---|---|
    | 请求体 | `input.query` + `parameters.top_n` | **扁平**：`query` / `documents` / `top_k` |
    | 结果位置 | `output.results[]` | `data[]` |
    | 超长输入 | 无开关 | `truncation`（默认 true，自动截断） |

    官方限制（rerank-2.5 / 2.5-lite）：
    - 单次文档数 ≤ **1000**
    - query + **任一单篇**文档 ≤ **32,000** token
    - 总量 `query_token × 文档数 + 所有文档 token 和` ≤ **600K** token
    - query 本身 ≤ 8,000 token

    ⚠️ **限速（2026-09-22 实测）**：
    - **未绑支付方式**：`3 RPM / 10K TPM` —— 一次评估要打 39 次，几乎全被 429
    - **绑卡后（Tier 1）**：`2000 RPM / 4M TPM`（差 667 倍）
    - 官方明说「**Even with a payment method entered, the free tokens will still apply**」
      —— 绑卡不等于扣钱，免费额度照用。
    - 所以本类默认带 429 退避重试；但 3 RPM 下重试也只是把失败变成很慢。

    实测（2026-09-22，rerank-2.5-lite）：
    3 篇短文档 + 1 个中文 query = 82 token，返回按分数降序的 `data[]`。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-2.5-lite",
        api_url: str = VOYAGE_RERANK_URL,
        timeout: float = 60.0,
        truncation: bool = True,
        max_retries: int = 7,
        retry_base: float = 2.0,
        min_interval: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.truncation = truncation
        self.max_retries = max(0, max_retries)
        self.retry_base = retry_base
        # 两次调用之间的最小间隔（秒）。0 = 不限速。
        #
        # 为什么需要「主动」限速而不是只靠 429 重试：限速是按分钟计的，
        # 撞上 429 再退避等于白打一次请求、还要等一整轮窗口。
        # 已知档位时直接设成对应间隔更省 —— 例如未绑卡档位
        # 3 RPM / 10K TPM、每次 ~6,900 token，瓶颈是 TPM：
        # 10_000 / 6_900 ≈ 1.45 次/分钟 → 间隔约 41s。
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0

    @classmethod
    def from_env(cls) -> "VoyageReranker":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 VOYAGE_API_KEY, 无法初始化 Voyage reranker")
        return cls(
            api_key=api_key,
            model=os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5-lite"),
            api_url=os.getenv("VOYAGE_RERANK_API_URL", VOYAGE_RERANK_URL),
            timeout=float(os.getenv("VOYAGE_RERANK_TIMEOUT", "60")),
            max_retries=int(os.getenv("VOYAGE_RERANK_RETRIES", "7")),
            min_interval=float(os.getenv("VOYAGE_RERANK_MIN_INTERVAL", "0")),
        )

    @staticmethod
    def _retry_after(response: Any) -> Optional[float]:
        """优先用服务端给的 Retry-After，没有再用退避。"""
        raw = None
        headers = getattr(response, "headers", None) or {}
        for key in ("Retry-After", "retry-after"):
            try:
                raw = headers.get(key)
            except AttributeError:
                raw = None
            if raw:
                break
        try:
            return float(raw) if raw else None
        except (TypeError, ValueError):
            return None

    def _pace(self) -> None:
        """主动限速：距上次调用不足 min_interval 就先等一等。

        限速是按分钟计的，撞上 429 再退避等于白打一次请求、还要等一整轮窗口。
        已知档位时直接按间隔发更省。
        """
        if self.min_interval <= 0:
            return
        gap = time.monotonic() - self._last_call
        if self._last_call and gap < self.min_interval:
            time.sleep(self.min_interval - gap)

    def _post(self, payload: dict):
        """POST + 主动限速 + 429 指数退避重试（官方推荐做法）。

        ⚠️ 退避要够长才有效：未绑卡档位的瓶颈是 **TPM**（10K/分钟），
        每次 ~6,900 token 意味着两次调用至少要隔 ~41s。
        退避上限 30s（2+4+8+16）时重试仍会失败 —— 实测 25/39 条 429。
        现在默认 7 次、封顶 60s，累计可等 ~182s。
        """
        delay = self.retry_base
        response = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            self._last_call = time.monotonic()
            if response.status_code != 429 or attempt == self.max_retries:
                return response
            wait = self._retry_after(response) or delay
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
        return response

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

        response = self._post({
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_k": min(top_k, len(documents)),
            "truncation": self.truncation,
        })
        if response.status_code != 200:
            raise RuntimeError(
                "Voyage rerank 返回 %s: %s"
                % (response.status_code, response.text[:500])
            )

        # 响应已按 relevance_score 降序；仍显式排序，避免上游改了行为就静默错位
        rows = sorted(
            response.json()["data"],
            key=lambda x: -float(x["relevance_score"]),
        )
        reranked: List[RetrievalHit] = []
        for result in rows:
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


def build_reranker(name: str = "voyage"):
    """按名字构造 reranker。

    voyage   -> VoyageReranker（rerank-2.5-lite，默认）
    dashscope-> DashScopeReranker（qwen3.7-text-rerank）
    none     -> None（不精排）
    """
    key = (name or "").strip().lower()
    if key in ("none", "off", ""):
        return None
    if key == "voyage":
        return VoyageReranker.from_env()
    if key in ("dashscope", "qwen"):
        return DashScopeReranker.from_env()
    raise ValueError("未知的 reranker: %s（可选 voyage / dashscope / none）" % name)
