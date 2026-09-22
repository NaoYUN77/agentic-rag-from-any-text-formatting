r"""Reranker 客户端测试。

重点：VoyageAI 与 DashScope 的**协议不同**（请求体扁平 vs 嵌套、结果在
`data[]` vs `output.results[]`）。只换 URL 会静默错位，所以这里把两家的
请求体形状和响应解析都钉住。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from hybrid_retriever import RetrievalHit  # noqa: E402
from reranker import (  # noqa: E402
    DashScopeReranker, VoyageReranker, build_reranker,
)


def hits(*texts: str):
    return [
        RetrievalHit(point_id=i, payload={"text": t}, dense_score=0.5)
        for i, t in enumerate(texts)
    ]


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class VoyageRerankerTests(unittest.TestCase):
    def test_request_body_is_flat_not_nested(self) -> None:
        """Voyage 的 query/documents 在**顶层**；DashScope 才嵌在 input/parameters 里。

        只换 URL 会得到一个字段全错的请求体，接口报错或静默返回空。
        """
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return _Resp({"data": [{"index": 0, "relevance_score": 0.9}]})

        with patch("reranker.requests.post", fake_post):
            VoyageReranker(api_key="k", model="rerank-2.5-lite").rerank(
                "q", hits("a", "b"), top_k=1)

        self.assertIn("api.voyageai.com", captured["url"])
        self.assertIn("/v1/rerank", captured["url"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer k")
        body = captured["body"]
        self.assertEqual(body["query"], "q")               # 顶层
        self.assertEqual(body["documents"], ["a", "b"])     # 顶层
        self.assertEqual(body["top_k"], 1)
        self.assertIn("truncation", body)
        self.assertNotIn("input", body)                     # 不能是 DashScope 形状
        self.assertNotIn("parameters", body)

    def test_parses_data_array_and_sorts_by_score(self) -> None:
        """结果在 `data[]`（不是 `output.results[]`），且要按分数降序。"""
        def fake_post(url, headers=None, json=None, timeout=None):
            # 故意乱序返回，验证我们显式排序而不是依赖上游顺序
            return _Resp({"data": [
                {"index": 2, "relevance_score": 0.10},
                {"index": 0, "relevance_score": 0.90},
                {"index": 1, "relevance_score": 0.50},
            ]})

        with patch("reranker.requests.post", fake_post):
            out = VoyageReranker(api_key="k").rerank("q", hits("a", "b", "c"), top_k=3)

        self.assertEqual([h.payload["text"] for h in out], ["a", "b", "c"])
        self.assertEqual([h.rerank_score for h in out], [0.90, 0.50, 0.10])
        # pre_rerank_rank 记录的是"进 rerank 前"的位置（1-based）
        self.assertEqual([h.pre_rerank_rank for h in out], [1, 2, 3])

    def test_empty_hits_short_circuits(self) -> None:
        with patch("reranker.requests.post") as post:
            self.assertEqual(VoyageReranker(api_key="k").rerank("q", [], top_k=5), [])
        post.assert_not_called()

    def test_top_k_is_capped_by_document_count(self) -> None:
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json
            return _Resp({"data": [{"index": 0, "relevance_score": 0.9}]})

        with patch("reranker.requests.post", fake_post):
            VoyageReranker(api_key="k").rerank("q", hits("a", "b"), top_k=99)
        self.assertEqual(captured["body"]["top_k"], 2)

    def test_non_200_raises_with_body(self) -> None:
        def fake_post(url, headers=None, json=None, timeout=None):
            return _Resp({"detail": "invalid api key"}, status=401)

        with patch("reranker.requests.post", fake_post):
            with self.assertRaises(RuntimeError) as ctx:
                VoyageReranker(api_key="bad").rerank("q", hits("a"), top_k=1)
        self.assertIn("401", str(ctx.exception))

    def test_429_is_retried_then_succeeds(self) -> None:
        """429 要退避重试 —— 未绑卡档位只有 3 RPM，一次评估几乎必然撞上。"""
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return _Resp({"detail": "rate limited"}, status=429)
            return _Resp({"data": [{"index": 0, "relevance_score": 0.7}]})

        with patch("reranker.requests.post", fake_post), \
                patch("reranker.time.sleep") as sleeper:
            out = VoyageReranker(api_key="k", retry_base=1.0).rerank(
                "q", hits("a"), top_k=1)

        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(out), 1)
        # 退避应递增：1s 然后 2s
        self.assertEqual([c.args[0] for c in sleeper.call_args_list], [1.0, 2.0])

    def test_429_gives_up_after_max_retries(self) -> None:
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            return _Resp({"detail": "rate limited"}, status=429)

        with patch("reranker.requests.post", fake_post), \
                patch("reranker.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                VoyageReranker(api_key="k", max_retries=2).rerank(
                    "q", hits("a"), top_k=1)
        self.assertIn("429", str(ctx.exception))
        self.assertEqual(calls["n"], 3)      # 首次 + 2 次重试

    def test_retry_after_header_wins_over_backoff(self) -> None:
        class _H(_Resp):
            headers = {"Retry-After": "7"}

        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _H({"detail": "slow down"}, status=429)
            return _Resp({"data": [{"index": 0, "relevance_score": 0.5}]})

        with patch("reranker.requests.post", fake_post), \
                patch("reranker.time.sleep") as sleeper:
            VoyageReranker(api_key="k", retry_base=1.0).rerank(
                "q", hits("a"), top_k=1)
        self.assertEqual(sleeper.call_args_list[0].args[0], 7.0)


class DashScopeRerankerTests(unittest.TestCase):
    def test_request_body_is_nested(self) -> None:
        """反向钉子：DashScope 用 input/parameters，别被 Voyage 的改动带偏。"""
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json
            return _Resp({"output": {"results": [
                {"index": 0, "relevance_score": 0.8}]}})

        with patch("reranker.requests.post", fake_post):
            DashScopeReranker(api_key="k").rerank("q", hits("a"), top_k=1)

        body = captured["body"]
        self.assertIn("input", body)
        self.assertEqual(body["input"]["query"], "q")
        self.assertIn("parameters", body)
        self.assertNotIn("query", body)          # 不在顶层


class BuildRerankerTests(unittest.TestCase):
    def test_none_disables(self) -> None:
        self.assertIsNone(build_reranker("none"))
        self.assertIsNone(build_reranker("off"))
        self.assertIsNone(build_reranker(""))

    def test_voyage_from_env(self) -> None:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "vk"}, clear=False):
            r = build_reranker("voyage")
        self.assertIsInstance(r, VoyageReranker)
        self.assertEqual(r.model, "rerank-2.5-lite")

    def test_dashscope_from_env(self) -> None:
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "dk"}, clear=False):
            r = build_reranker("dashscope")
        self.assertIsInstance(r, DashScopeReranker)

    def test_missing_key_raises_actionable_error(self) -> None:
        env = {k: v for k, v in __import__("os").environ.items()
               if k != "VOYAGE_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                build_reranker("voyage")
        self.assertIn("VOYAGE_API_KEY", str(ctx.exception))

    def test_unknown_name_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_reranker("cohere")
        self.assertIn("voyage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
