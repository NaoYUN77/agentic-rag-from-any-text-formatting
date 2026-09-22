r"""检索深度解析测试：保证 `top_k` 契约能被完整满足。

回归背景（真实踩过，同一族"静默少给内容"）：

  调用方要 top_k=30，实际只拿到 20 条 —— 因为 `rerank_candidate_k` 默认 20，
  候选池比 top_k 小，rerank 最多只能吐出池子那么多。
  调用方要 top_k=45，实际只拿到 34 条 —— 因为 `candidate_k=20` 是**每路**深度，
  而 hybrid 是两路并集，且两路命中大量重叠。

两个都不会报错，只是结果变少。这类问题极难发现（指标上看不出来），
所以用测试把"每路深度必须够"这个不变量钉住。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from hybrid_retriever import resolve_depths  # noqa: E402


class ResolveDepthsTests(unittest.TestCase):
    def test_keeps_caller_value_when_already_deep_enough(self) -> None:
        """调用方给的深度够用时不改动它 —— 不要偷偷加深。"""
        for mode in ("hybrid", "dense", "sparse"):
            with self.subTest(mode=mode):
                self.assertEqual(resolve_depths(5, mode, 20), 20)

    def test_raises_depth_when_limit_exceeds_it(self) -> None:
        """要 45 条但每路只搜 20 -> 必须抬到 45。"""
        for mode in ("hybrid", "dense", "sparse"):
            with self.subTest(mode=mode):
                self.assertEqual(resolve_depths(45, mode, 20), 45)

    def test_hybrid_does_not_use_half_limit(self) -> None:
        """⚠️ 关键回归：**不能**按 `ceil(limit/2)` 算。

        那样只在两路完全不重叠时成立。实测 `candidate_k=23`、要 45 条时
        只拿到 34 条 —— 两个通道命中的块大量重叠，并集远小于 2×candidate_k。
        """
        depth = resolve_depths(45, "hybrid", 1)
        self.assertEqual(depth, 45)
        self.assertNotEqual(depth, (45 + 1) // 2)   # 不能是 23

    def test_single_channel_needs_full_limit(self) -> None:
        """单路模式最多只能给 candidate_k 条，所以深度必须 >= limit。"""
        self.assertEqual(resolve_depths(30, "dense", 5), 30)
        self.assertEqual(resolve_depths(30, "sparse", 5), 30)

    def test_boundary_is_inclusive(self) -> None:
        """depth == limit 就够，不该再多加。"""
        self.assertEqual(resolve_depths(20, "hybrid", 20), 20)
        self.assertEqual(resolve_depths(20, "hybrid", 19), 20)

    def test_limit_of_one(self) -> None:
        self.assertEqual(resolve_depths(1, "hybrid", 20), 20)
        self.assertEqual(resolve_depths(1, "dense", 0), 1)


if __name__ == "__main__":
    unittest.main()
