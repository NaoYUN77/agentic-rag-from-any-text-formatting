r"""拒答（abstention）测试。

判据是 **query 的 dense 排序 top-1 余弦相似度**低于阈值 —— 与
`eval/run_eval.py` 的「拒答能力」一节用的是**同一个量**。
两边定义必须一致，否则阈值对不上（这是最容易搞错的地方：
最终排序 top-1 的 `dense_score` 在 rrf / rerank 之后已经换人了）。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from hybrid_retriever import RetrievalHit, _top_dense  # noqa: E402
from rag_server import _load_abstain_threshold, _should_abstain  # noqa: E402


class LoadThresholdTests(unittest.TestCase):
    """阈值来源：环境变量 > 标定文件 > 0（关闭）。

    支持文件是为了让「标定 -> 生效」一步到位。只靠环境变量的话，
    重算完还得手工搬数字，很容易忘 —— 然后线上带着**过期阈值**跑，
    症状是「明明有答案却被拒答」，极难归因。
    """

    def test_env_wins_over_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / ".abstain_threshold"
            f.write_text("0.5285\n", encoding="utf-8")
            self.assertEqual(_load_abstain_threshold("0.9", f), 0.9)

    def test_falls_back_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / ".abstain_threshold"
            f.write_text("0.5285\n", encoding="utf-8")
            self.assertEqual(_load_abstain_threshold("", f), 0.5285)

    def test_no_source_means_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / ".abstain_threshold"
            self.assertEqual(_load_abstain_threshold("", missing), 0.0)

    def test_invalid_env_does_not_crash(self) -> None:
        """环境变量填错不该让服务起不来 —— 退化成关闭，并打一行警告。"""
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / ".abstain_threshold"
            self.assertEqual(_load_abstain_threshold("abc", missing), 0.0)

    def test_garbage_file_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / ".abstain_threshold"
            f.write_text("not-a-number\n", encoding="utf-8")
            self.assertEqual(_load_abstain_threshold("", f), 0.0)


class TopDenseTests(unittest.TestCase):
    def test_returns_first_hits_dense_score(self) -> None:
        """取的是 dense 列表里**第一个**的分数，不是最大值也不是最终排序的。"""
        hits = [
            RetrievalHit(point_id=1, payload={}, dense_score=0.81),
            RetrievalHit(point_id=2, payload={}, dense_score=0.95),
        ]
        self.assertEqual(_top_dense(hits), 0.81)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_top_dense([]))

    def test_missing_score_returns_none(self) -> None:
        hits = [RetrievalHit(point_id=1, payload={}, dense_score=None)]
        self.assertIsNone(_top_dense(hits))


class ShouldAbstainTests(unittest.TestCase):
    def test_threshold_zero_disables_abstention(self) -> None:
        """默认 0 = 关闭，必须完全保持原有行为（不拒答）。"""
        with patch("rag_server.ABSTAIN_THRESHOLD", 0.0):
            self.assertFalse(_should_abstain(0.0))
            self.assertFalse(_should_abstain(0.1))
            self.assertFalse(_should_abstain(0.9))

    def test_below_threshold_abstains(self) -> None:
        with patch("rag_server.ABSTAIN_THRESHOLD", 0.5285):
            self.assertTrue(_should_abstain(0.382))   # 实测 un_01
            self.assertTrue(_should_abstain(0.450))   # 实测 un_02

    def test_above_threshold_answers(self) -> None:
        with patch("rag_server.ABSTAIN_THRESHOLD", 0.5285):
            self.assertFalse(_should_abstain(0.811))  # 实测 cr_01
            self.assertFalse(_should_abstain(0.593))  # 实测 un_03：没被抓住的那条

    def test_exactly_at_threshold_answers(self) -> None:
        """用 `<` 而不是 `<=` —— 边界上宁可作答，不误拒。"""
        with patch("rag_server.ABSTAIN_THRESHOLD", 0.5):
            self.assertFalse(_should_abstain(0.5))

    def test_missing_score_answers_not_abstains(self) -> None:
        """判据缺失时（如 mode=sparse 没跑 dense）宁可作答。

        **没依据就拒答**比「答错」更糟 —— 用户会以为语料里真没有。
        """
        with patch("rag_server.ABSTAIN_THRESHOLD", 0.5285):
            self.assertFalse(_should_abstain(None))


if __name__ == "__main__":
    unittest.main()
