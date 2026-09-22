r"""拒答（abstention）测试。

判据是 **query 的 dense 排序 top-1 余弦相似度**低于阈值 —— 与
`eval/run_eval.py` 的「拒答能力」一节用的是**同一个量**。
两边定义必须一致，否则阈值对不上（这是最容易搞错的地方：
最终排序 top-1 的 `dense_score` 在 rrf / rerank 之后已经换人了）。

⚠️ 本测试**不 import `rag_server`** —— 那会拖上 fastapi / uvicorn 等一整套
Web 依赖，而 CI 只装 Phase 0 的 8 个包（撞过：整个测试套件因缺 fastapi 失败）。
拒答逻辑是纯函数，抽在 `abstention.py` 里就是为了这个。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from abstention import load_threshold, should_abstain  # noqa: E402
from hybrid_retriever import RetrievalHit, _top_dense  # noqa: E402


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
        for score in (0.0, 0.1, 0.9):
            with self.subTest(score=score):
                self.assertFalse(should_abstain(score, 0.0))

    def test_below_threshold_abstains(self) -> None:
        self.assertTrue(should_abstain(0.382, 0.5285))   # 实测 un_01
        self.assertTrue(should_abstain(0.450, 0.5285))   # 实测 un_02

    def test_above_threshold_answers(self) -> None:
        self.assertFalse(should_abstain(0.811, 0.5285))  # 实测 cr_01
        self.assertFalse(should_abstain(0.593, 0.5285))  # 实测 un_03：没被抓住的那条

    def test_exactly_at_threshold_answers(self) -> None:
        """用 `<` 而不是 `<=` —— 边界上宁可作答，不误拒。"""
        self.assertFalse(should_abstain(0.5, 0.5))

    def test_missing_score_answers_not_abstains(self) -> None:
        """判据缺失时（如 mode=sparse 没跑 dense）宁可作答。

        **没依据就拒答**比「答错」更糟 —— 用户会以为语料里真没有。
        """
        self.assertFalse(should_abstain(None, 0.5285))


class LoadThresholdTests(unittest.TestCase):
    """阈值来源：环境变量 > 标定文件 > 0（关闭）。

    支持文件是为了让「标定 -> 生效」一步到位。只靠环境变量的话，
    重算完还得手工搬数字，很容易忘 —— 然后线上带着**过期阈值**跑，
    症状是「明明有答案却被拒答」，极难归因。
    """

    def _file(self, d: str, content: str) -> Path:
        f = Path(d) / ".abstain_threshold"
        f.write_text(content, encoding="utf-8")
        return f

    def test_env_wins_over_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_threshold("0.9", self._file(d, "0.5285\n")), 0.9)

    def test_falls_back_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_threshold("", self._file(d, "0.5285\n")), 0.5285)

    def test_no_source_means_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / ".abstain_threshold"
            self.assertEqual(load_threshold("", missing), 0.0)

    def test_invalid_env_does_not_crash(self) -> None:
        """环境变量填错不该让服务起不来 —— 退化成关闭，并打一行警告。"""
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / ".abstain_threshold"
            self.assertEqual(load_threshold("abc", missing), 0.0)

    def test_garbage_file_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                load_threshold("", self._file(d, "not-a-number\n")), 0.0)

    def test_default_dir_is_pipeline(self) -> None:
        """不传文件路径时默认找 pipeline/.abstain_threshold。"""
        with tempfile.TemporaryDirectory() as d:
            f = self._file(d, "0.42\n")
            self.assertEqual(load_threshold("", None, default_dir=Path(d)), 0.42)
            self.assertTrue(f.exists())


class NoHeavyImportTests(unittest.TestCase):
    """回归：这个测试文件**不能**依赖 Web 框架。

    CI 只装 Phase 0 的 8 个包（llama-index-core / trafilatura / bs4 /
    markdownify / lxml / readability-lxml / qdrant-client / jieba）。
    曾经因为 `from rag_server import ...` 把 fastapi 拖进来，整个套件失败。
    """

    def test_rag_server_not_imported(self) -> None:
        self.assertNotIn("rag_server", sys.modules)
        self.assertNotIn("fastapi", sys.modules)


if __name__ == "__main__":
    unittest.main()
