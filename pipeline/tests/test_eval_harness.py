r"""评估 harness 的自检。

最容易犯的错是: harness 自己写错了, 却拿它去指导优化决策。
所以这里重点验证 harness **本身**是对的, 而不是验证检索质量。

其中【退化测试】最有价值: 把 gold 换成检索结果 top1, recall@1 必须是 1.0。
这一条能一次性验证 resolve、检索、指标三处都没接错。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from eval.metrics import (  # noqa: E402
    aggregate, hit_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k,
)
from eval.qrels import GoldSpan, QrelsError, QrelsItem  # noqa: E402
from eval.resolve import (  # noqa: E402
    GoldResolveError, FragmentIndex, FragmentRef, normalize_text, resolve_gold,
)
from eval.run_eval import gap_analysis, required_stages  # noqa: E402


class GapAnalysisTests(unittest.TestCase):
    """issue 03 的待验证项：`top1 - top2` 分差与排序可靠性的关系。

    ⚠️ 用 hit@5 而不是 MRR —— 分差小天然让 MRR 偏低，拿 MRR 分析是循环论证。
    这里把「用 hit@5」这个选择钉住。
    """

    @staticmethod
    def row(gap, hit, mrr=0.0):
        return {
            "dense_score_gap": gap,
            "stages": {"dense": {"scores": {"hit@5": hit, "mrr@20": mrr}}},
        }

    def test_insufficient_rows_returns_none(self) -> None:
        rows = [self.row(0.1, 1.0) for _ in range(5)]
        self.assertIsNone(gap_analysis(rows))

    def test_rows_without_gap_are_skipped(self) -> None:
        rows = [self.row(None, 1.0) for _ in range(20)]
        self.assertIsNone(gap_analysis(rows))

    def test_median_split_and_delta(self) -> None:
        # 分差小的 6 条全不中，分差大的 6 条全中 -> delta = 1.0
        rows = [self.row(0.001 * (i + 1), 0.0) for i in range(6)]
        rows += [self.row(1.0 + i, 1.0) for i in range(6)]
        out = gap_analysis(rows)
        self.assertEqual(out["n"], 12)
        self.assertEqual(out["low_gap"]["n"], 6)
        self.assertEqual(out["high_gap"]["n"], 6)
        self.assertEqual(out["low_gap"]["hit@5"], 0.0)
        self.assertEqual(out["high_gap"]["hit@5"], 1.0)
        self.assertEqual(out["delta"], 1.0)

    def test_uses_hit_at_5_not_mrr(self) -> None:
        """分差小但 hit@5 高时，结论必须是「分差小那组不差」。

        若实现里误用了 MRR（分差小天然 MRR 低），这里会得出反向结论。
        """
        rows = [self.row(0.001 * (i + 1), 1.0, mrr=0.10) for i in range(6)]
        rows += [self.row(1.0 + i, 1.0, mrr=0.99) for i in range(6)]
        out = gap_analysis(rows)
        self.assertEqual(out["low_gap"]["hit@5"], 1.0)      # 不是 0.10
        self.assertEqual(out["delta"], 0.0)

    def test_negative_delta_when_small_gap_is_better(self) -> None:
        rows = [self.row(0.001 * (i + 1), 1.0) for i in range(6)]
        rows += [self.row(1.0 + i, 0.0) for i in range(6)]
        out = gap_analysis(rows)
        self.assertLess(out["delta"], 0)


class StageDependencyTests(unittest.TestCase):
    """`--stages` 的下游阶段必须自动带上 dense + sparse。

    回归: 曾经只请求 `--stages rrf` 时，dense/sparse 因为不在 stages 里而不被
    计算，rrf 拿两个空列表做融合 → **指标全 0 却不报错**，看起来像「模型效果差」。
    实测踩过：单独跑 rrf 得 0.000，跑全链路得 0.971。
    """

    def test_downstream_pulls_in_both_retrievers(self) -> None:
        for stage in ("union", "rrf", "rerank"):
            with self.subTest(stage=stage):
                need = required_stages([stage])
                self.assertIn("dense", need)
                self.assertIn("sparse", need)
                self.assertIn(stage, need)

    def test_dense_only_does_not_pull_sparse(self) -> None:
        """只测 dense 时不该白跑 sparse —— 省一半检索开销。"""
        need = required_stages(["dense"])
        self.assertIn("dense", need)
        self.assertNotIn("sparse", need)
        self.assertNotIn("rrf", need)

    def test_full_list_is_unchanged(self) -> None:
        stages = ["dense", "sparse", "union", "rrf", "rerank"]
        self.assertEqual(required_stages(stages), set(stages))

    def test_empty_stages_stay_empty(self) -> None:
        self.assertEqual(required_stages([]), set())


class NormalizeTests(unittest.TestCase):
    """normalize_text 的边界行为。

    背景: 去掉 Markdown 中转层后, 正文里不再有 ** 之类的渲染标记,
    而旧 gold 片段里可能还带着。两侧统一归一化, 标注才不会随解析层变动失效。

    但归一化**不能过度** —— 见 test_preserves_identifiers。
    """

    def test_strips_bold_markers(self) -> None:
        self.assertEqual(
            normalize_text("**Context** refers to the set of tokens."),
            "Context refers to the set of tokens.",
        )

    def test_strips_italic_and_code(self) -> None:
        self.assertEqual(normalize_text("__bold__ and _it_"), "bold and it")
        self.assertEqual(normalize_text("`code_span` here"), "code_span here")

    def test_preserves_identifiers(self) -> None:
        """下划线在标识符里不是强调标记, 不能删。

        第一版实现用 `[*_`]` 无脑删除, 把 doc_a 变成了 doca ——
        这类文本在技术文档里遍地都是, 必须保。
        """
        for s in ("这是文档 doc_a 里的一段可检索正文。",
                  "use snake_case_names here"):
            self.assertEqual(normalize_text(s), s)

    def test_leaves_lone_asterisks_alone(self) -> None:
        """空格包围的 * 是乘号/符号, 不是强调。"""
        self.assertEqual(normalize_text("x * 2 * y"), "x * 2 * y")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("multi\nline   text"), "multi line text")

    def test_strips_markdown_links(self) -> None:
        """OLD(Markdown 中转)语料里超链接是 [anchor](url), NEW(DOM 直产)只有 anchor。

        URL 不是正文内容, 两侧统一剥掉后标注才跨 parser 存活
        (2026-09-19 ce_02 在 OLD 语料 FAIL 的根因)。
        """
        self.assertEqual(
            normalize_text("a [simple definition](https://x/y) for agents"),
            "a simple definition for agents",
        )
        # 链接 anchor 里带强调标记: 先剥链接再剥强调
        self.assertEqual(
            normalize_text("see [**Managed Agents**](https://x) here"),
            "see Managed Agents here",
        )

    def test_collapses_space_before_punctuation(self) -> None:
        """NEW 语料 DOM 行内元素边界会挤出 "Agents , on" 这样的标点前空格。

        两侧统一折叠 (2026-09-19 ag_01/cx_03 在 OLD 语料 FAIL 的根因)。
        """
        self.assertEqual(
            normalize_text("Agents , on the other hand"),
            "Agents, on the other hand",
        )
        self.assertEqual(
            normalize_text("sandboxing , resource limits"),
            "sandboxing, resource limits",
        )

    def test_empty(self) -> None:
        self.assertEqual(normalize_text(""), "")


class NormalizedResolveTests(unittest.TestCase):
    """归一化必须同时作用于索引侧和查询侧 —— 只做一边等于没做。"""

    def test_gold_with_markers_matches_text_without(self) -> None:
        idx = FragmentIndex()
        # 索引侧: 新语料, 没有标记
        idx.add(FragmentRef(
            chunk_id="c1", point_id=None, artifact_id="doc_a",
            block_id="b0001", block_type="text", start_char=0, end_char=40,
            text=normalize_text("Context refers to the set of tokens included."),
        ))
        # gold 侧: 旧标注, 带 ** 标记
        rep = resolve_gold(
            GoldSpan(text="**Context** refers to the set of tokens included."),
            idx, query_id="q1", gold_index=0,
        )
        self.assertTrue(rep.resolved)
        self.assertEqual(rep.chunk_ids, {"c1"})


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = {"a", "b", "c"}
        self.r = ["a", "x", "c", "y"]

    def test_recall(self) -> None:
        self.assertAlmostEqual(recall_at_k(self.r, self.gold, 2), 1 / 3)
        self.assertAlmostEqual(recall_at_k(self.r, self.gold, 4), 2 / 3)

    def test_hit(self) -> None:
        self.assertEqual(hit_at_k(self.r, self.gold, 1), 1.0)
        self.assertEqual(hit_at_k(["x", "y"], self.gold, 2), 0.0)

    def test_mrr(self) -> None:
        self.assertEqual(mrr_at_k(self.r, self.gold, 4), 1.0)
        self.assertEqual(mrr_at_k(["x", "a"], self.gold, 2), 0.5)
        self.assertEqual(mrr_at_k(["x", "y"], self.gold, 2), 0.0)

    def test_ndcg_perfect_ranking_is_one(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], self.gold, 3), 1.0)

    def test_ndcg_is_rank_sensitive(self) -> None:
        """把命中提前, nDCG 必须上升 —— 否则指标接错了。"""
        self.assertGreater(
            ndcg_at_k(["a", "x", "y"], self.gold, 3),
            ndcg_at_k(["x", "y", "a"], self.gold, 3),
        )

    def test_precision_uses_actual_length(self) -> None:
        """k 大于实际返回条数时按实际算, 避免虚低。"""
        self.assertAlmostEqual(precision_at_k(["a"], self.gold, 10), 1.0)

    def test_empty_gold_returns_none_not_zero(self) -> None:
        """unanswerable 没有 gold, 三个指标都必须返回 None。"""
        self.assertIsNone(recall_at_k(["a"], set(), 5))
        self.assertIsNone(mrr_at_k(["a"], set(), 5))
        self.assertIsNone(hit_at_k(["a"], set(), 5))

    def test_aggregate_excludes_none(self) -> None:
        agg = aggregate([
            {"recall@5": None}, {"recall@5": 1.0}, {"recall@5": 0.5},
        ])
        self.assertAlmostEqual(agg["recall@5"], 0.75)
        self.assertEqual(agg["recall@5__n"], 2.0)

    def test_degenerate_gold_equals_top1(self) -> None:
        """退化测试: gold 就是 top1 时, recall@1 / mrr@1 / hit@1 都必须是 1.0。

        这一条一次性验证 resolve + 检索 + 指标三处都没接错。
        """
        retrieved = ["doc_x_c0001", "doc_x_c0002"]
        gold = {"doc_x_c0001"}
        self.assertEqual(recall_at_k(retrieved, gold, 1), 1.0)
        self.assertEqual(mrr_at_k(retrieved, gold, 1), 1.0)
        self.assertEqual(hit_at_k(retrieved, gold, 1), 1.0)


class QrelsSchemaTests(unittest.TestCase):
    def test_rejects_non_unanswerable_without_gold(self) -> None:
        with self.assertRaises(QrelsError):
            QrelsItem(query_id="x", query="q", gold=[],
                      category="semantic").validate()

    def test_rejects_unanswerable_with_gold(self) -> None:
        with self.assertRaises(QrelsError):
            QrelsItem(query_id="x", query="q",
                      gold=[GoldSpan(text="一段足够长的原文片段用于测试")],
                      category="unanswerable").validate()

    def test_rejects_too_short_gold(self) -> None:
        """太短的片段会多命中, 必须拒绝。"""
        with self.assertRaises(QrelsError):
            QrelsItem(query_id="x", query="q", gold=[GoldSpan(text="太短")],
                      category="semantic").validate()

    def test_rejects_unknown_category(self) -> None:
        with self.assertRaises(QrelsError):
            QrelsItem(query_id="x", query="q",
                      gold=[GoldSpan(text="一段足够长的原文片段用于测试")],
                      category="not_a_category").validate()

    def test_accepts_valid_item(self) -> None:
        QrelsItem(
            query_id="x", query="q",
            gold=[GoldSpan(text="一段足够长的原文片段用于测试")],
            category="semantic", split="dev",
        ).validate()


class ResolveTests(unittest.TestCase):
    def _index(self) -> FragmentIndex:
        idx = FragmentIndex()
        # 同一个 block_id 出现在两篇文档里 —— 模拟实测的跨文档碰撞
        for aid in ("doc_a", "doc_b"):
            idx.add(FragmentRef(
                chunk_id=f"{aid}_c0001", point_id=None, artifact_id=aid,
                block_id="b0001", block_type="text",
                start_char=0, end_char=20,
                text="这是文档 %s 里的一段可检索正文。" % aid,
            ))
        # image 的空 fragment, 必须被跳过
        idx.add(FragmentRef(
            chunk_id="doc_a_c0002", point_id=None, artifact_id="doc_a",
            block_id="b0002", block_type="image",
            start_char=0, end_char=0, text="",
        ))
        return idx

    def test_composite_key_prevents_cross_document_collision(self) -> None:
        """核心回归: 同一个 block_id 在不同文档里必须分开。

        若只用裸 block_id 建键, 这里两个 fragment 会互相覆盖,
        gold 反查会命中错误的 chunk 且不报错。
        """
        idx = self._index()
        self.assertEqual(idx.keys, 2, "应为 (doc_a,b0001) 与 (doc_b,b0001) 两个键")

    def test_skips_image_empty_fragment(self) -> None:
        idx = self._index()
        self.assertEqual(idx.skipped_image, 1)

    def test_resolve_matches_by_text(self) -> None:
        idx = self._index()
        rep = resolve_gold(
            GoldSpan(text="这是文档 doc_a 里的一段可检索正文。"),
            idx, query_id="q1", gold_index=0,
        )
        self.assertTrue(rep.resolved)
        self.assertEqual(rep.chunk_ids, {"doc_a_c0001"})

    def test_resolve_raises_when_not_found(self) -> None:
        """解析失败必须抛错 —— 静默跳过会让 recall 虚高。"""
        idx = self._index()
        with self.assertRaises(GoldResolveError):
            resolve_gold(GoldSpan(text="这段文字在语料里绝对不存在 xyzzy"),
                         idx, query_id="q1", gold_index=0)

    def test_resolve_raises_on_too_short_snippet(self) -> None:
        idx = self._index()
        with self.assertRaises(GoldResolveError):
            resolve_gold(GoldSpan(text="短"), idx, query_id="q1", gold_index=0)

    def test_gold_can_map_to_multiple_chunks(self) -> None:
        """同一段文字可能落在多个 chunk 里, 所以 gold 必须是集合。

        注: overlap 已于 2026-09-21 取消, 不再是这里的成因。但重复文本仍会出现
        （如各页重复的样板文字、同一 block 被多处引用），因此 resolve_gold
        返回集合这个行为依然必要。
        """
        idx = FragmentIndex()
        for i in (1, 2):
            idx.add(FragmentRef(
                chunk_id=f"doc_a_c000{i}", point_id=None, artifact_id="doc_a",
                block_id="b0001", block_type="text",
                start_char=0, end_char=24,
                text="这段文字同时出现在两个 chunk 里。",
            ))
        rep = resolve_gold(
            GoldSpan(text="这段文字同时出现在两个 chunk 里。"),
            idx, query_id="q1", gold_index=0,
        )
        self.assertEqual(len(rep.chunk_ids), 2)


if __name__ == "__main__":
    unittest.main()
