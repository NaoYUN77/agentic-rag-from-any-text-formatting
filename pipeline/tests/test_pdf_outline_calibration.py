r"""outline 模糊匹配 + 字号层级标定 的单元测试。

背景
----
这两个机制来自对 RAGFlow 的调研 (`rag/app/manual.py` + `rag/nlp/__init__.py`):

1. **字符二元组 Jaccard 匹配 outline** —— RAGFlow 用它把正文标题匹配到
   PDF 书签, 比精确匹配更耐 OCR/换行噪声。

2. **用文档自身观测到的层级来定层级** —— RAGFlow 的 `title_frequency`
   返回 `most_level`(最频繁的标题层级), 用文档分布而非绝对数值决定层级。

本项目的落地方式与 RAGFlow 有两点不同, 都在测试里固定住:

- 阈值取 0.85 而非 RAGFlow 的 0.8 (0.8 会把"追加后缀"的不同标题误判为同一标题)。
- 不做 `lvl <= most_level` 截断 —— 实测本文档 most_level=6, 照搬会把
  真实的深层章节全部截掉。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from collections import Counter

from ingest.parsers.pdf_common import (
    _bigram_jaccard,
    _calibrate_font_levels,
    _char_bigrams,
    _OutlineMatcher,
)


class CharBigramTests(unittest.TestCase):
    """二元组与 Jaccard 的基础行为。"""

    def test_ignores_whitespace_differences(self) -> None:
        """归一化后空白差异消失, 完全相同。"""
        a = _char_bigrams("A.2. 准备 HAPROXY 节点")
        b = _char_bigrams("A.2. 准备 HAPROXY 节 点")
        self.assertEqual(_bigram_jaccard(a, b), 1.0)

    def test_single_char_ocr_error_still_scores_high(self) -> None:
        """单字符 OCR 错应仍在阈值之上。"""
        a = _char_bigrams("2.3.1. keepalived Scheduling Algorithms")
        b = _char_bigrams("2.3.1. keepalive Scheduling Algorithms")
        self.assertGreaterEqual(_bigram_jaccard(a, b), 0.9)

    def test_appended_suffix_stays_below_threshold(self) -> None:
        """标题追加后缀是**不同章节**, 必须落在阈值之下。

        实测该比值 = 0.833。若阈值取 RAGFlow 的 0.8 会误匹配,
        所以本项目用 0.85。
        """
        a = _char_bigrams("HAPROXY 调度算法")
        b = _char_bigrams("HAPROXY 调度算法概述")
        score = _bigram_jaccard(a, b)
        self.assertLess(score, 0.85, f"追加后缀不应过阈值: {score}")
        self.assertGreater(score, 0.8, "这正是 0.8 会误判的区间")

    def test_unrelated_titles_score_zero(self) -> None:
        a = _char_bigrams("法律通告")
        b = _char_bigrams("摘要")
        self.assertEqual(_bigram_jaccard(a, b), 0.0)

    def test_empty_inputs(self) -> None:
        self.assertEqual(_char_bigrams(""), set())
        self.assertEqual(_bigram_jaccard(set(), set()), 0.0)


class OutlineMatcherTests(unittest.TestCase):
    """两级匹配: 精确优先, 模糊兜底。"""

    _ENTRIES = [
        (1, "第 2 章 KEEPALIVED 概述", 8),
        (2, "2.3. KEEPALIVED 计划概述", 11),
        (3, "2.3.1. keepalived Scheduling Algorithms", 11),
    ]

    def test_exact_match(self) -> None:
        m = _OutlineMatcher(self._ENTRIES)
        self.assertEqual(m.match("第 2 章 KEEPALIVED 概述", 8), 1)
        self.assertEqual(m.match("2.3. KEEPALIVED 计划概述", 11), 2)

    def test_exact_match_tolerates_whitespace(self) -> None:
        """归一化后应命中精确层。"""
        m = _OutlineMatcher(self._ENTRIES)
        self.assertEqual(m.match("第 2 章  KEEPALIVED   概述", 8), 1)

    def test_fuzzy_recovers_ocr_error(self) -> None:
        """精确层失败时, 模糊层应把单字符 OCR 错救回来。"""
        m = _OutlineMatcher(self._ENTRIES)
        # "keepalived" 少一个 d —— 精确匹配必然失败
        self.assertEqual(
            m.match("2.3.1. keepalive Scheduling Algorithms", 11), 3)

    def test_fuzzy_does_not_match_different_section(self) -> None:
        """追加后缀的不同标题不得被模糊匹配吃掉。"""
        m = _OutlineMatcher(self._ENTRIES)
        self.assertIsNone(m.match("2.3. KEEPALIVED 计划概述补充说明", 11))

    def test_no_match_for_unrelated_text(self) -> None:
        m = _OutlineMatcher(self._ENTRIES)
        self.assertIsNone(m.match("这是一段完全无关的正文内容", 11))

    def test_short_title_skips_fuzzy(self) -> None:
        """过短标题二元组太少, 模糊匹配会误伤 —— 直接跳过。"""
        m = _OutlineMatcher([(1, "1", 5), (1, "2", 6)])
        self.assertIsNone(m.match("3", 5))

    def test_page_slack_allows_off_by_one(self) -> None:
        """outline 页码与正文页码可能差一页, 模糊层允许 ±1。"""
        m = _OutlineMatcher(self._ENTRIES, page_slack=1)
        # 标题在 outline 记 p11, 但正文出现在 p12
        self.assertEqual(
            m.match("2.3.1. keepalive Scheduling Algorithms", 12), 3)

    def test_page_slack_zero_disables_tolerance(self) -> None:
        m = _OutlineMatcher(self._ENTRIES, page_slack=0)
        self.assertIsNone(m.match("2.3.1. keepalive Scheduling Algorithms", 12))

    def test_caches_results(self) -> None:
        m = _OutlineMatcher(self._ENTRIES)
        first = m.match("第 2 章 KEEPALIVED 概述", 8)
        second = m.match("第 2 章 KEEPALIVED 概述", 8)
        self.assertEqual(first, second)
        # 缓存键是 (归一化文本, 页码); _normalize 去空白但**不**改大小写
        self.assertIn(("第2章KEEPALIVED概述", 8), m._cache)


class CalibrateFontLevelsTests(unittest.TestCase):
    """用 outline 实测层级校正字号排名阶梯。"""

    _RANK_LADDER = {
        26.9: 1, 23.0: 2, 16.3: 3, 15.4: 4, 14.4: 5, 13.4: 6, 12.5: 7,
    }

    def test_no_outline_falls_back_to_rank_ladder(self) -> None:
        """没有 outline 观测时退化为原排名阶梯, 不得更差。"""
        result = _calibrate_font_levels(self._RANK_LADDER, {})
        self.assertEqual(result, self._RANK_LADDER)

    def test_direct_observation_wins(self) -> None:
        """有 outline 观测的字号直接采用实测层级。"""
        obs = {16.3: Counter({1: 4}), 14.4: Counter({2: 17}),
               12.5: Counter({3: 21})}
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        self.assertEqual(result[16.3], 1)
        self.assertEqual(result[14.4], 2)
        self.assertEqual(result[12.5], 3)

    def test_majority_vote_on_conflicting_observations(self) -> None:
        """同一字号有多个观测层级时取多数票。"""
        obs = {14.4: Counter({2: 17, 3: 2})}
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        self.assertEqual(result[14.4], 2)

    def test_majority_tie_prefers_shallower(self) -> None:
        """平票时取较浅层级, 避免过度加深。"""
        obs = {14.4: Counter({2: 3, 3: 3})}
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        self.assertEqual(result[14.4], 2)

    def test_interpolates_for_unobserved_size(self) -> None:
        """无观测的字号按字号大小找最近的已标定字号。"""
        obs = {16.3: Counter({1: 4}), 12.5: Counter({3: 21})}
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        # 15.4 更靠近 16.3 -> lv1
        self.assertEqual(result[15.4], 1)
        # 13.4 更靠近 12.5 -> lv3
        self.assertEqual(result[13.4], 3)

    def test_observation_for_size_not_in_ladder_is_ignored(self) -> None:
        """观测里出现阶梯外的字号时应忽略, 不污染结果。"""
        obs = {99.9: Counter({1: 5}), 14.4: Counter({2: 17})}
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        self.assertNotIn(99.9, result)
        self.assertEqual(result[14.4], 2)

    def test_front_matter_tier_no_longer_deepest(self) -> None:
        """核心回归: 前置页字号不得拿到比所有正文标题都深的层级。

        真实文档里 13.4pt 被排名阶梯判成 lv6(比最深正文 lv4 还深),
        而 outline 实测 13.4pt 对应 lv1(目录)。标定后应修正。
        """
        obs = {
            16.3: Counter({1: 4}), 14.4: Counter({2: 17}),
            13.4: Counter({1: 1}), 12.5: Counter({3: 21}),
        }
        result = _calibrate_font_levels(self._RANK_LADDER, obs)
        self.assertEqual(result[13.4], 1)
        deepest_body = max(result[s] for s in (16.3, 14.4, 12.5))
        self.assertLessEqual(
            result[13.4], deepest_body,
            "前置页字号不应比正文标题更深",
        )

    def test_empty_ladder(self) -> None:
        self.assertEqual(_calibrate_font_levels({}, {}), {})


class RealPdfCalibrationTests(unittest.TestCase):
    """真实 PDF 上的端到端断言 (文件不存在时自动跳过)。"""

    _PDF = Path(
        r"C:\Users\l\Desktop"
        r"\Red_Hat_Enterprise_Linux-7-Load_Balancer_Administration-zh-CN.pdf"
    )

    def _artifact(self):
        if not self._PDF.exists():
            self.skipTest(f"真实 PDF 不存在, 跳过: {self._PDF}")
        try:
            import pdfplumber
        except ImportError:  # pragma: no cover
            self.skipTest("pdfplumber 未安装")

        import hashlib

        from ingest.models import SourcePayload
        from ingest.parsers.pdf_common import build_artifact
        from ingest.parsers.pdf_pdfplumber import PdfplumberSource

        data = self._PDF.read_bytes()
        payload = SourcePayload(
            source_type="file", uri=str(self._PDF), final_uri=str(self._PDF),
            mime_type="application/pdf", content_type="application/pdf",
            data=data, encoding=None, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        with pdfplumber.open(str(self._PDF)) as pdf:
            return build_artifact(PdfplumberSource(pdf), payload)

    def test_no_heading_level_exceeds_real_structure_depth(self) -> None:
        """真实文档里不应出现"层级比所有章节都深"的标题。

        标定前: 13.4pt 前置页标题拿到 lv6, 而最深正文只有 lv4 ——
        前置页反而比正文更深, 语义错误。
        """
        art = self._artifact()
        levels = [
            b.metadata["level"] for b in art.blocks if b.type == "heading"
        ]
        self.assertTrue(levels)
        # 文档真实结构由 outline 给出, 最深 4 级; 字号启发式不应突破它太多
        self.assertLessEqual(
            max(levels), 4,
            f"存在超出真实结构深度的标题层级: {sorted(set(levels))}",
        )

    def test_calibration_metadata_is_exposed(self) -> None:
        """标定结果与观测样本应写进 artifact metadata, 便于诊断。"""
        art = self._artifact()
        md = art.metadata
        self.assertIn("font_levels_calibrated", md)
        self.assertIn("font_level_observations", md)
        self.assertTrue(md["font_levels_calibrated"])

    def test_front_matter_tier_is_calibrated_down(self) -> None:
        """13.4pt 这一档必须被标定到浅层, 而不是排名阶梯给的 lv6。"""
        art = self._artifact()
        md = art.metadata
        rank_ladder = md["heading_size_levels"]
        calibrated = md["font_levels_calibrated"]
        self.assertIn("13.4", rank_ladder)
        self.assertEqual(rank_ladder["13.4"], 6, "前置条件: 排名阶梯给 lv6")
        self.assertLess(
            calibrated["13.4"], 6,
            f"13.4pt 应被标定到更浅的层级, 实际 {calibrated['13.4']}",
        )

    def test_outline_coverage_is_high(self) -> None:
        """outline 是权威层级来源, 覆盖率应远高于字号启发式。"""
        art = self._artifact()
        heads = [b for b in art.blocks if b.type == "heading"]
        from_outline = sum(1 for b in heads if b.metadata.get("from_outline"))
        self.assertGreater(
            from_outline / max(len(heads), 1), 0.8,
            f"outline 覆盖率过低: {from_outline}/{len(heads)}",
        )


if __name__ == "__main__":
    unittest.main()
