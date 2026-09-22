r"""质量门测试 —— 重点是**结构保真度**这一维。

背景：质量门此前只看 token 数/长度/重复率/乱码率，对「结构丢了」完全无感。
issues 11/12 的两次回归（heading 层级被压平、`section_path` 全空、PDF 页码丢失）
当时都没被发现 —— 一本讲配置的手册丢了全部配置示例，质量分仍是 0.99。

这里既钉「该报的要报」，也钉「不该报的别报」（有 heading 才检查层级，
没 heading 的文档不该因为层级单一被扣分）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ingest.models import DocumentArtifact, DocumentBlock  # noqa: E402
from ingest.quality import assess_raw_quality  # noqa: E402

STRUCT_FLAGS = (
    "pdf_without_page_numbers",
    "flat_heading_levels",
    "section_path_all_empty",
    "long_document_without_headings",
)


def make_block(bid, btype, text, level=None, page=None, section_path=None):
    return DocumentBlock(
        block_id=bid, type=btype, text=text, markdown=text,
        section_path=list(section_path or []), order=0, page=page,
        quality_score=1.0,
        metadata={"level": level} if level is not None else {},
    )


def make_artifact(blocks, mime="text/markdown", title="测试文档"):
    return DocumentArtifact(
        schema_version="1.0", artifact_id="doc_q", source_uri="t",
        source_type="local_file", mime_type=mime, parser="plain_text",
        title=title, blocks=list(blocks),
    )


def struct_flags(report):
    return [f for f in report.flags if f in STRUCT_FLAGS]


class StructuralFidelityTests(unittest.TestCase):
    def test_healthy_pdf_reports_nothing(self) -> None:
        blocks = [
            make_block("h1", "heading", "第 1 章", level=1, page=1),
            make_block("t1", "text", "正文一。" * 40, page=1, section_path=["第 1 章"]),
            make_block("h2", "heading", "1.1 节", level=2, page=2),
            make_block("t2", "text", "正文二。" * 40, page=2,
                       section_path=["第 1 章", "1.1 节"]),
            make_block("h3", "heading", "1.2 节", level=2, page=3),
            make_block("t3", "text", "正文三。" * 40, page=3,
                       section_path=["第 1 章", "1.2 节"]),
            make_block("h4", "heading", "第 2 章", level=1, page=4),
            make_block("t4", "text", "正文四。" * 40, page=4, section_path=["第 2 章"]),
            make_block("h5", "heading", "2.1 节", level=2, page=5),
            make_block("t5", "text", "正文五。" * 40, page=5,
                       section_path=["第 2 章", "2.1 节"]),
            make_block("h6", "heading", "2.2 节", level=2, page=6),
            make_block("t6", "text", "正文六。" * 40, page=6,
                       section_path=["第 2 章", "2.2 节"]),
        ]
        report = assess_raw_quality(make_artifact(blocks, mime="application/pdf"))
        self.assertEqual(struct_flags(report), [])
        self.assertEqual(report.metrics["page_coverage"], 1.0)
        self.assertEqual(report.metrics["distinct_heading_levels"], 2)
        self.assertEqual(report.metrics["section_path_coverage"], 1.0)

    def test_pdf_without_page_numbers_is_flagged(self) -> None:
        """页码是引用（"第 N 页"）的刚需；经 Markdown 中转的解析器原理上保不住。"""
        blocks = [
            make_block("h1", "heading", "第 1 章", level=1),
            make_block("t1", "text", "正文。" * 80, section_path=["第 1 章"]),
        ]
        report = assess_raw_quality(make_artifact(blocks, mime="application/pdf"))
        self.assertIn("pdf_without_page_numbers", report.flags)
        self.assertEqual(report.metrics["page_coverage"], 0.0)

    def test_non_pdf_without_pages_is_not_flagged(self) -> None:
        """HTML/Markdown 本来就没有页的概念，不该因此被扣分。"""
        blocks = [
            make_block("h1", "heading", "标题", level=1),
            make_block("t1", "text", "正文。" * 80, section_path=["标题"]),
        ]
        report = assess_raw_quality(make_artifact(blocks, mime="text/html"))
        self.assertNotIn("pdf_without_page_numbers", report.flags)

    def test_flat_heading_levels_is_flagged(self) -> None:
        """回归：`2.3.1.` 被压成 lv2、`第3章` 也成 lv2 —— 层级被压平。"""
        blocks = [make_block("h%d" % i, "heading", "第 %d 节" % i, level=1)
                  for i in range(6)]
        blocks.append(make_block("t1", "text", "正文。" * 80,
                                 section_path=["第 1 节"]))
        report = assess_raw_quality(make_artifact(blocks))
        self.assertIn("flat_heading_levels", report.flags)
        self.assertEqual(report.metrics["distinct_heading_levels"], 1)

    def test_few_headings_do_not_trigger_flat_levels(self) -> None:
        """只有 1-2 个标题时层级单一很正常，不该报。"""
        blocks = [
            make_block("h1", "heading", "标题", level=1),
            make_block("t1", "text", "正文。" * 80, section_path=["标题"]),
        ]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertNotIn("flat_heading_levels", report.flags)

    def test_section_path_all_empty_is_flagged(self) -> None:
        """回归：同级标题互相嵌套，导致正文块的 section_path 全空。"""
        blocks = [make_block("h%d" % i, "heading", "第 %d 节" % i, level=2)
                  for i in range(4)]
        blocks += [make_block("t%d" % i, "text", "正文。" * 60) for i in range(4)]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertIn("section_path_all_empty", report.flags)
        self.assertEqual(report.metrics["section_path_coverage"], 0.0)

    def test_document_without_headings_is_not_flagged(self) -> None:
        """无 heading 的文档没有结构可言，不该因为「覆盖率为 0」被扣分。"""
        blocks = [make_block("t%d" % i, "text", "正文。" * 60) for i in range(5)]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertEqual(struct_flags(report), [])

    def test_structural_flags_lower_score_but_keep_document_indexable(self) -> None:
        """结构问题只做小扣分：目的是可见 + 压到 medium，不是一棒打死。"""
        blocks = [
            make_block("h%d" % i, "heading", "第 %d 节" % i, level=1)
            for i in range(6)
        ]
        blocks += [make_block("t%d" % i, "text", "正文。" * 60) for i in range(6)]
        report = assess_raw_quality(make_artifact(blocks, mime="application/pdf"))
        self.assertTrue(struct_flags(report))
        self.assertLess(report.score, 1.0)
        self.assertNotEqual(report.status, "reject")   # 内容还在，仍应进索引
        self.assertTrue(report.reasons)                # 要说清楚为什么被扣


class LongDocumentWithoutHeadingsTests(unittest.TestCase):
    """判据 4：**长文却一个标题都没有**。

    判据 2/3 都要求「有标题才检查」，于是 0 标题的文档反而**静默通过**。
    但一篇长文没有标题是可疑的：要么源本身无结构，要么结构在抽取时丢了。

    真实触发：3 篇 openai 文档抓不到源 HTML，回退用旧的 `document.md`
    （markdown 解析器），其中 2 篇 7846 / 12804 字符且 **0 标题** ——
    它们因此没有 section_path，按章节过滤和引用都无从谈起。
    而全部有标题的文档都 ≥ 8907 字符。
    """

    def test_long_document_without_headings_is_flagged(self) -> None:
        blocks = [make_block("t%d" % i, "text", "正文。" * 300) for i in range(3)]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertIn("long_document_without_headings", report.flags)
        self.assertTrue(any("没有任何标题" in r for r in report.reasons))

    def test_short_document_without_headings_is_not_flagged(self) -> None:
        """短文没标题很正常（一段话的便签、卡片），不该报。"""
        blocks = [make_block("t0", "text", "很短的一段话。")]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertNotIn("long_document_without_headings", report.flags)

    def test_long_document_with_headings_is_not_flagged(self) -> None:
        blocks = [make_block("h%d" % i, "heading", "第 %d 节" % i, level=1)
                  for i in range(6)]
        blocks += [make_block("t%d" % i, "text", "正文。" * 300) for i in range(3)]
        report = assess_raw_quality(make_artifact(blocks))
        self.assertNotIn("long_document_without_headings", report.flags)

    def test_threshold_boundary(self) -> None:
        """阈值 2000 字符：刚好够长才报。"""
        just_under = [make_block("t0", "text", "正" * 1999)]
        just_over = [make_block("t0", "text", "正" * 2001)]
        self.assertNotIn(
            "long_document_without_headings",
            assess_raw_quality(make_artifact(just_under)).flags)
        self.assertIn(
            "long_document_without_headings",
            assess_raw_quality(make_artifact(just_over)).flags)


if __name__ == "__main__":
    unittest.main()
