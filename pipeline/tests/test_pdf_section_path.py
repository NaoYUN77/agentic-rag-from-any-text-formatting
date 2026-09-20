r"""PDF section_path 嵌套回归测试。

背景
----
2026-09-19 修复的 bug: `heading_stack` 只存标题文本, 弹栈条件是
`while len(stack) >= level`。这把"栈深度"和"字号排名"两种量纲相比,
导致**同字号的连续标题不会互相弹出**, 而是层层嵌套。

触发条件有两个, 缺一不可:
  1. 文档有 >= 7 档标题字号, 使某一档的 rank **超过**当前栈深度。
     旧实现 `min(idx+1, 6)` 把第 7 档压到 lv6; 即便不压, rank 仍可能
     大于栈深 (例如 rank=6, 栈只有 3 层)。
  2. 同一 rank 的标题连续出现 (如「法律通告」和「摘要」同为 13.4pt)。

实测文档 (Red Hat 负载平衡器手册) 恰好满足: 7 档字号, 且前置页有
三个 13.4pt 标题。旧逻辑下 `len(stack) >= 6` 永不成立, 于是:

    配置 Keepalived 和 HAProxy   -> 父级
    法律通告                      -> 错成 配置Keepalived 的子级
    摘要                          -> 错成 法律通告 的子级 (!)

摘要正文的 section_path 变成 [..., '法律通告', '摘要'] —— 摘要被塞进
法律通告里, 整棵章节树错位。

修复
----
1. `_heading_size_levels` 去掉 `min(idx+1, 6)` 上限, 改为单调递增。
2. `heading_stack` 改存 `(rank, text)`, 弹栈条件改为
   `while stack and stack[-1][0] >= level` —— 同级标题正确成为兄弟。

注意: 测试用例**必须**包含 >= 7 档字号, 否则复现不出旧 bug
(已实测: 只有 1~2 档字号时新旧实现结果相同, 测试无法区分)。
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingest.models import SourcePayload
from ingest.parsers.pdf_common import PdfLine, build_artifact


class _FakePdfSource:
    """最小 PdfSource 实现, 用于精确控制字号与标题结构。"""

    def __init__(self, lines_by_page: List[List[PdfLine]]) -> None:
        self._pages = lines_by_page
        self.page_count = len(lines_by_page)

    def metadata(self) -> Dict[str, Any]:
        return {"title": "fake"}

    def outline(self) -> List[Tuple[int, str, int]]:
        return []

    def page_height(self, index: int) -> float:
        return 800.0

    def page_lines(self, index: int) -> List[PdfLine]:
        return self._pages[index]

    def close(self) -> None:
        return None


def _line(text: str, size: float, y: float) -> PdfLine:
    return PdfLine(text=text, bbox=[40.0, y, 560.0, y + size], size=size,
                   font="Fake")


def _payload() -> SourcePayload:
    data = b"%PDF-fake"
    return SourcePayload(
        source_type="file", uri="fake.pdf", final_uri="fake.pdf",
        mime_type="application/pdf", content_type="application/pdf",
        data=data, encoding=None, size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


# 7 档标题字号: 复现真实文档的字号分布, 使最低档拿到 rank 7
_SEVEN_SIZES = [26.9, 23.0, 16.3, 15.4, 14.4, 13.4, 12.5]


def _page_with_seven_sizes(extra: List[PdfLine],
                           tail: Optional[List[PdfLine]] = None) -> List[PdfLine]:
    """构造一页: 正文确定 body_size, 再放 7 档标题字号, 最后追加 extra 行。

    tail 用于在每个 extra 标题后插入正文, 以便断言正文的 section_path。
    若提供 tail, 则 extra 与 tail 交替排列。
    """
    lines: List[PdfLine] = []
    y = 80.0
    for _ in range(8):
        lines.append(_line("正文内容样本" * 8, 10.6, y))
        y += 20.0
    y += 40.0
    for size in _SEVEN_SIZES:
        lines.append(_line(f"占位标题{size}", size, y))
        y += 50.0
    if tail:
        for head, body in zip(extra, tail):
            lines.append(_line(head.text, head.size, y))
            y += 50.0
            lines.append(_line(body.text, body.size, y))
            y += 40.0
    else:
        for ln in extra:
            lines.append(_line(ln.text, ln.size, y))
            y += 50.0
    return lines


class SectionPathNestingTests(unittest.TestCase):
    """同字号标题必须是兄弟, 不能互相嵌套。"""

    def test_same_rank_headings_are_siblings(self) -> None:
        """同 rank 标题必须落在同一层级 (不互相嵌套)。

        构造: 先让 7 档字号确定 rank (13.4pt -> rank 6), 再放三个 13.4pt 标题。
        旧实现 `while len(stack) >= level` 在 rank(6) > 栈深 时永不弹栈,
        同级标题会层层嵌套; 修复后三者 section_path 完全相同。
        """
        heads = [
            _line("配置 Keepalived 和 HAProxy", 13.4, 0.0),
            _line("法律通告", 13.4, 0.0),
            _line("摘要", 13.4, 0.0),
        ]
        art = build_artifact(
            _FakePdfSource([_page_with_seven_sizes(heads)]), _payload())
        heads_map = {
            h.text: tuple(h.section_path)
            for h in art.blocks
            if h.type == "heading" and not h.text.startswith("占位标题")
        }
        self.assertEqual(len(heads_map), 3, f"应识别出 3 个目标标题: {heads_map}")
        # 三个同 rank 标题必须互为兄弟 => section_path 完全一致
        self.assertEqual(
            len(set(heads_map.values())), 1,
            f"同 rank 标题的 section_path 应完全相同: {heads_map}",
        )

    def test_same_size_abstract_not_nested_under_legal(self) -> None:
        """摘要正文的 section_path 不得包含「法律通告」。"""
        heads = [
            _line("配置 Keepalived 和 HAProxy", 13.4, 0.0),
            _line("法律通告", 13.4, 0.0),
            _line("摘要", 13.4, 0.0),
        ]
        bodies = [
            _line("配置说明正文" * 6, 10.6, 0.0),
            _line("版权信息正文" * 6, 10.6, 0.0),
            _line("摘要内容正文" * 6, 10.6, 0.0),
        ]
        art = build_artifact(
            _FakePdfSource([_page_with_seven_sizes(heads, bodies)]), _payload())

        abstract_body = next(
            b for b in art.blocks if b.text.startswith("摘要内容正文"))
        self.assertNotIn(
            "法律通告", abstract_body.section_path,
            f"摘要正文被错误嵌套进法律通告: {abstract_body.section_path}",
        )
        self.assertEqual(abstract_body.section_path[-1], "摘要")
        self.assertNotIn(
            "配置 Keepalived 和 HAProxy", abstract_body.section_path,
            f"摘要正文被错误嵌套进前置标题: {abstract_body.section_path}",
        )

        # 版权正文应归属法律通告
        legal_body = next(
            b for b in art.blocks if b.text.startswith("版权信息正文"))
        self.assertEqual(legal_body.section_path[-1], "法律通告")

    def test_distinct_sizes_get_distinct_levels(self) -> None:
        """不同字号必须拿到不同 level (旧实现 min(idx+1,6) 会压平第 7 档)。"""
        art = build_artifact(
            _FakePdfSource([_page_with_seven_sizes([])]), _payload())
        headings = [
            b for b in art.blocks
            if b.type == "heading" and b.text.startswith("占位标题")
        ]
        levels = [h.metadata["level"] for h in headings]
        self.assertEqual(
            len(set(levels)), len(levels),
            f"不同字号不应共用同一 level (压平 bug): levels={levels}",
        )
        self.assertEqual(levels, sorted(levels), "字号越大 level 应越小")

    def test_shallow_then_deep_headings_shrink_stack(self) -> None:
        """深标题之后遇到浅标题, 栈必须正确收缩。"""
        extra: List[PdfLine] = []
        art = build_artifact(
            _FakePdfSource([_page_with_seven_sizes(extra)]), _payload())
        heads = [b for b in art.blocks if b.type == "heading"]
        # 占位标题按字号从大到小排列, 应形成严格递增的路径深度
        depths = [len(h.section_path) for h in heads]
        self.assertEqual(depths, list(range(len(depths))),
                         f"路径深度应随字号递减而递增: {depths}")

    def test_section_path_depth_never_exceeds_headings(self) -> None:
        """section_path 长度不得超过实际出现过的标题层数。"""
        art = build_artifact(
            _FakePdfSource([_page_with_seven_sizes([])]), _payload())
        n_heads = sum(1 for b in art.blocks if b.type == "heading")
        self.assertLessEqual(
            max(len(b.section_path) for b in art.blocks),
            n_heads,
            "section_path 深度异常, 疑似同级标题被嵌套",
        )


class RealPdfSectionPathTests(unittest.TestCase):
    """对真实 PDF 的回归断言 (文件不存在时自动跳过)。

    这是最强的回归守卫: 合成用例只能覆盖部分路径, 真实文档同时具备
    7 档字号 + 浅栈 + 同 rank 连续标题三个条件。
    """

    _PDF = Path(
        r"C:\Users\l\Desktop"
        r"\Red_Hat_Enterprise_Linux-7-Load_Balancer_Administration-zh-CN.pdf"
    )

    def test_abstract_not_nested_under_legal_notice(self) -> None:
        """摘要正文的 section_path 不得包含「法律通告」 (2026-09-19 bug)。"""
        if not self._PDF.exists():
            self.skipTest(f"真实 PDF 不存在, 跳过: {self._PDF}")

        try:
            import pdfplumber
        except ImportError:  # pragma: no cover
            self.skipTest("pdfplumber 未安装")

        from ingest.parsers.pdf_pdfplumber import PdfplumberSource

        data = self._PDF.read_bytes()
        payload = SourcePayload(
            source_type="file", uri=str(self._PDF), final_uri=str(self._PDF),
            mime_type="application/pdf", content_type="application/pdf",
            data=data, encoding=None, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        with pdfplumber.open(str(self._PDF)) as pdf:
            art = build_artifact(PdfplumberSource(pdf), payload, pages="1-12")

        headings = [b for b in art.blocks if b.type == "heading"]
        by_text = {h.text: h for h in headings}

        # 前置页三个 13.4pt 标题必须同级
        if "法律通告" in by_text and "摘要" in by_text:
            self.assertEqual(
                by_text["法律通告"].section_path,
                by_text["摘要"].section_path,
                "「法律通告」与「摘要」应为兄弟, 不应互相嵌套",
            )
            self.assertNotIn(
                "法律通告", by_text["摘要"].section_path,
                "「摘要」被错误嵌套进「法律通告」",
            )

        # 摘要正文不得带「法律通告」
        for b in art.blocks:
            if b.text.startswith("构建负载平衡器系统"):
                self.assertNotIn(
                    "法律通告", b.section_path,
                    f"摘要正文 section_path 被污染: {b.section_path}",
                )
                self.assertEqual(b.section_path[-1], "摘要")
                break

    def test_heading_levels_are_unique_per_size(self) -> None:
        """真实 PDF 的 7 档字号必须得到 7 个不同 level (不被压平到 6)。"""
        if not self._PDF.exists():
            self.skipTest(f"真实 PDF 不存在, 跳过: {self._PDF}")

        try:
            import pdfplumber
        except ImportError:  # pragma: no cover
            self.skipTest("pdfplumber 未安装")

        from ingest.parsers.pdf_common import (
            _body_font_size,
            _heading_size_levels,
        )
        from ingest.parsers.pdf_pdfplumber import PdfplumberSource

        with pdfplumber.open(str(self._PDF)) as pdf:
            src = PdfplumberSource(pdf)
            levels = _heading_size_levels(src, _body_font_size(src))

        self.assertGreaterEqual(len(levels), 2, "应识别出多档标题字号")
        self.assertEqual(
            len(set(levels.values())), len(levels),
            f"不同字号共用了同一 level (压平 bug): {levels}",
        )


if __name__ == "__main__":
    unittest.main()
