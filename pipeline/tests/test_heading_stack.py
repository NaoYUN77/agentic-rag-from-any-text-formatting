r"""heading_stack 的层级正确性 —— 跨 parser 回归测试。

背景
----
同一个 bug 曾存在于**三个** parser, 因为修得晚, 只有 pdf 先修好了:

    pdf_common.py   ✅ 2026-09-19 修
    html.py         ❌ 2026-09-20 修 (本文件守)
    markdown.py     ❌ 2026-09-20 修 (本文件守)

bug 形态
--------
弹栈条件写成 `while len(heading_stack) >= level`, 把**栈深度**和
**heading 层级**两种量纲相比。栈长 1、level 2 时 `1 >= 2` 为假 → 不弹栈
→ **新的同级标题被 append 成前一个同级标题的子节点**。

实测后果 (10 篇语料): parents 16 → 54, 不同顶层 scope 9 → 41。
即 issues/10「parent 分组退化」的上游成因之一。

注意
----
本测试用例**必须包含"同级标题连续出现"**, 否则新旧实现结果相同, 没有区分度
(这一点在 pdf 那轮已经踩过: 第一版测试注入旧 bug 后仍全过)。
"""

from __future__ import annotations

import hashlib
import unittest

from ingest.models import SourcePayload
from ingest.parsers.html import parse_html
from ingest.parsers.markdown import parse_markdown_text


def _payload(data: bytes, mime: str) -> SourcePayload:
    return SourcePayload(
        source_type="url", uri="https://example.com/doc", final_uri="https://example.com/doc",
        mime_type=mime, content_type=mime,
        data=data, encoding="utf-8",
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def _paths(artifact) -> dict:
    """{标题文本: section_path} —— 只取 heading 块。"""
    return {b.text: list(b.section_path) for b in artifact.blocks if b.type == "heading"}


class HtmlHeadingStackTests(unittest.TestCase):
    """HTML 路径: 同级标题必须是兄弟。"""

    _HTML = b"""<html><head><title>T</title></head><body><article>
    <h2>The anatomy of a skill</h2>
    <p>Intro paragraph one.</p>
    <h3>Skills and the context window</h3>
    <p>Body A.</p>
    <h3>Skills and code execution</h3>
    <p>Body B.</p>
    <h2>Developing and evaluating skills</h2>
    <p>Body C.</p>
    <h2>Acknowledgements</h2>
    <p>Body D.</p>
    </article></body></html>"""

    def test_same_level_headings_are_siblings(self) -> None:
        paths = _paths(parse_html(_payload(self._HTML, "text/html")))
        self.assertEqual(paths["The anatomy of a skill"], [])
        # 两个 h3 必须是兄弟, 都在同一个 h2 下 —— 不能互相嵌套
        self.assertEqual(paths["Skills and the context window"],
                         ["The anatomy of a skill"])
        self.assertEqual(
            paths["Skills and code execution"],
            ["The anatomy of a skill"],
            "同级 h3 不应嵌套在另一个 h3 下",
        )
        # 后续 h2 必须回到顶层, 不能挂在第一个 h2 下
        for t in ("Developing and evaluating skills", "Acknowledgements"):
            self.assertEqual(paths[t], [], f"同级 h2 「{t}」不应嵌套在其它 h2 下")

    def test_deeper_heading_still_nests(self) -> None:
        """修 bug 不能把"真子级"也压平。"""
        paths = _paths(parse_html(_payload(self._HTML, "text/html")))
        self.assertEqual(
            paths["Skills and the context window"],
            ["The anatomy of a skill"],
            "h3 应仍是 h2 的子级",
        )


class MarkdownHeadingStackTests(unittest.TestCase):
    """Markdown 路径: 同一套语义。"""

    _MD = (b"## The anatomy of a skill\n\nIntro.\n\n"
           b"### Skills and the context window\n\nBody A.\n\n"
           b"### Skills and code execution\n\nBody B.\n\n"
           b"## Developing and evaluating skills\n\nBody C.\n\n"
           b"## Acknowledgements\n\nBody D.\n")

    def test_same_level_headings_are_siblings(self) -> None:
        art = parse_markdown_text(self._MD.decode(), _payload(self._MD, "text/markdown"))
        paths = _paths(art)
        self.assertEqual(paths["The anatomy of a skill"], [])
        self.assertEqual(paths["Skills and the context window"],
                         ["The anatomy of a skill"])
        self.assertEqual(paths["Skills and code execution"],
                         ["The anatomy of a skill"],
                         "同级 ### 不应嵌套在另一个 ### 下")
        for t in ("Developing and evaluating skills", "Acknowledgements"):
            self.assertEqual(paths[t], [], f"同级 ## 「{t}」不应嵌套在其它 ## 下")

    def test_body_blocks_inherit_correct_section(self) -> None:
        """正文块的 section_path 也要跟着正确。

        注意语义: 正文块位于某标题**之后**, 所以应继承那个标题;
        而标题块自身的 section_path **不含自己**。
        """
        art = parse_markdown_text(self._MD.decode(), _payload(self._MD, "text/markdown"))
        cases = {
            "Body A.": ["The anatomy of a skill", "Skills and the context window"],
            "Body B.": ["The anatomy of a skill", "Skills and code execution"],
            "Body C.": ["Developing and evaluating skills"],
            "Body D.": ["Acknowledgements"],
        }
        for prefix, expected in cases.items():
            blocks = [b for b in art.blocks if b.text.startswith(prefix)]
            self.assertTrue(blocks, f"找不到正文块 {prefix}")
            self.assertEqual(
                blocks[0].section_path, expected,
                f"{prefix} 的 section_path 应为 {expected}, 实际 {blocks[0].section_path}",
            )


class CrossParserConsistencyTests(unittest.TestCase):
    """三个 parser 的 heading 语义必须一致。"""

    def test_html_and_markdown_agree_on_nesting(self) -> None:
        html = HtmlHeadingStackTests._HTML
        md = MarkdownHeadingStackTests._MD
        hp = _paths(parse_html(_payload(html, "text/html")))
        mp = _paths(parse_markdown_text(md.decode(), _payload(md, "text/markdown")))
        for title in ("The anatomy of a skill", "Skills and the context window",
                      "Skills and code execution", "Developing and evaluating skills",
                      "Acknowledgements"):
            self.assertIn(title, hp, f"HTML 缺少标题 {title}")
            self.assertIn(title, mp, f"Markdown 缺少标题 {title}")
            self.assertEqual(hp[title], mp[title], f"「{title}」的层级在两路不一致")


if __name__ == "__main__":
    unittest.main()
