r"""HTML / plain_text 直产 Block 的回归测试。

这两条路径原本都经 `parse_markdown_text` 中转, 会丢结构或凭空造结构。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parent.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ingest.models import SourcePayload  # noqa: E402
from ingest.parsers.html import parse_html  # noqa: E402
from ingest.parsers.markdown import parse_markdown_text  # noqa: E402
from ingest.parsers.plain import parse_plain  # noqa: E402


def html_source(html: str, url: str = "https://example.com/a") -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=url, final_uri=url,
        mime_type="text/html", content_type="text/html",
        data=html.encode("utf-8"), encoding="utf-8",
    )


def md_source(md: str, url: str = "https://example.com/a.md") -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=url, final_uri=url,
        mime_type="text/markdown", content_type="text/markdown",
        data=md.encode("utf-8"), encoding="utf-8",
    )


def plain_source(text: str, url: str = "https://example.com/a.txt") -> SourcePayload:
    return SourcePayload(
        source_type="url", uri=url, final_uri=url,
        mime_type="text/plain", content_type="text/plain",
        data=text.encode("utf-8"), encoding="utf-8",
    )


class BlockUrlPropagationTests(unittest.TestCase):
    """每个 parser 都必须把 source 的 url 写到 block 上。

    背景: 最初只有 html 路径写了 url, markdown 路径 5 个构造点全漏,
    pdf 路径 2 个全漏 —— 导致走这些路径的文档 block.url 全是 None,
    引用展示时给不出"这段出自哪个 URL"。

    用 AST 静态检查兜底: 新增构造点若忘了 url, 这里就会失败。
    """

    _PARSERS = ["html", "plain", "markdown", "pdf_common"]

    def test_every_block_constructor_passes_url(self) -> None:
        import ast

        parsers_dir = _PIPELINE / "ingest" / "parsers"
        missing = []
        for name in self._PARSERS:
            tree = ast.parse((parsers_dir / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "DocumentBlock"
                        and "url" not in {k.arg for k in node.keywords}):
                    missing.append(f"{name}.py:{node.lineno}")
        self.assertEqual(
            missing, [],
            "以下 DocumentBlock 构造点未传 url=, block 级 URL 会丢: " + ", ".join(missing),
        )

    def test_html_blocks_carry_url(self) -> None:
        art = parse_html(html_source(SAMPLE, url="https://example.com/page"))
        self.assertTrue(art.blocks)
        for b in art.blocks:
            self.assertEqual(b.url, "https://example.com/page", b.block_id)

    def test_markdown_blocks_carry_url(self) -> None:
        md = "# 标题\n\n正文一段。\n\n```python\nx = 1\n```\n\n- 项一\n- 项二\n"
        art = parse_markdown_text(md, md_source(md, url="https://example.com/doc"))
        self.assertTrue(art.blocks)
        for b in art.blocks:
            self.assertEqual(b.url, "https://example.com/doc", b.block_id)

    def test_plain_blocks_carry_url(self) -> None:
        art = parse_plain(plain_source("一段正文。", url="https://example.com/t.txt"))
        self.assertTrue(art.blocks)
        for b in art.blocks:
            self.assertEqual(b.url, "https://example.com/t.txt", b.block_id)


SAMPLE = """<!DOCTYPE html><html><head><title>示例页</title></head><body><article>
<h1>主标题</h1>
<p>第一段，含一个<a href="https://ref.example/x">参考链接</a>。</p>
<h2>小节一</h2>
<ul><li>项一</li><li>项二</li></ul>
<pre><code class="language-python">def f():
    return 1</code></pre>
<h3>更深的节</h3>
<p>第三段。</p>
<figure><img src="https://img.example/1.png" alt="图示">
<figcaption>图 1 的说明</figcaption></figure>
</article></body></html>"""


class HtmlDomBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.art = parse_html(html_source(SAMPLE))

    def test_produces_blocks_without_markdown(self) -> None:
        self.assertTrue(self.art.blocks)
        self.assertEqual(self.art.parser, "html_readability")

    def test_url_is_set_on_every_block(self) -> None:
        """新增的 url 字段: 每个 block 都要带来源 URL。"""
        self.assertTrue(self.art.blocks)
        for b in self.art.blocks:
            self.assertEqual(b.url, "https://example.com/a")

    def test_heading_levels_from_tags(self) -> None:
        heads = {b.text: b.metadata["level"]
                 for b in self.art.blocks if b.type == "heading"}
        self.assertEqual(heads.get("主标题"), 1)
        self.assertEqual(heads.get("小节一"), 2)
        self.assertEqual(heads.get("更深的节"), 3)

    def test_section_path_excludes_self(self) -> None:
        """与 markdown / pdf 路径语义一致: 标题块的 section_path 不含自己。"""
        h2 = next(b for b in self.art.blocks
                  if b.type == "heading" and b.text == "小节一")
        self.assertEqual(h2.section_path, ["主标题"])

    def test_code_language_extracted(self) -> None:
        codes = [b for b in self.art.blocks if b.type == "code"]
        self.assertTrue(codes, "应产出 code 块")
        self.assertEqual(codes[0].code_language, "python")
        self.assertIn("def f():", codes[0].text)

    def test_list_becomes_single_block(self) -> None:
        lists = [b for b in self.art.blocks if b.type == "list"]
        self.assertTrue(lists)
        self.assertIn("项一", lists[0].text)
        self.assertIn("项二", lists[0].text)
        self.assertEqual(lists[0].metadata.get("item_count"), 2)

    def test_figure_caption_preserved(self) -> None:
        """figure/figcaption 的关联在 Markdown 路径里会丢。"""
        images = [b for b in self.art.blocks if b.type == "image"]
        self.assertTrue(images)
        self.assertEqual(images[0].caption, "图 1 的说明")
        self.assertEqual(images[0].image_uri, "https://img.example/1.png")

    def test_links_collected_in_metadata(self) -> None:
        """href 在 Markdown 路径里会丢。"""
        with_links = [b for b in self.art.blocks if b.metadata.get("links")]
        self.assertTrue(with_links)
        hrefs = [l["href"] for b in with_links for l in b.metadata["links"]]
        self.assertIn("https://ref.example/x", hrefs)

    def test_title_from_metadata(self) -> None:
        self.assertEqual(self.art.title, "示例页")


class PlainTextNoMarkdownTests(unittest.TestCase):
    """回归: 纯文本不能被当 Markdown 解析。"""

    def _parse(self, text: str):
        return parse_plain(SourcePayload(
            source_type="local_file", uri="t.txt", final_uri="t.txt",
            mime_type="text/plain", content_type="text/plain",
            data=text.encode("utf-8"), encoding="utf-8",
        ))

    def test_hash_lines_are_not_headings(self) -> None:
        """核心回归: '# 这是注释' 不能被当成 heading。

        原实现复用 parse_markdown_text, 会把 shell 注释变成标题。
        """
        art = self._parse(
            "# 这是配置文件的注释\n"
            "下面是一些说明文字。\n"
            "\n"
            "# 又一行注释\n"
            "更多说明。\n"
        )
        self.assertFalse(
            any(b.type == "heading" for b in art.blocks),
            "纯文本里的 # 行不应产生 heading",
        )
        self.assertTrue(any("# 这是配置文件的注释" in b.text for b in art.blocks))

    def test_paragraphs_split_by_blank_line(self) -> None:
        art = self._parse("第一段。\n\n第二段。\n\n第三段。\n")
        self.assertEqual(len(art.blocks), 3)
        self.assertTrue(all(b.type == "text" for b in art.blocks))

    def test_rst_underline_heading_detected(self) -> None:
        """RST 的下划线标题是该格式的真实语法, 应当识别。"""
        art = self._parse(
            "文档标题\n========\n\n正文内容。\n\n小节\n----\n\n小节正文。\n"
        )
        heads = {b.text: b.metadata["level"]
                 for b in art.blocks if b.type == "heading"}
        self.assertEqual(heads.get("文档标题"), 1)
        self.assertEqual(heads.get("小节"), 2)

    def test_url_field_set(self) -> None:
        art = self._parse("一段文字。\n")
        self.assertTrue(art.blocks)
        self.assertEqual(art.blocks[0].url, "t.txt")

    def test_bullet_lines_become_list(self) -> None:
        art = self._parse("- 项一\n- 项二\n")
        self.assertEqual(art.blocks[0].type, "list")


if __name__ == "__main__":
    unittest.main()
