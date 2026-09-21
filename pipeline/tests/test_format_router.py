from __future__ import annotations

import unittest

from ingest.models import DocumentArtifact, DocumentBlock, IndexDecision, QualityReport, SourcePayload
from ingest.parsers.html import parse_html
from ingest.parsers.markdown import parse_markdown_text
from ingest.chunker import BlockAwareHierarchicalChunkBuilder
from ingest.quality import assess_raw_quality, decide_index
from ingest.router import FormatRouter
from generation import build_context


def source(data: bytes, mime: str, uri: str = "test") -> SourcePayload:
    return SourcePayload(
        source_type="local_file",
        uri=uri,
        final_uri=uri,
        mime_type=mime,
        content_type=mime,
        data=data,
        size_bytes=len(data),
        sha256="x",
    )


class FormatRouterTests(unittest.TestCase):
    def test_pdf_magic(self) -> None:
        route = FormatRouter().route(source(b"%PDF-1.4\n", "application/octet-stream"))
        # 默认用 pdfplumber(MIT): 能拿页码/bbox/字号/原生大纲, 且无 copyleft 义务。
        # 回退只剩 PyMuPDF —— pdf_mineru 已于 2026-09-20 移除
        # (唯一经 Markdown 中转的 PDF 路径, 会丢页码/坐标/字号)。
        self.assertEqual(route.parser, "pdf_pdfplumber")
        self.assertEqual(route.fallbacks, ["pdf_pymupdf"])

    def test_no_markdown_transit_parser_registered(self) -> None:
        """架构不变量: 不得再有经 Markdown 中转的解析器。

        历史上 html / plain / pdf_mineru 都走 `HTML|PDF -> Markdown -> Block`,
        前两者已改为直产 Block, pdf_mineru 已移除。
        这里锁住该不变量, 防止回退。
        """
        from ingest.parser_registry import PARSERS

        self.assertNotIn("pdf_mineru", PARSERS)
        # 唯一允许经 parse_markdown_text 的是 markdown 本身(原生格式)
        self.assertEqual(PARSERS["markdown"].__module__, "ingest.parser_registry")

    def test_html_code_block(self) -> None:
        html = b"""
        <html><head><title>Demo</title></head><body>
        <article><h1>Demo</h1><p>Hello world.</p>
        <pre><code class="language-python">print('hello')</code></pre>
        </article></body></html>
        """
        artifact = parse_html(source(html, "text/html"))
        self.assertTrue(any(b.type == "code" and "print" in b.text for b in artifact.blocks))

    def test_markdown_heading_and_formula(self) -> None:
        md = b"# Title\n\nText here.\n\n$$E = mc^2$$\n"
        artifact = parse_markdown_text(md.decode(), source(md, "text/markdown"))
        self.assertEqual(artifact.title, "Title")
        self.assertTrue(any(b.type == "formula" for b in artifact.blocks))

    def test_empty_artifact_rejected(self) -> None:
        artifact = DocumentArtifact(
            schema_version="1.0",
            artifact_id="doc_empty",
            source_uri="test",
            source_type="local_file",
            mime_type="text/plain",
            parser="plain_text",
        )
        artifact.quality = assess_raw_quality(artifact)
        decision = decide_index(artifact)
        self.assertEqual(artifact.quality.status, "reject")
        self.assertFalse(decision.dense_index)
        self.assertFalse(decision.sparse_index)

    def test_block_aware_chunker_keeps_code_and_fragments(self) -> None:
        md = (
            "# Guide\n\n"
            "## 1. Proxy\n\n"
            "Proxy passes requests to the backend.\n\n"
            "```nginx\nlocation / { proxy_pass http://backend; }\n```\n"
        )
        artifact = parse_markdown_text(md, source(md.encode(), "text/markdown"))
        artifact.quality = assess_raw_quality(artifact)
        artifact.index_decision = decide_index(artifact)
        parents, chunks = BlockAwareHierarchicalChunkBuilder().build(artifact)
        self.assertEqual(len(parents), 1)
        self.assertTrue(chunks)
        self.assertTrue(any(f.block_type == "code" for c in chunks for f in c.fragments))
        self.assertTrue(all(c.full_text for c in chunks))

    def test_lv1_heading_groups_with_its_own_content(self) -> None:
        """回归: lv1 标题不能与自己的正文块分到不同 parent。

        标题块的 section_path 是【祖先链, 不含自己】, 所以 lv1 标题的
        section_path 为空。若 _scope 直接返回 fallback(文档标题), 该标题
        会与自己的正文块分到不同分组, 实测会导致多个 lv1 标题塌缩成一个
        "章节目录" chunk(只有标题、无正文、sparse_text 为空)。见 issues/10。
        """
        md = (
            "# 第 1 章 概述\n\n"
            "第一章的正文内容。\n\n"
            "# 第 2 章 配置\n\n"
            "第二章的正文内容。\n"
        )
        artifact = parse_markdown_text(md, source(md.encode(), "text/markdown"))
        parents, chunks = BlockAwareHierarchicalChunkBuilder().build(artifact)

        # 两个 lv1 标题各自成组, 不与彼此合并
        self.assertEqual(len(parents), 2)
        # 每个 parent 都要有正文, 不能是"只有标题"的空壳
        for parent in parents:
            body = [
                c for c in chunks
                if c.parent_id == parent.parent_id
                and any(f.block_type != "heading" for f in c.fragments)
            ]
            self.assertTrue(body, f"{parent.parent_id} 没有正文块")

    def test_nested_sections_do_not_collapse_into_one_parent(self) -> None:
        """回归: 顶层只有一个标题时, 子章节不能被合并成"整篇一个 parent"。

        旧 `_scope()` 只取 `section_path[0]`。实测 `openai_scaling_storage`
        有 9 个章节, 但它们全都挂在同一个 lv1 标题下 -> `section_path[0]`
        恒为文档标题 -> 整篇只剩 1 个 parent, parent expansion 失去意义。

        改用完整 `section_path` 后, 每个子章节应各自成组。见 issues/10。
        """
        md = (
            "# 文档标题\n\n"
            "## 第一节\n\n"
            "第一节的正文内容。\n\n"
            "## 第二节\n\n"
            "第二节的正文内容。\n\n"
            "## 第三节\n\n"
            "第三节的正文内容。\n"
        )
        artifact = parse_markdown_text(md, source(md.encode(), "text/markdown"))
        parents, chunks = BlockAwareHierarchicalChunkBuilder().build(artifact)

        # 3 个子章节 -> 至少 3 个分组（旧实现这里只有 1 个）
        self.assertGreaterEqual(len(parents), 3)

        # 且不能有哪个 parent 吞掉整篇正文
        body = [
            c for c in chunks
            if any(f.block_type != "heading" for f in c.fragments)
        ]
        self.assertGreater(len(body), 1)
        for parent in parents:
            mine = [c for c in body if c.parent_id == parent.parent_id]
            self.assertLess(
                len(mine), len(body),
                f"{parent.parent_id} 吞掉了整篇正文, 分组退化",
            )

    def test_block_aware_chunker_respects_budget_without_overlap(self) -> None:
        """固定大小切块: 尊重 token 预算, 且块与块之间**不重复**源文本。

        overlap 已于 2026-09-21 取消。旧实现会把上一块尾部带回新块,
        因此下面对"片段区间互不重叠"的断言在旧实现下会失败 ——
        它是这次改动的回归护栏。
        """
        text = " ".join(
            f"Sentence {i} explains proxy routing and backend health checks."
            for i in range(1, 80)
        )
        blocks = [
            DocumentBlock(
                block_id="h1",
                type="heading",
                text="Proxy",
                section_path=["Proxy"],
                order=0,
                quality_score=1.0,
            ),
            DocumentBlock(
                block_id="b1",
                type="text",
                text=text,
                section_path=["Proxy"],
                order=1,
                quality_score=1.0,
            ),
        ]
        artifact = DocumentArtifact(
            schema_version="1.0",
            artifact_id="doc_chunk_budget",
            source_uri="test",
            source_type="local_file",
            mime_type="text/plain",
            parser="plain_text",
            blocks=blocks,
            quality=QualityReport(score=1.0, status="high"),
            index_decision=IndexDecision("high", True, True),
        )
        _, chunks = BlockAwareHierarchicalChunkBuilder(
            chunk_tokens=80,
        ).build(artifact)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.token_count <= 80 for c in chunks))
        self.assertTrue(all(chunk.fragments for chunk in chunks))
        for chunk in chunks:
            for fragment in chunk.fragments:
                source_block = next(b for b in blocks if b.block_id == fragment.block_id)
                self.assertEqual(
                    fragment.text,
                    source_block.text[fragment.start_char:fragment.end_char],
                )
        self.assertTrue(all("Proxy" not in chunk.sparse_text for chunk in chunks))

        # overlap 取消后的核心不变量: 正文块的字符区间互不重叠（无重复覆盖）。
        spans = sorted(
            (f.start_char, f.end_char)
            for chunk in chunks
            for f in chunk.fragments
            if f.block_id == "b1"
        )
        self.assertGreater(len(spans), 1)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(
                prev_end, next_start,
                "片段区间重叠 -> 说明仍存在 overlap",
            )
        # 且没有丢内容: 切块应覆盖正文块的绝大部分字符。
        # 不要求 100% —— SentenceSplitter 会吃掉句间空白, 实测缺口约 0.3%。
        covered = sum(e - s for s, e in spans)
        self.assertGreaterEqual(covered, len(text) * 0.99)
        self.assertLessEqual(covered, len(text))


class CitationContextTests(unittest.TestCase):
    """引用展示: metadata 里的页码与文档信息要能还原成出处。"""

    def _hit(self, **over):
        payload = {
            "text": "它降低了 49% 的检索失败率。",
            "window_text": "使用两种技术。它降低了 49% 的检索失败率。结合 rerank 降低 67%。",
            "title": "负载平衡器管理",
            "file_name": "redhat.pdf",
            "source_uri": "https://example.com/x.pdf",
            "section_path": ["第 2 章 概述", "2.3. 计划概述"],
            "page_start": 11,
            "page_end": 11,
            "page": 11,
            "bbox": [52.2, 626.5, 535.8, 682.2],
            "unit_kind": "sentence",
            "chunk_id": "doc_x_s0201",
            "artifact_id": "doc_x",
            "chunk_index": 200,
        }
        payload.update(over)
        from hybrid_retriever import RetrievalHit
        return RetrievalHit(point_id=1, payload=payload)

    def test_header_carries_document_and_page(self) -> None:
        context, citations = build_context([self._hit()])
        self.assertIn("[C1]", context)
        self.assertIn("负载平衡器管理", context)
        self.assertIn("第 11 页", context)
        self.assertIn("第 2 章 概述 > 2.3. 计划概述", context)
        self.assertEqual(len(citations), 1)

    def test_locator_is_human_readable(self) -> None:
        _, citations = build_context([self._hit()])
        loc = citations[0].locator()
        self.assertIn("负载平衡器管理", loc)
        self.assertIn("第 11 页", loc)
        self.assertIn("2.3. 计划概述", loc)

    def test_page_range_rendered(self) -> None:
        _, citations = build_context([self._hit(page_start=7, page_end=8)])
        self.assertIn("第 7-8 页", citations[0].locator())

    def test_auto_prefers_window_so_single_sentence_has_context(self) -> None:
        """单句策略下, 正文只有一句话(可能含指代), 生成需用窗口补上下文。"""
        context, _ = build_context([self._hit()], context_source="auto")
        self.assertIn("使用两种技术", context)

    def test_text_mode_uses_sentence_only(self) -> None:
        context, _ = build_context([self._hit()], context_source="text")
        self.assertNotIn("使用两种技术", context)
        self.assertIn("它降低了 49%", context)


if __name__ == "__main__":
    unittest.main()
