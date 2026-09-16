from __future__ import annotations

import unittest

from ingest.models import DocumentArtifact, DocumentBlock, IndexDecision, QualityReport, SourcePayload
from ingest.parsers.html import parse_html
from ingest.parsers.markdown import parse_markdown_text
from ingest.chunker import BlockAwareHierarchicalChunkBuilder
from ingest.quality import assess_raw_quality, decide_index
from ingest.router import FormatRouter


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
        self.assertEqual(route.parser, "pdf_mineru")

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

    def test_block_aware_chunker_respects_budget_and_overlap(self) -> None:
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
            overlap_tokens=30,
        ).build(artifact)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(c.overlap_from_previous > 0 for c in chunks[1:]))
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


if __name__ == "__main__":
    unittest.main()
