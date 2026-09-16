r"""HTML -> DocumentArtifact。"""

from __future__ import annotations

from typing import Optional

import trafilatura
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

from ..models import DocumentArtifact, DocumentBlock, SourcePayload
from .markdown import parse_markdown_text


def _extract_trafilatura(html: str, url: Optional[str]) -> tuple[str, dict]:
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_images=True,
        include_formatting=True,
        include_tables=True,
        favor_precision=True,
        deduplicate=True,
    ) or ""
    metadata = {}
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
        if meta:
            metadata = {
                "title": meta.title,
                "author": meta.author,
                "date": meta.date,
                "sitename": meta.sitename,
                "url": meta.url,
            }
    except Exception:
        pass
    return markdown, metadata


def _readability_fallback(html: str) -> str:
    doc = ReadabilityDocument(html)
    return doc.summary(html_partial=True) or ""


def _extract_code_blocks(html: str) -> list[tuple[Optional[str], str]]:
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[Optional[str], str]] = []
    for node in soup.select("pre code, pre"):
        text = node.get_text("\n", strip=False).strip("\n")
        if not text:
            continue
        language = None
        classes = node.get("class") or []
        for cls in classes:
            if cls.startswith("language-"):
                language = cls.split("-", 1)[1]
                break
        out.append((language, text))
    return out


def parse_html(source: SourcePayload) -> DocumentArtifact:
    encoding = source.encoding or "utf-8"
    html = source.data.decode(encoding, errors="replace")
    markdown, metadata = _extract_trafilatura(html, source.final_uri)

    parser = "trafilatura"
    if not markdown.strip():
        fallback = _readability_fallback(html)
        if fallback:
            soup = BeautifulSoup(fallback, "lxml")
            markdown = soup.get_text("\n", strip=True)
            parser = "readability_fallback"

    artifact = parse_markdown_text(markdown, source, parser=parser)
    artifact.title = metadata.get("title") or artifact.title
    if metadata.get("author"):
        artifact.authors = [metadata["author"]]
    artifact.published_at = metadata.get("date")
    artifact.metadata.update({k: v for k, v in metadata.items() if v})

    existing_code = {b.text.strip() for b in artifact.blocks if b.type == "code"}
    for language, code in _extract_code_blocks(html):
        if code in existing_code:
            continue
        artifact.blocks.append(DocumentBlock(
            block_id=f"b{len(artifact.blocks) + 1:04d}",
            type="code",
            text=code,
            markdown=f"```{language or ''}\n{code}\n```",
            code_language=language,
            order=len(artifact.blocks) + 1,
            metadata={"source": "html_code_supplement"},
        ))
    return artifact
