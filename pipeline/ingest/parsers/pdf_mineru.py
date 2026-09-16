r"""PDF -> Markdown via MinerU。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from pdf_loader import clean, extract

from ..models import DocumentArtifact, SourcePayload
from .markdown import parse_markdown_text


def _source_pdf_path(source: SourcePayload) -> tuple[str, Optional[str]]:
    if source.local_path:
        return source.local_path, None
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(source.data)
    temp.close()
    return temp.name, temp.name


def parse_pdf_mineru(
    source: SourcePayload,
    pages: Optional[str] = None,
    language: str = "ch",
) -> DocumentArtifact:
    pdf_path, temp_path = _source_pdf_path(source)
    try:
        markdown = extract(pdf_path, pages=pages, language=language)
        markdown = clean(markdown)
        artifact = parse_markdown_text(markdown, source, parser="mineru")
        artifact.metadata.update({
            "pdf_pages_requested": pages,
            "pdf_language": language,
        })
        return artifact
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
