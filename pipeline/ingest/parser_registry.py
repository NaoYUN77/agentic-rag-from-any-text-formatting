r"""Parser registry。"""

from __future__ import annotations

from typing import Callable, Dict

from .models import DocumentArtifact, SourcePayload
from .parsers.docling_optional import parse_docling
from .parsers.html import parse_html
from .parsers.markdown import parse_markdown_text
from .parsers.office_optional import parse_office
from .parsers.pdf_mineru import parse_pdf_mineru
from .parsers.pdf_pymupdf import parse_pdf_pymupdf
from .parsers.plain import parse_plain


ParserFn = Callable[..., DocumentArtifact]


def markdown_parser(source: SourcePayload) -> DocumentArtifact:
    return parse_markdown_text(
        source.data.decode(source.encoding or "utf-8", errors="replace"),
        source,
        parser="markdown",
    )


PARSERS: Dict[str, ParserFn] = {
    "markdown": markdown_parser,
    "plain_text": parse_plain,
    "html_trafilatura": parse_html,
    "html_readability": parse_html,
    "pdf_mineru": parse_pdf_mineru,
    "pdf_docling": parse_docling,
    "pdf_pymupdf": parse_pdf_pymupdf,
    "office_optional": parse_office,
}


def get_parser(name: str) -> ParserFn:
    try:
        return PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"未知 parser: {name}") from exc
