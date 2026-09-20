r"""Parser registry。"""

from __future__ import annotations

from typing import Callable, Dict

from .models import DocumentArtifact, SourcePayload
from .parsers.docling_optional import parse_docling
from .parsers.html import parse_html
from .parsers.markdown import parse_markdown_text
from .parsers.office_optional import parse_office
from .parsers.pdf_pdfplumber import parse_pdf_pdfplumber
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
    # 两个 html 名字都指向同一个实现: parse_html 现在从 DOM 直产 Block,
    # 不再经 Markdown, 也不再使用 trafilatura 的内容抽取（只借它的元数据）。
    # 保留 html_trafilatura 这个名字仅为兼容历史产物里的 parser 字段。
    "html_readability": parse_html,
    "html_trafilatura": parse_html,
    "pdf_pdfplumber": parse_pdf_pdfplumber,
    "pdf_pymupdf": parse_pdf_pymupdf,
    "pdf_docling": parse_docling,
    "office_optional": parse_office,
}


def get_parser(name: str) -> ParserFn:
    try:
        return PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"未知 parser: {name}") from exc
