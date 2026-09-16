r"""PyMuPDF optional fallback adapter。"""

from __future__ import annotations

from ..models import DocumentArtifact, SourcePayload


def parse_pdf_pymupdf(source: SourcePayload) -> DocumentArtifact:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未安装 PyMuPDF, 无法使用 pdf_pymupdf fallback") from exc
    raise NotImplementedError("pdf_pymupdf adapter 尚未接入")
