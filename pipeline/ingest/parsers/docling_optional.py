r"""Docling optional adapter。"""

from __future__ import annotations

from ..models import DocumentArtifact, SourcePayload


def parse_docling(source: SourcePayload) -> DocumentArtifact:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "未安装 Docling。安装后再启用英文/数学 PDF 解析: pip install docling"
        ) from exc
    raise NotImplementedError("Docling 已检测到，但 adapter 尚未接入")
