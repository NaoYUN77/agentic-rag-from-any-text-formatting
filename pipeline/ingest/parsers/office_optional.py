r"""Office optional adapter。"""

from __future__ import annotations

from ..models import DocumentArtifact, SourcePayload


def parse_office(source: SourcePayload) -> DocumentArtifact:
    try:
        from markitdown import MarkItDown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "未安装 MarkItDown。安装后再启用 Office 解析: pip install markitdown"
        ) from exc
    raise NotImplementedError("Office adapter 尚未接入")
