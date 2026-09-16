r"""Plain text -> DocumentArtifact。"""

from __future__ import annotations

from ..models import SourcePayload
from .markdown import parse_markdown_text


def parse_plain(source: SourcePayload):
    encoding = source.encoding or "utf-8"
    return parse_markdown_text(
        source.data.decode(encoding, errors="replace"),
        source,
        parser="plain_text",
    )
