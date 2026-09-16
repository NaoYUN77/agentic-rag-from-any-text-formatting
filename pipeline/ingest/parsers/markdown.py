r"""Markdown / plain text -> DocumentArtifact。"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from ..models import DocumentArtifact, DocumentBlock, SourcePayload


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^```([A-Za-z0-9_+.-]*)\s*$")
IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def infer_language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk and latin and min(cjk, latin) / max(cjk, latin, 1) > 0.15:
        return "mixed"
    if cjk > latin:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def parse_markdown_text(text: str, source: SourcePayload, parser: str = "markdown") -> DocumentArtifact:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    blocks: List[DocumentBlock] = []
    heading_stack: List[str] = []
    current_parent: Optional[str] = None
    paragraph: List[str] = []
    order = 0

    def next_id() -> str:
        return f"b{order:04d}"

    def flush_paragraph() -> None:
        nonlocal paragraph, order
        if not paragraph:
            return
        raw = "\n".join(paragraph).strip()
        paragraph = []
        if not raw:
            return
        order += 1
        block_id = next_id()

        if raw.startswith("$$") and raw.endswith("$$") and len(raw) >= 4:
            blocks.append(DocumentBlock(
                block_id=block_id,
                type="formula",
                text=raw,
                markdown=raw,
                latex=raw[2:-2].strip(),
                parent_id=current_parent,
                section_path=list(heading_stack),
                order=order,
            ))
            return

        img = IMAGE_RE.match(raw)
        if img:
            blocks.append(DocumentBlock(
                block_id=block_id,
                type="image",
                text=img.group(1),
                markdown=raw,
                caption=img.group(1) or None,
                image_uri=img.group(2),
                parent_id=current_parent,
                section_path=list(heading_stack),
                order=order,
            ))
            return

        block_type = "list" if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", raw) else "text"
        blocks.append(DocumentBlock(
            block_id=block_id,
            type=block_type,
            text=raw,
            markdown=raw,
            parent_id=current_parent,
            section_path=list(heading_stack),
            order=order,
        ))

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            flush_paragraph()
            fence = FENCE_RE.match(line.strip())
            language = fence.group(1) if fence else None
            code_lines: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            order += 1
            code = "\n".join(code_lines)
            blocks.append(DocumentBlock(
                block_id=next_id(),
                type="code",
                text=code,
                markdown=f"```{language or ''}\n{code}\n```",
                code_language=language,
                parent_id=current_parent,
                section_path=list(heading_stack),
                order=order,
            ))
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            order += 1
            block_id = next_id()
            blocks.append(DocumentBlock(
                block_id=block_id,
                type="heading",
                text=title,
                markdown=line,
                parent_id=None,
                section_path=list(heading_stack[:-1]),
                order=order,
                metadata={"level": level},
            ))
            current_parent = block_id
            i += 1
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        i += 1

    flush_paragraph()
    title = next((b.text for b in blocks if b.type == "heading" and b.metadata.get("level") == 1), None)
    if not title:
        title = next((b.text for b in blocks if b.type == "heading"), None)
    artifact = DocumentArtifact(
        schema_version="1.0",
        artifact_id=f"doc_{uuid.uuid4().hex[:12]}",
        source_uri=source.final_uri,
        source_type=source.source_type,
        mime_type=source.mime_type,
        parser=parser,
        parser_version="internal-v1",
        title=title,
        language=infer_language(text),
        blocks=blocks,
        metadata={
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
            "fetched_at": source.fetched_at,
        },
    )
    return artifact
