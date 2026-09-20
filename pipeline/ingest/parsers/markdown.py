r"""Markdown / plain text -> DocumentArtifact。"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..models import DocumentArtifact, DocumentBlock, SourcePayload, stable_artifact_id


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
    heading_stack: List[Tuple[int, str]] = []
    current_parent: Optional[str] = None
    paragraph: List[str] = []
    order = 0
    # 与 html / plain 路径保持一致: block 级 url 从 final_uri 取。
    # 之前这里压根没赋 url, 导致走 markdown 回退的文档 block.url 全是 None,
    # 引用展示时无法给出"这段话出自哪个 URL"。
    url = source.final_uri or source.uri or None

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
                section_path=[t for _, t in heading_stack],
                order=order,
                url=url,
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
                section_path=[t for _, t in heading_stack],
                order=order,
                url=url,
            ))
            return

        block_type = "list" if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", raw) else "text"
        blocks.append(DocumentBlock(
            block_id=block_id,
            type=block_type,
            text=raw,
            markdown=raw,
            parent_id=current_parent,
            section_path=[t for _, t in heading_stack],
            order=order,
            url=url,
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
                section_path=[t for _, t in heading_stack],
                order=order,
                url=url,
            ))
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # 用 level 比较(不是栈深): 同级标题必须互斥。
            # 历史 bug (2026-09-20 修复, 与 html.py / pdf_common 同一类):
            # 旧写法 `while len(heading_stack) >= level` 拿"栈深度"和"层级"相比,
            # 栈长 1、level 2 时 `1 >= 2` 为假 → 同级标题被 append 成子节点。
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            order += 1
            block_id = next_id()
            blocks.append(DocumentBlock(
                block_id=block_id,
                type="heading",
                text=title,
                markdown=line,
                parent_id=None,
                section_path=[t for _, t in heading_stack[:-1]],
                order=order,
                url=url,
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
        artifact_id=stable_artifact_id(source),
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
