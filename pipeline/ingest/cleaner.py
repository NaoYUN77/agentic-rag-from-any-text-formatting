r"""Block 级清洗。"""

from __future__ import annotations

import re
from typing import List

from .models import DocumentArtifact, DocumentBlock


_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BOILERPLATE = re.compile(
    r"(newsletter|subscribe|sign up|delivered monthly|cookie policy|"
    r"privacy policy|follow us|all rights reserved)",
    re.I,
)


def _clean_text(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _HTML_COMMENT.sub("", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def clean_blocks(blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    cleaned: List[DocumentBlock] = []
    previous_signature = None
    for block in blocks:
        if block.type == "code":
            block.text = block.text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
            block.markdown = block.markdown or f"```{block.code_language or ''}\n{block.text}\n```"
        elif block.type == "formula":
            block.latex = (block.latex or "").strip()
            block.text = block.text.strip()
            block.markdown = block.markdown or f"$${block.latex}$$"
        else:
            block.text = _clean_text(block.text)
            block.markdown = _clean_text(block.markdown or block.text)

        if not block.text and not block.image_uri:
            continue
        if (
            block.type in {"text", "heading", "list"}
            and len(block.text) < 300
            and _BOILERPLATE.search(block.text)
        ):
            continue
        signature = (block.type, block.text)
        if block.type in {"text", "heading"} and signature == previous_signature:
            continue
        cleaned.append(block)
        previous_signature = signature

    for order, block in enumerate(cleaned, 1):
        block.order = order
    return cleaned


def clean_artifact(artifact: DocumentArtifact) -> DocumentArtifact:
    artifact.blocks = clean_blocks(artifact.blocks)
    return artifact
