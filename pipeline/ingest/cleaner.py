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

# ---- 目录（TOC）识别 ----
#
# 目录是纯导航内容：真正的正文在它引用的那些章节里，而章节本身也会被索引。
# 把目录留在索引里只会制造假命中 —— 查「KEEPALIVED 概述」会命中目录条目
# 而不是正文。实测这份 PDF 的目录块 5300 字符，会参与召回。见 issues/12。
#
# 判据：多级编号 + 标题 + 结尾页码。
#   「1.1. KEEPALIVED 3」「A.1. 先决条件 38」
# 要求 `(?:\.[0-9]+)+` 至少一层，是为了**避免把有序列表误判成目录** ——
# 「1. 第一步」这种单级编号不带点分层级，不匹配。
#
# 实测（Red Hat 负载均衡手册前 20 页）：目录块命中率 0.79（31/39 行），
# 其余 125 个 block **全部 < 0.2** —— 阈值 0.5 有 2.5 倍余量，零误报。
_TOC_ENTRY = re.compile(r"^\s*[0-9A-Z]+(?:\.[0-9]+)+\.?\s+\S.*?\s+\d+\s*$")
_TOC_MIN_LINES = 5
_TOC_RATIO = 0.5


def is_toc_block(block: DocumentBlock) -> bool:
    """判断一个 block 是不是「目录条目列表」。

    只看 text/list —— heading 是单行标题，code/table 是原子块，都不可能是目录。
    """
    if block.type not in {"text", "list"}:
        return False
    lines = [ln for ln in block.text.split("\n") if ln.strip()]
    if len(lines) < _TOC_MIN_LINES:
        return False
    hits = sum(1 for ln in lines if _TOC_ENTRY.match(ln))
    return hits / len(lines) >= _TOC_RATIO


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
        # 目录在清洗【之前】判 —— _clean_text 会折叠空白，可能改变行形态。
        if is_toc_block(block):
            continue
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
    # 丢内容必须留痕 —— 否则以后没人知道少了什么、为什么少。
    dropped_toc = [b.block_id for b in artifact.blocks if is_toc_block(b)]
    artifact.blocks = clean_blocks(artifact.blocks)
    if dropped_toc:
        artifact.warnings.append(
            "dropped_toc_blocks: " + ",".join(dropped_toc)
        )
    return artifact
