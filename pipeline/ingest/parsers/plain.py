r"""纯文本 -> DocumentArtifact, 直接构造 Block。

为什么不再复用 Markdown 解析器
-----------------------------
原实现是 `parse_plain -> parse_markdown_text`，等于**把纯文本当 Markdown 解析**。
后果: 一份 .txt 里只要有一行以 `#` 开头（比如 shell 注释、配置示例、
甚至一句普通的井号开头的话），就会被当成 heading —— 凭空造出层级结构。

纯文本没有标记语法，所以这里只做两件事:
    1. 按空行切段落
    2. 认 RST 式下划线标题（=== / --- 紧跟一行文字）

第 2 条是必要的: 路由表把 .rst 也送到这里，而下划线标题是 RST 的
**真实语法**，不是猜测 —— 与"把 # 当标题"性质不同。
"""
from __future__ import annotations

import re
from typing import List

from ..models import DocumentArtifact, DocumentBlock, SourcePayload, stable_artifact_id
from .markdown import infer_language


# RST 下划线标题: 一行文字, 下一行是同长的 = - ~ ^ " ' ` # * + 之一
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+_:.])\1{2,}\s*$")

# 下划线字符 -> 层级（RST 里按出现顺序决定, 这里给常用映射）
_UNDERLINE_LEVEL = {
    "=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "'": 6,
    "`": 6, "#": 6, "*": 6, "+": 6, ":": 6, ".": 6, "_": 6,
}


def parse_plain(source: SourcePayload) -> DocumentArtifact:
    """纯文本 -> DocumentArtifact（不解释标记语法, 除 RST 下划线标题）。"""
    encoding = source.encoding or "utf-8"
    text = source.data.decode(encoding, errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    blocks: List[DocumentBlock] = []
    heading_stack: List[str] = []
    order = 0
    para: List[str] = []
    url = source.final_uri or None

    def flush() -> None:
        nonlocal order, para
        if not para:
            return
        body = "\n".join(para).strip()
        para = []
        if not body:
            return
        order += 1
        # 纯文本里的列表只能靠行首标记认 —— 这是排版事实, 不是语法解释
        is_list = bool(re.match(r"^\s*(?:[-*+•·]|\(?\d+[.)])\s+\S", body))
        blocks.append(DocumentBlock(
            block_id=f"b{order:04d}",
            type="list" if is_list else "text",
            text=body,
            markdown=body,
            section_path=list(heading_stack),
            order=order,
            url=url,
            metadata={"source_format": "plain_text"},
        ))

    i = 0
    while i < len(lines):
        line = lines[i]

        # RST 下划线标题: 当前行有文字, 下一行是下划线。
        # RST 的规则是下划线**不短于**标题（可以更长），不是"长度相近"。
        if (line.strip() and i + 1 < len(lines)
                and _RST_UNDERLINE.match(lines[i + 1].strip())
                and len(lines[i + 1].strip()) >= len(line.strip())):
            flush()
            level = _UNDERLINE_LEVEL.get(lines[i + 1].strip()[0], 6)
            title = line.strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            order += 1
            blocks.append(DocumentBlock(
                block_id=f"b{order:04d}",
                type="heading",
                text=title,
                markdown=title,
                section_path=list(heading_stack),
                order=order,
                url=url,
                metadata={"level": level, "source_format": "plain_text",
                          "from_rst_underline": True},
            ))
            heading_stack.append(title)
            i += 2
            continue

        if not line.strip():
            flush()
        else:
            para.append(line)
        i += 1

    flush()

    title = next((b.text for b in blocks if b.type == "heading"
                  and b.metadata.get("level") == 1), None)
    if not title:
        title = next((b.text for b in blocks if b.type == "heading"), None)
    if not title and blocks:
        # 纯文本没有标题时, 用第一段的首行当标题
        first = blocks[0].text.strip().split("\n", 1)[0]
        title = first[:80] or None

    return DocumentArtifact(
        schema_version="1.0",
        artifact_id=stable_artifact_id(source),
        source_uri=source.final_uri,
        source_type=source.source_type,
        mime_type=source.mime_type or "text/plain",
        parser="plain_text",
        parser_version="plain-v1",
        title=title,
        language=infer_language(text[:20000]),
        blocks=blocks,
        metadata={
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
            "fetched_at": source.fetched_at,
            "url": url,
        },
    )
