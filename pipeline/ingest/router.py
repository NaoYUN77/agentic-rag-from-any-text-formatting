r"""格式路由: SourcePayload -> RouteDecision。"""

from __future__ import annotations

from .models import RouteDecision, SourcePayload


PDF_MIME = "application/pdf"
HTML_MIMES = {"text/html", "application/xhtml+xml"}
MARKDOWN_MIMES = {"text/markdown", "text/x-markdown"}
TEXT_MIMES = {"text/plain"}
OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class FormatRouter:
    def route(self, source: SourcePayload) -> RouteDecision:
        mime = source.mime_type.lower()
        suffix = ""
        if source.local_path:
            suffix = source.local_path.lower().rsplit(".", 1)[-1] if "." in source.local_path else ""

        if mime == PDF_MIME or source.data[:5] == b"%PDF-":
            # 默认用 pdfplumber(MIT): 能拿到页码/bbox/字号/原生大纲,
            # 且许可证无 copyleft 义务。PyMuPDF(AGPL) 作为回退。
            #
            # 2026-09-20: 移除 pdf_mineru 回退。原因:
            #   ① 它是唯一经 `parse_markdown_text` 中转的 PDF 路径
            #      (PDF -> Markdown -> Block), 而 Markdown 表达不了页码/
            #      坐标/字号, 这些信息在进入 Block 前就永久丢失;
            #   ② 实测其产物 font_size 全为 null, heading 只剩 lv1/lv2
            #      (见 issues/11: MinerU 把 lv3/lv4 压平), 结构信息不可信;
            #   ③ 依赖外部 CLI `mineru-open-api`(需单独安装), 且
            #      flash-extract 有 20 页上限。
            # 两个 PDF 回退现在都是"直产 Block"的路径, 架构上不再有
            # Markdown 中转的解析器。
            return RouteDecision(
                source_type=source.source_type,
                mime_type=PDF_MIME,
                parser="pdf_pdfplumber",
                fallbacks=["pdf_pymupdf"],
                confidence=0.98,
                reason="PDF magic bytes/content-type",
            )

        if mime in HTML_MIMES or b"<html" in source.data[:4096].lower():
            head = source.data[:200000].lower()
            code_heavy = b"<pre" in head or b"<code" in head
            return RouteDecision(
                source_type=source.source_type,
                mime_type=mime or "text/html",
                parser="html_readability",
                fallbacks=[],
                confidence=0.95,
                reason="HTML content-type/sniff",
                mode="code_heavy" if code_heavy else "article",
            )

        if mime in MARKDOWN_MIMES or suffix in {"md", "markdown"}:
            return RouteDecision(
                source_type=source.source_type,
                mime_type=mime or "text/markdown",
                parser="markdown",
                confidence=0.99,
                reason="Markdown MIME/extension",
            )

        if mime in TEXT_MIMES or suffix in {"txt", "rst"}:
            return RouteDecision(
                source_type=source.source_type,
                mime_type=mime or "text/plain",
                parser="plain_text",
                confidence=0.90,
                reason="plain text MIME/extension",
            )

        if mime in OFFICE_MIMES or suffix in {"docx", "pptx", "xlsx"}:
            return RouteDecision(
                source_type=source.source_type,
                mime_type=mime,
                parser="office_optional",
                fallbacks=[],
                confidence=0.90,
                reason="Office Open XML extension",
            )

        raise ValueError(f"暂不支持格式: mime={mime!r}, suffix={suffix!r}")
