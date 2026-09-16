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
            return RouteDecision(
                source_type=source.source_type,
                mime_type=PDF_MIME,
                parser="pdf_mineru",
                fallbacks=["pdf_docling", "pdf_pymupdf"],
                confidence=0.98,
                reason="PDF magic bytes/content-type",
            )

        if mime in HTML_MIMES or b"<html" in source.data[:4096].lower():
            head = source.data[:200000].lower()
            code_heavy = b"<pre" in head or b"<code" in head
            return RouteDecision(
                source_type=source.source_type,
                mime_type=mime or "text/html",
                parser="html_trafilatura",
                fallbacks=["html_readability"],
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
