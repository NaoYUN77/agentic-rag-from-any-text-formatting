r"""PDF -> DocumentArtifact, 基于 PyMuPDF。

许可证提示
----------
**PyMuPDF 是 AGPL-3.0**（或向 Artifex 购买商业许可）。通过 `import` 链接
使用即构成衍生作品, copyleft 义务会传导到本项目。

因此本 adapter 不是默认选择:
  - 需要许可证干净 -> 用 `pdf_pdfplumber`（MIT）
  - 已有 PyMuPDF 商业许可, 或项目本身就以 AGPL 开源 -> 可用本 adapter

两者在本项目需要的结构信息上等价（同一份 51 页 PDF 实测: 51 页、
67 条大纲、字符级 bbox/字号/字体名一致）。PyMuPDF 的优势是速度(C 库),
pdfplumber 的优势是许可证。

本文件只做数据源适配, 全部解析逻辑在 `pdf_common`。

依赖
----
    pip install pymupdf
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..models import DocumentArtifact, SourcePayload
from .pdf_common import PdfLine, build_artifact


class PyMuPdfSource:
    """把 PyMuPDF 的页面适配成 PdfLine 序列。"""

    def __init__(self, doc: Any) -> None:
        self._doc = doc
        self.page_count = doc.page_count
        self._lines_cache: Dict[int, List[PdfLine]] = {}

    def metadata(self) -> Dict[str, Any]:
        info = getattr(self._doc, "metadata", None) or {}
        return {
            "title": info.get("title"),
            "producer": info.get("producer"),
            "creator": info.get("creator"),
            "author": info.get("author"),
        }

    def outline(self) -> List[Tuple[int, str, int]]:
        """PyMuPDF 的 get_toc 直接给出 (level, title, page), 页码从 1 开始。"""
        out: List[Tuple[int, str, int]] = []
        try:
            toc = self._doc.get_toc(simple=True)
        except Exception:
            return out
        for level, title, page in toc:
            try:
                out.append((int(level), str(title), int(page)))
            except (TypeError, ValueError):
                continue
        return out

    def page_height(self, index: int) -> float:
        try:
            return float(self._doc[index].rect.height)
        except Exception:
            return 0.0

    def page_lines(self, index: int) -> List[PdfLine]:
        if index in self._lines_cache:
            return self._lines_cache[index]

        page = self._doc[index]
        lines: List[PdfLine] = []
        for blk in page.get_text("dict").get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                fonts: Dict[str, int] = {}
                for s in spans:
                    fn = s.get("font") or ""
                    fonts[fn] = fonts.get(fn, 0) + len(s.get("text") or "")
                font = max(fonts.items(), key=lambda kv: kv[1])[0] if fonts else ""
                lines.append(PdfLine(
                    text=text,
                    bbox=[round(float(v), 1) for v in line["bbox"]],
                    size=round(max(float(s["size"]) for s in spans), 1),
                    font=font,
                ))

        lines.sort(key=lambda x: (round(x.bbox[1]), x.bbox[0]))
        self._lines_cache[index] = lines
        return lines

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:
            pass


def parse_pdf_pymupdf(
    source: SourcePayload,
    pages: Optional[str] = None,
) -> DocumentArtifact:
    """PDF -> DocumentArtifact（PyMuPDF / AGPL）。"""
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 PyMuPDF, 无法使用 pdf_pymupdf: pip install pymupdf"
            ) from exc

    pdf_path = source.local_path
    temp_path: Optional[str] = None
    if not pdf_path:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(source.data)
        tmp.close()
        pdf_path = tmp.name
        temp_path = tmp.name

    try:
        doc = pymupdf.open(pdf_path)
        try:
            src = PyMuPdfSource(doc)
            return build_artifact(src, source, pages, parser_name="pymupdf")
        finally:
            doc.close()
    finally:
        if temp_path:
            from pathlib import Path

            Path(temp_path).unlink(missing_ok=True)
