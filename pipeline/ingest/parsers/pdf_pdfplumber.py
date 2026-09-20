r"""PDF -> DocumentArtifact, 基于 pdfplumber (MIT 许可)。

为什么默认用这个而不是 PyMuPDF
-------------------------------
PyMuPDF 是 AGPL-3.0: 通过 `import` 链接使用即构成衍生作品, copyleft 义务
会传导到本项目。pdfplumber 是 MIT, 无义务传导。

实测两者在本项目需要的结构信息上等价(同一份 51 页 PDF):
    页数        51 / 51
    原生大纲    67 / 67 条, 层级与页码一致
    字符级 bbox ✓
    字号        ✓
    字体名      ✓

代价是速度: pdfplumber 基于纯 Python 的 pdfminer.six, 比 PyMuPDF 的 C 库慢。
对离线批处理不构成问题。

依赖
----
    pip install pdfplumber
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..models import DocumentArtifact, SourcePayload
from .pdf_common import PdfLine, build_artifact


class PdfplumberSource:
    """把 pdfplumber 的页面适配成 PdfLine 序列。"""

    def __init__(self, pdf: Any) -> None:
        self._pdf = pdf
        self.page_count = len(pdf.pages)
        self._height_cache: Dict[int, float] = {}
        self._lines_cache: Dict[int, List[PdfLine]] = {}
        self._pageid_to_no = self._build_pageid_map(pdf)

    # ---- 页面 id -> 页码, 供大纲定位用 ----
    @staticmethod
    def _build_pageid_map(pdf: Any) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for idx, page in enumerate(pdf.pages):
            try:
                pageid = page.page_obj.pageid
            except Exception:
                continue
            mapping[pageid] = idx + 1
        return mapping

    # ---- 协议实现 ----
    def metadata(self) -> Dict[str, Any]:
        info = getattr(self._pdf, "metadata", None) or {}
        return {
            "title": info.get("Title"),
            "producer": info.get("Producer"),
            "creator": info.get("Creator"),
            "author": info.get("Author"),
        }

    def outline(self) -> List[Tuple[int, str, int]]:
        """返回 [(level, title, page), ...], 页码从 1 开始。"""
        out: List[Tuple[int, str, int]] = []
        doc = getattr(self._pdf, "doc", None)
        if doc is None:
            return out

        try:
            from pdfminer.pdfinterp import resolve1
            from pdfminer.psparser import PSLiteral
        except ImportError:
            return out

        try:
            entries = doc.get_outlines()
        except Exception:
            return out

        for entry in entries:
            try:
                level, title, dest, _action, _se = entry
            except (ValueError, TypeError):
                continue

            # dest 的形态比较绕, 实测需要处理三层:
            #   1. PSLiteral    —— 具名目标, 如 /'__WKANCHOR_3s' (本 PDF 就是这种)
            #   2. str          —— 具名目标的字符串形式
            #   3. list/tuple   —— 显式目标 [page_ref, /XYZ, ...]
            # 且 doc.get_dest() 返回的是 PDFObjRef, 必须再 resolve1 一次才是数组。
            if isinstance(dest, PSLiteral):
                try:
                    dest = doc.get_dest(dest.name)
                except Exception:
                    dest = None
            elif isinstance(dest, str):
                try:
                    dest = doc.get_dest(dest)
                except Exception:
                    dest = None

            if dest is not None and not isinstance(dest, (list, tuple)):
                try:
                    dest = resolve1(dest)
                except Exception:
                    dest = None

            page_no = None
            if isinstance(dest, (list, tuple)) and dest:
                first = dest[0]
                # 形态 A: 直接是 0-based 页索引 (本 PDF 就是这种, 已与 PyMuPDF 核对一致)
                if isinstance(first, int):
                    candidate = first + 1
                    if 1 <= candidate <= self.page_count:
                        page_no = candidate
                # 形态 B: 是页面引用对象, 需 resolve 后取 pageid
                if page_no is None:
                    try:
                        target = resolve1(first)
                        page_id = getattr(target, "pageid", None)
                        if page_id is not None:
                            page_no = self._pageid_to_no.get(page_id)
                    except Exception:
                        page_no = None

            if page_no is None:
                continue
            out.append((int(level), str(title), int(page_no)))
        return out

    def page_height(self, index: int) -> float:
        if index not in self._height_cache:
            try:
                self._height_cache[index] = float(self._pdf.pages[index].height)
            except Exception:
                self._height_cache[index] = 0.0
        return self._height_cache[index]

    def page_lines(self, index: int) -> List[PdfLine]:
        if index in self._lines_cache:
            return self._lines_cache[index]

        page = self._pdf.pages[index]

        # 去重: wkhtmltopdf 等工具会用 faux-bold 手法把同一段文字
        # 在相差约 0.5pt 的 x 位置绘制两遍, pdfminer 不做去重,
        # 会把 "法律通告" 提取成 "法法律律通通告告"。
        # 实测该 PDF 有 205/463 个块受影响。tolerance=1.5 可消除。
        try:
            page = page.dedupe_chars(tolerance=1.5)
        except Exception:
            pass

        lines: List[PdfLine] = []
        try:
            raw_lines = page.extract_text_lines()
        except Exception:
            raw_lines = []

        for raw in raw_lines:
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            chars = raw.get("chars") or []
            if chars:
                size = round(max(c.get("size", 0.0) for c in chars), 1)
                # 取字符数最多的字体名, 避免被个别装饰字符带偏
                fonts: Dict[str, int] = {}
                for c in chars:
                    fn = c.get("fontname") or ""
                    fonts[fn] = fonts.get(fn, 0) + 1
                font = max(fonts.items(), key=lambda kv: kv[1])[0] if fonts else ""
            else:
                size = 0.0
                font = ""
            lines.append(PdfLine(
                text=text,
                bbox=[
                    round(float(raw.get("x0", 0.0)), 1),
                    round(float(raw.get("top", 0.0)), 1),
                    round(float(raw.get("x1", 0.0)), 1),
                    round(float(raw.get("bottom", 0.0)), 1),
                ],
                size=size,
                font=font,
            ))

        lines.sort(key=lambda x: (round(x.bbox[1]), x.bbox[0]))
        self._lines_cache[index] = lines
        return lines

    def close(self) -> None:
        try:
            self._pdf.close()
        except Exception:
            pass


def parse_pdf_pdfplumber(
    source: SourcePayload,
    pages: Optional[str] = None,
) -> DocumentArtifact:
    """PDF -> DocumentArtifact（pdfplumber / MIT）。"""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 pdfplumber, 无法使用 pdf_pdfplumber: pip install pdfplumber"
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
        with pdfplumber.open(pdf_path) as pdf:
            src = PdfplumberSource(pdf)
            return build_artifact(src, source, pages, parser_name="pdfplumber")
    finally:
        if temp_path:
            from pathlib import Path

            Path(temp_path).unlink(missing_ok=True)
