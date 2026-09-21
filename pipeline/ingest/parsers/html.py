r"""HTML -> DocumentArtifact, 直接从 DOM 构造 Block, 不经 Markdown 中转。

为什么去掉 Markdown 中转
------------------------
原实现是 `HTML -> markdownify -> Markdown 字符串 -> parse_markdown_text()`。
中间那层 Markdown 会丢掉 HTML 本来有的结构信息:

    链接 href         Markdown 只留文本与目标, 丢了锚文本 / rel / target
    代码语言          靠 ```lang 传递, 但 html 的 class 更可靠
    图片 caption      figure / figcaption 的关联在 Markdown 里不存在
    表格结构          合并单元格、表头在 Markdown 表格里表达不了
    DOM 路径          完全丢失

实测对比（同一份 anthropic 文章, 199 KB）:

    trafilatura xml 输出    只剩 p / ref / hi / code, 标题与列表全被压平
    readability 输出        65 个可块化元素, h2/h3/h4/ol/li/pre/figure 都在

所以清洗用 readability（保留 DOM），不用 trafilatura 的内容抽取 ——
只借用它的元数据抽取，那部分不受影响。

与 PDF 路径的对应关系
--------------------
    PDF 有 page / bbox       HTML 没有页与坐标的概念
    PDF 用原生大纲定层级       HTML 用 h1~h6 标签
    —— 两者都直产 DocumentBlock, 下游无需区分来源
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import trafilatura
from bs4 import BeautifulSoup, Tag
from readability import Document as ReadabilityDocument

from ..models import DocumentArtifact, DocumentBlock, SourcePayload, stable_artifact_id
from .markdown import infer_language


# --------------------------------------------------------------------------
# 标签分类
# --------------------------------------------------------------------------

_HEADING_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# 这些标签自身就是一个块
_LEAF_TAGS = {
    "p", "pre", "table", "blockquote", "ul", "ol", "dl",
    "figure", "img", "hr", "address", "details",
}
_LEAF_TAGS |= set(_HEADING_LEVEL)

# 这些标签是容器, 需要往里递归
_CONTAINER_TAGS = {
    "html", "body", "main", "article", "section", "div",
    "header", "footer", "aside", "nav", "form", "fieldset",
    "figure", "details", "summary", "span", "font", "center",
}

# 内联标签: 文本已由父块的 get_text() 带上, 不单独成块
_INLINE_TAGS = {
    "a", "strong", "b", "em", "i", "u", "s", "code", "small",
    "sub", "sup", "mark", "abbr", "cite", "q", "time", "kbd", "samp", "var",
}

# 常见语言名, 用于从 class 里认出代码语言
_KNOWN_LANGS = {
    "python", "py", "javascript", "js", "typescript", "ts", "java", "go",
    "rust", "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin", "scala",
    "bash", "sh", "shell", "zsh", "powershell", "sql", "json", "yaml", "yml",
    "toml", "ini", "xml", "html", "css", "scss", "markdown", "md", "dockerfile",
    "makefile", "nginx", "text", "plaintext", "diff", "patch",
}


def _clean_text(text: str) -> str:
    """归一化空白: 折叠连续空格, 压缩多空行, 去掉零宽字符。"""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _code_language(el: Tag) -> Optional[str]:
    """从 class 里认代码语言。支持 language-xxx / lang-xxx / 裸语言名。"""
    nodes = [el] + el.find_all("code")
    for node in nodes:
        for cls in (node.get("class") or []):
            low = str(cls).lower()
            for prefix in ("language-", "lang-", "highlight-"):
                if low.startswith(prefix):
                    return low[len(prefix):]
            if low in _KNOWN_LANGS:
                return low
    return None


def _block_links(el: Tag) -> List[Dict[str, str]]:
    """收集块内的链接 —— Markdown 路径会丢掉锚文本与目标。"""
    out: List[Dict[str, str]] = []
    for a in el.find_all("a", href=True):
        href = str(a["href"])
        if not href or href.startswith("#"):
            continue
        out.append({
            "text": _clean_text(a.get_text(" ", strip=True))[:200],
            "href": href,
        })
    return out


# --------------------------------------------------------------------------
# DOM -> Block
# --------------------------------------------------------------------------

class _Builder:
    """按文档顺序遍历 DOM, 产出 DocumentBlock。

    层级用 heading_stack 维护, 语义与 markdown / pdf 路径一致:
    **标题块自身的 section_path 不含自己**。

    heading_stack 存 `(level, title)` 二元组
    ----------------------------------------
    历史 bug (2026-09-20 修复, 与 pdf_common 同一类)
    ------------------------------------------------
    旧实现只存 title, 弹栈条件是 `while len(stack) >= level` ——
    把"栈深度"和"heading 层级"两种量纲相比。

    后果: 栈长 1、level 2 时 `1 >= 2` 为假 → 不弹栈 → **新的同级标题
    被 append 成前一个同级标题的子节点**。实测 `anthropic_agent_skills`:

        b0006 lv2 The anatomy of a skill        section_path=[]
        b0016 lv3 Skills and the context window section_path=['The anatomy...']
        b0021 lv3 Skills and code execution     section_path=[..., 'Skills and the context window']  ← 同级却嵌套
        b0026 lv2 Developing and evaluating...  section_path=['The anatomy...']                       ← 同级却嵌套
        b0038 lv2 Acknowledgements              section_path=['The anatomy...']                       ← 同级却嵌套

    10 篇语料实测影响: parents 16 → 54, 不同顶层 scope 9 → 41。

    正确语义: 新标题 level 必须严格深于栈顶, 否则先把不浅于它的弹出。
    """

    def __init__(self, url: Optional[str]) -> None:
        self.url = url
        self.blocks: List[DocumentBlock] = []
        self.heading_stack: List[Tuple[int, str]] = []
        self.order = 0

    def _add(
        self,
        btype: str,
        text: str,
        el: Optional[Tag] = None,
        level: Optional[int] = None,
        code_language: Optional[str] = None,
        image_uri: Optional[str] = None,
        caption: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = _clean_text(text)
        if not text and btype != "image":
            return

        self.order += 1
        metadata: Dict[str, Any] = {}
        if el is not None:
            links = _block_links(el)
            if links:
                metadata["links"] = links[:20]
            tag = getattr(el, "name", None)
            if tag:
                metadata["tag"] = str(tag)
        if level is not None:
            metadata["level"] = level
        if extra:
            metadata.update(extra)

        self.blocks.append(DocumentBlock(
            block_id=f"b{self.order:04d}",
            type=btype,
            text=text,
            markdown=text,
            section_path=[t for _, t in self.heading_stack],
            order=self.order,
            url=self.url,
            code_language=code_language,
            image_uri=image_uri,
            caption=caption,
            metadata=metadata,
        ))

    def _add_heading(self, el: Tag, level: int) -> None:
        title = _clean_text(el.get_text(" ", strip=True))
        if not title:
            return
        # 用 level 比较(不是栈深): 同级标题必须互斥, 见类 docstring 的历史 bug。
        while self.heading_stack and self.heading_stack[-1][0] >= level:
            self.heading_stack.pop()
        self._add("heading", title, el=el, level=level)
        self.heading_stack.append((level, title))

    def _add_list(self, el: Tag) -> None:
        """整个列表成一个块 —— 列表是语义单元, 拆散会丢上下文。"""
        items = [_clean_text(li.get_text(" ", strip=True))
                 for li in el.find_all("li", recursive=False)]
        items = [x for x in items if x]
        if not items:
            single = _clean_text(el.get_text(" ", strip=True))
            items = [single] if single else []
        if not items:
            return
        body = "\n".join("- " + x for x in items)
        self._add("list", body, el=el, extra={"item_count": len(items)})

    def _add_code(self, el: Tag) -> None:
        # pre 里可能套 code, 取最内层的文本以保留原始换行
        inner = el.find("code") or el
        code = inner.get_text("", strip=False)
        code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not code.strip():
            return
        self._add("code", code, el=el, code_language=_code_language(el))

    def _add_table(self, el: Tag) -> None:
        """表格转成 Markdown 风格的行文本。

        注意: 合并单元格 (rowspan / colspan) 会退化, 这是已知局限。
        """
        rows: List[str] = []
        for tr in el.find_all("tr"):
            cells = [_clean_text(td.get_text(" ", strip=True))
                     for td in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            return
        cap_el = el.find("caption")
        self._add(
            "table", "\n".join(rows), el=el,
            caption=_clean_text(cap_el.get_text(" ", strip=True)) if cap_el else None,
            extra={"row_count": len(rows)},
        )

    def _add_figure(self, el: Tag) -> None:
        img = el.find("img")
        cap_el = el.find("figcaption")
        caption = _clean_text(cap_el.get_text(" ", strip=True)) if cap_el else ""
        if img is not None:
            uri = img.get("src") or img.get("data-src") or ""
            alt = _clean_text(str(img.get("alt") or ""))
            self._add(
                "image", alt or caption or "image", el=el,
                image_uri=str(uri) if uri else None,
                caption=caption or alt or None,
            )
            return
        if caption:
            self._add("text", caption, el=el)

    def _add_img(self, el: Tag) -> None:
        uri = el.get("src") or el.get("data-src") or ""
        alt = _clean_text(str(el.get("alt") or ""))
        if uri or alt:
            self._add("image", alt or "image", el=el,
                      image_uri=str(uri) if uri else None,
                      caption=alt or None)

    def walk(self, el: Tag) -> None:
        """按文档顺序遍历。块级标签成块; 容器递归; 内联忽略。"""
        for child in el.children:
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()

            if name in _HEADING_LEVEL:
                self._add_heading(child, _HEADING_LEVEL[name])
            elif name in ("ul", "ol", "dl"):
                self._add_list(child)
            elif name == "pre":
                self._add_code(child)
            elif name == "table":
                self._add_table(child)
            elif name == "figure":
                self._add_figure(child)
            elif name == "img":
                self._add_img(child)
            elif name in ("blockquote", "address"):
                self._add("text", child.get_text(" ", strip=True), el=child,
                          extra={"quote": True})
            elif name == "details":
                self._add("text", child.get_text(" ", strip=True), el=child)
            elif name == "hr":
                continue
            elif name == "p":
                self._add("text", child.get_text(" ", strip=True), el=child)
            elif name in _CONTAINER_TAGS:
                # 容器里若直接挂着文本(没被 p 包裹), 也要收进来
                direct = _clean_text("".join(
                    str(c) for c in child.children if not isinstance(c, Tag)
                ))
                if len(direct) >= 2:
                    self._add("text", direct, el=child)
                self.walk(child)
            elif name in _INLINE_TAGS:
                continue
            else:
                # 未知标签: 有块级文本就当段落, 否则继续递归
                txt = _clean_text(child.get_text(" ", strip=True))
                if txt and child.find(_LEAF_TAGS | _CONTAINER_TAGS) is None:
                    self._add("text", txt, el=child)
                else:
                    self.walk(child)


# --------------------------------------------------------------------------
# 元数据
# --------------------------------------------------------------------------

def _extract_metadata(html: str, url: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
        if meta:
            out = {
                "title": meta.title,
                "author": meta.author,
                "date": meta.date,
                "sitename": meta.sitename,
                "url": meta.url,
            }
    except Exception:
        pass
    return {k: v for k, v in out.items() if v}


# <title> 里常见的"站点名"分隔符
_TITLE_SEP = re.compile(r"\s*[|\-–—\\»·:：]\s*|\s+::\s+")


def _strip_site_suffix(title: str, site_names: List[str]) -> str:
    """剥掉 <title> 末尾的站点名后缀, 如 "文章标题 \\ Anthropic"。

    保留 <title> 的原义（它比 h1 权威），但去掉噪声后缀 ——
    后缀通常与 og:site_name / sitename 一致, 据此判定。
    """
    if not title or not site_names:
        return title
    for sep_match in reversed(list(_TITLE_SEP.finditer(title))):
        tail = title[sep_match.end():].strip()
        if not tail or len(tail) > 40:
            continue
        for name in site_names:
            if not name:
                continue
            a, b = tail.lower(), str(name).lower()
            if a == b or a in b or b in a:
                head = title[:sep_match.start()].strip()
                # 剥离后太短说明原串可能不是"标题+站点名", 保留原样。
                # 阈值取 4: 中文标题 4 字已足够表意, 而 "A | Site" 这类
                # 会被正确拦下。
                if len(head) >= 4:
                    return head
    return title


def _pick_title(
    html: str, metadata: Dict[str, Any], blocks: List[DocumentBlock],
) -> Optional[str]:
    """按可靠性排序选标题。

    实测: trafilatura 的 title 会取第一个 h1（可能是栏目名/面包屑），
    而 `<title>` 标签才是文档真正的标题。所以优先直接读标签,
    再剥掉末尾的站点名后缀。
    """
    soup = BeautifulSoup(html, "lxml")
    site_names = [metadata.get("sitename"), metadata.get("title_site")]
    og_site = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site and og_site.get("content"):
        site_names.append(str(og_site["content"]))
    site_names = [str(x) for x in site_names if x]

    # 1. <title>（剥站点后缀）
    if soup.title and _clean_text(soup.title.get_text()):
        return _strip_site_suffix(_clean_text(soup.title.get_text()), site_names)
    # 2. og:title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        val = _clean_text(str(og["content"]))
        if val:
            return val
    # 3. readability 的推断
    try:
        val = _clean_text(ReadabilityDocument(html).title())
        if val:
            return _strip_site_suffix(val, site_names)
    except Exception:
        pass
    # 4. trafilatura 的推断
    if metadata.get("title"):
        return str(metadata["title"])
    # 5. 首个 heading
    return next((b.text for b in blocks if b.type == "heading"), None)


# 会被 readability 误删、需要从原始 DOM 补齐的块类型
_SUPPLEMENT_TYPES = ("list", "table", "code", "image")


def _supplement_missing(html: str, builder: "_Builder") -> None:
    """补齐 readability 丢掉的块类型。

    readability 的清洗很激进: 实测在小文档上会整段删掉 <ul>, 而列表往往是
    正文的一部分。这里做一次保守的兜底 —— 只有当某类型**一个都没有**时,
    才从原始 DOM 里捞, 并按 (类型, 文本) 去重, 避免引入重复。
    """
    have = {b.type for b in builder.blocks}
    missing = [t for t in _SUPPLEMENT_TYPES if t not in have]
    if not missing:
        return

    raw = _Builder(url=builder.url)
    raw.walk(BeautifulSoup(html, "lxml"))

    existing = {(b.type, b.text) for b in builder.blocks}
    added = 0
    for b in raw.blocks:
        if b.type not in missing:
            continue
        key = (b.type, b.text)
        if key in existing:
            continue
        existing.add(key)
        builder.order += 1
        b.block_id = f"b{builder.order:04d}"
        b.order = builder.order
        b.section_path = []
        builder.blocks.append(b)
        added += 1
    if added:
        # 补齐的块排到末尾, 重新按 order 排序以保持文档顺序稳定
        builder.blocks.sort(key=lambda x: x.order)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def parse_html(source: SourcePayload) -> DocumentArtifact:
    """HTML -> DocumentArtifact（DOM 直产 Block, 不经 Markdown）。"""
    encoding = source.encoding or "utf-8"
    html = source.data.decode(encoding, errors="replace")
    url = source.final_uri or source.uri or None

    metadata = _extract_metadata(html, url)

    # readability 提正文, 保留 DOM 结构（对比见模块 docstring）
    try:
        doc = ReadabilityDocument(html)
        summary = doc.summary(html_partial=True) or html
    except Exception:
        summary = html

    builder = _Builder(url=url)
    builder.walk(BeautifulSoup(summary, "lxml").body or
                 BeautifulSoup(summary, "lxml"))

    # readability 清洗激进, 会整段删掉 <ul> / <table> 等（实测小文档上丢过 ul）。
    # 只有某类型一个都没有时才从原始 DOM 兜底补齐。
    _supplement_missing(html, builder)

    title = _pick_title(html, metadata, builder.blocks)
    authors = [metadata["author"]] if metadata.get("author") else []

    return DocumentArtifact(
        schema_version="1.0",
        artifact_id=stable_artifact_id(source),
        source_uri=source.final_uri,
        source_type=source.source_type,
        mime_type=source.mime_type or "text/html",
        parser="html_readability",
        parser_version="html-dom-v1",
        title=title,
        authors=authors,
        published_at=metadata.get("date"),
        language=infer_language(" ".join(b.text for b in builder.blocks[:200])),
        blocks=builder.blocks,
        metadata={
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
            "fetched_at": source.fetched_at,
            "url": url,
            **{k: v for k, v in metadata.items() if k != "url"},
        },
    )
