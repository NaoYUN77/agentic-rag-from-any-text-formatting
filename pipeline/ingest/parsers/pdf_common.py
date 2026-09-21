r"""PDF 解析的共享核心逻辑, 与具体 PDF 库解耦。

为什么单独抽一层
----------------
PDF 库的许可证差异很大:
    pdfplumber  MIT          —— 无义务传导
    PyMuPDF     AGPL-3.0     —— 通过 import 链接即构成衍生作品

如果把"大纲匹配 / 页眉过滤 / 段落合并 / 层级分配"这些逻辑写进某个库的
adapter 里, 换库就等于重写。本模块只依赖一个极小的数据源协议
(`PdfSource`), 各库 adapter 只负责把页面读成统一的行结构。

数据源协议
----------
实现 `PdfSource` 即可接入:

    page_count            -> int
    metadata()            -> dict
    outline()             -> [(level, title, page), ...]   页码从 1 开始
    page_height(index)    -> float
    page_lines(index)     -> [PdfLine, ...]

核心逻辑的四个关键处理(均为实测踩坑后得出)
------------------------------------------
1. **层级优先用原生大纲**。PDF 自带的大纲是权威结构, 比"从字号猜"可靠。
2. **页眉页脚需位置 + 字号 + 大纲豁免三重判据**。章首页的正文标题与页眉
   文本相同、位置同在页眉区; 实测第 5 章的正文标题字号(10.6)甚至与
   正文相同, 只能靠"(标题, 页码)命中大纲"来救回。
3. **段落需重建**。部分 PDF(wkhtmltopdf 等)会把一个段落拆成多个文本块,
   必须按"同字号 + 行距连续"合并。
4. **字号层级要滤离群值**。实测存在 67.2pt(正文 6.3 倍)的装饰字号,
   若直接按字号降序排名会霸占 lv1。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..models import DocumentArtifact, DocumentBlock, SourcePayload, stable_artifact_id
from .markdown import infer_language


# --------------------------------------------------------------------------
# 数据源协议
# --------------------------------------------------------------------------

@dataclass
class PdfLine:
    """一行的统一表示。bbox 为 [x0, y0, x1, y1], 坐标原点在左上。"""
    text: str
    bbox: List[float]
    size: float
    font: str


class PdfSource(Protocol):
    page_count: int

    def metadata(self) -> Dict[str, Any]: ...
    def outline(self) -> List[Tuple[int, str, int]]: ...
    def page_height(self, index: int) -> float: ...
    def page_lines(self, index: int) -> List[PdfLine]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# 归一化与判据
# --------------------------------------------------------------------------

_FULLWIDTH = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}
_FULLWIDTH["\u3000"] = " "


def _normalize(text: str) -> str:
    """去掉空白、统一全半角、去掉首尾标点, 用于标题匹配。"""
    text = text.translate(_FULLWIDTH)
    text = re.sub(r"\s+", "", text)
    return text.strip(".,:;，。：；、")


_LIST_RE = re.compile(r"^\s*(?:[-*+•·]|\(?\d+[.)]|[a-zA-Z][.)])\s+\S")

_MONO_HINTS = ("mono", "courier", "consol", "menlo", "code", "fixed", "typewriter")


def _looks_like_list(text: str) -> bool:
    first = text.strip().split("\n", 1)[0]
    return bool(_LIST_RE.match(first))


def _is_mono(font: str) -> bool:
    low = (font or "").lower()
    return any(h in low for h in _MONO_HINTS)


# --------------------------------------------------------------------------
# 页眉页脚
# --------------------------------------------------------------------------

def _collect_zones(
    src: PdfSource,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.90,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[int]]:
    """扫描全部页, 找出跨页重复的页眉页脚, 并记录其**典型字号**。

    字号取众数而非最大值 —— 关键细节: 章首页的正文标题与页眉文本相同、
    位置同在页眉区, 取最大值会被正文标题撑大, 导致页眉与正文标题一起误判。
    """
    top_sizes: Dict[str, Counter] = defaultdict(Counter)
    bottom_sizes: Dict[str, Counter] = defaultdict(Counter)
    pageno_ys: Counter = Counter()

    for pno in range(src.page_count):
        height = src.page_height(pno)
        if not height:
            continue
        for line in src.page_lines(pno):
            if not line.text:
                continue
            y = line.bbox[1]
            if y < height * top_ratio:
                top_sizes[line.text][line.size] += 1
            elif y > height * bottom_ratio:
                bottom_sizes[line.text][line.size] += 1
                if re.fullmatch(r"\d{1,4}", line.text):
                    pageno_ys[round(y)] += 1

    def _mode(counter: Counter) -> float:
        return counter.most_common(1)[0][0]

    headers = {t: _mode(c) for t, c in top_sizes.items() if sum(c.values()) >= 3}
    footers = {t: _mode(c) for t, c in bottom_sizes.items() if sum(c.values()) >= 3}
    page_number_y = pageno_ys.most_common(1)[0][0] if pageno_ys else None
    return headers, footers, page_number_y


def _is_header_footer(
    text: str,
    y: float,
    height: float,
    headers: Dict[str, float],
    footers: Dict[str, float],
    page_number_y: Optional[int],
    outline_keys: set,
    page_no: int,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.90,
) -> bool:
    """判断某行是否为页眉/页脚。

    核心判据: 位置在页眉/页脚区 + 文本跨页重复出现。

    对大纲标题做豁免: 若 (归一化文本, 当前页) 命中大纲, 说明这一页就是该标题
    所在的页, 那它必然是正文标题而非页眉。实测两种必须靠此豁免才能救回的情况:
      - 章首页的正文标题与页眉同文本同位置 (如"第 1 章 ...")
      - 部分章标题字号与正文相同 (实测第 5 章标题 size=10.6, 页眉 size=9.6)
    """
    if (_normalize(text), page_no) in outline_keys:
        return False

    if y < height * top_ratio and text in headers:
        return True
    if y > height * bottom_ratio and text in footers:
        return True

    if page_number_y is not None and abs(y - page_number_y) <= 2:
        if re.fullmatch(r"\d{1,4}", text):
            return True
    return False


# --------------------------------------------------------------------------
# 大纲与字号
# --------------------------------------------------------------------------

def _char_bigrams(text: str) -> set:
    """字符二元组集合, 用于标题模糊匹配。

    为什么用二元组而不是分词: 中文标题没有空格, 且 OCR/换行会引入细微差异,
    二元组对这类噪声不敏感。RAGFlow 的 outline 匹配用同一思路
    (`rag/app/manual.py`, 阈值 0.8)。
    """
    t = _normalize(text)
    if len(t) < 2:
        return {t} if t else set()
    return {t[i] + t[i + 1] for i in range(len(t) - 1)}


def _bigram_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


class _OutlineMatcher:
    """把行文本匹配到 PDF outline, 返回层级。

    两级匹配:
      1. **精确**: (归一化文本, 页码) 命中索引 —— 快路径, 覆盖绝大多数情况
      2. **模糊**: 同页(及相邻页)上做字符二元组 Jaccard >= threshold

    为什么需要第 2 级
    -----------------
    精确匹配要求文本逐字一致。当 OCR 或换行把标题拆散/改写时, 精确匹配会
    静默漏掉, 那个标题就退化成"字号启发式"推断 —— 而字号启发式正是本文件
    里最容易出错的部分 (见 `_calibrate_font_levels`)。模糊匹配把这种漏检
    兜住, 且**只在精确匹配失败时**才启用, 不改变快路径行为。

    实测 (Red Hat 负载平衡器手册): 67/67 条 outline 全部精确命中,
    模糊层未触发 —— 它是鲁棒性保险, 不是当前的收益来源。

    阈值为什么是 0.85 而不是 RAGFlow 的 0.8
    --------------------------------------
    实测二元组 Jaccard 的区分度:

        长标题单字符 OCR 错 ("keepalived"->"keepalive")   0.941  ← 应匹配
        短标题单字符 OCR 错 ("Keepalived"->"Keepalive")   0.850  ← 应匹配
        纯空白差异 (归一化后)                              1.000  ← 精确层已覆盖
        标题追加后缀 ("...调度算法" vs "...调度算法概述")   0.833  ← 应拒绝!
        语义无关 ("法律通告" vs "摘要")                     0.000

    0.8 会把"追加后缀"这种**不同章节**误判为同一标题; 0.85 恰好把
    0.833 挡在外面, 同时仍接受单字符 OCR 错。宁可漏匹配 (退回字号启发式)
    也不要错匹配 (给标题安上错误层级)。
    """

    def __init__(
        self,
        entries: Sequence[Tuple[int, str, int]],
        threshold: float = 0.85,
        page_slack: int = 1,
    ) -> None:
        self._threshold = threshold
        self._page_slack = page_slack
        self._exact: Dict[Tuple[str, int], int] = {}
        # page -> [(bigrams, level, raw_title)]  仅保留有标题文本的条目
        self._by_page: Dict[int, List[Tuple[set, int, str]]] = {}
        self._cache: Dict[Tuple[str, int], Optional[int]] = {}

        for level, title, page in entries:
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                continue
            norm = _normalize(str(title))
            if not norm:
                continue
            key = (norm, page_int)
            if key not in self._exact or level < self._exact[key]:
                self._exact[key] = int(level)
            self._by_page.setdefault(page_int, []).append(
                (_char_bigrams(str(title)), int(level), str(title)))

    def match(self, text: str, page: int) -> Optional[int]:
        norm = _normalize(text)
        if not norm:
            return None
        key = (norm, page)
        if key in self._cache:
            return self._cache[key]

        result = self._exact.get(key)
        if result is None:
            result = self._fuzzy(norm, page)
        self._cache[key] = result
        return result

    def _fuzzy(self, norm: str, page: int) -> Optional[int]:
        """在 page ± slack 范围内找 Jaccard 最高的 outline 条目。"""
        if len(norm) < 4:
            # 太短的标题 (如 "1." / "索引") 二元组太少, 模糊匹配会误伤
            return None
        grams = _char_bigrams(norm)
        best_level: Optional[int] = None
        best_score = 0.0
        for delta in range(-self._page_slack, self._page_slack + 1):
            for other, level, _raw in self._by_page.get(page + delta, ()):  # type: ignore[arg-type]
                score = _bigram_jaccard(grams, other)
                if score > best_score:
                    best_score, best_level = score, level
        if best_score >= self._threshold:
            return best_level
        return None


def _build_outline_index(src: PdfSource) -> Dict[Tuple[str, int], int]:
    """(归一化标题, 页码) -> 层级。"""
    index: Dict[Tuple[str, int], int] = {}
    try:
        entries = src.outline()
    except Exception:
        return index
    for level, title, page in entries:
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            continue
        key = (_normalize(str(title)), page_int)
        if key not in index or level < index[key]:
            index[key] = int(level)
    return index


def _body_font_size(src: PdfSource, max_pages: int = 30) -> float:
    """按字符数统计最高频字号, 作为正文字号。"""
    sizes: Counter = Counter()
    for pno in range(min(src.page_count, max_pages)):
        for line in src.page_lines(pno):
            sizes[round(line.size, 1)] += len(line.text)
    if not sizes:
        return 0.0
    return sizes.most_common(1)[0][0]


def _heading_size_levels(
    src: PdfSource,
    body_size: float,
    ratio: float = 1.15,
    max_ratio: float = 3.0,
    min_hits: int = 1,
    max_pages: int = 40,
    max_level: int = 10,
) -> Dict[float, int]:
    """按字号给标题分配层级: 字号越大层级越浅。

    稳健性处理:
      - 下限 body_size * 1.15: 避免把略大于正文的作者名/日期行当标题
      - 上限 body_size * 3.0 : 滤掉装饰性离群字号
        (实测该 PDF 有 67.2pt = 正文 6.3 倍, 会霸占 lv1 把真标题挤下去)
      - min_hits: 只保留出现 >= min_hits 次的字号

    层级 = 字号排名, **逐档递增, 不同字号必定得到不同层级**。

    历史 bug (2026-09-19 修复)
    --------------------------
    旧实现是 `min(idx + 1, 6)`, 上限 6 会把第 7 档及以后**压平到同一级**。
    实测该 PDF 有 7 个不同标题字号, 于是 13.4pt 和 12.5pt 都拿到 lv6:

        26.9->lv1  23.0->lv2  16.3->lv3  15.4->lv4
        14.4->lv5  13.4->lv6  12.5->lv6   <-- 两档被压平

    后果是 `heading_stack` 的 pop 逻辑错乱 (pop 到 `len >= level` 才停),
    使「摘要」(13.4pt) 与「法律通告」(13.4pt) 被当成同级兄弟后**又互相嵌套**,
    正文块 b0010 的 section_path 变成 `[... '法律通告', '摘要']` —— 摘要被
    塞进法律通告里, 整棵章节树错位。

    现在改为 `idx + 1` 单调递增, 仅保留一个足够宽的 `max_level` 作保护
    (防止极端文档产生几十级深栈)。该 PDF 修复后得到 7 级, 与 PDF 自带
    outline 的 67 条书签层级一致。
    """
    if not body_size:
        return {}
    low = body_size * ratio
    high = body_size * max_ratio
    hits: Counter = Counter()
    for pno in range(min(src.page_count, max_pages)):
        for line in src.page_lines(pno):
            if len(line.text.strip()) > 120:
                continue
            size = round(line.size, 1)
            if low < size <= high:
                hits[size] += 1
    kept = sorted((s for s, n in hits.items() if n >= min_hits), reverse=True)
    return {size: min(idx + 1, max_level) for idx, size in enumerate(kept)}


def _collect_font_level_observations(
    src: PdfSource,
    matcher: "_OutlineMatcher",
    size_levels: Dict[float, int],
    wanted: set,
    max_chars: int = 120,
) -> Dict[float, Counter]:
    """统计每个标题字号在 outline 中**实测**到的层级。

    只统计被 outline 命中的行 —— 那些行的层级来自 PDF 书签, 是可信的。
    返回值形如 {14.4: Counter({2: 17}), 12.5: Counter({3: 21})}。

    这是一次预扫描; `page_lines` 在适配层有缓存, 不重复解析 PDF。
    """
    observations: Dict[float, Counter] = {}
    for pno in range(src.page_count):
        if pno not in wanted:
            continue
        for ln in src.page_lines(pno):
            text = ln.text.strip()
            if not text or len(text) > max_chars:
                continue
            size = round(ln.size, 1)
            if size not in size_levels:
                continue
            level = matcher.match(text, pno + 1)
            if level is None:
                continue
            observations.setdefault(size, Counter())[int(level)] += 1
    return observations


def _calibrate_font_levels(
    size_levels: Dict[float, int],
    observations: Dict[float, Counter],
) -> Dict[float, int]:
    """用 outline 实测的 (字号 -> 层级) 校正字号排名阶梯。

    问题
    ----
    `_heading_size_levels` 返回的是**字号排名**（"在本文档所有标题字号里排第几"），
    不是**结构深度**。两者在字号档数 > 真实层级数时会严重错位。

    实测 (Red Hat 负载平衡器手册):

        字号排名阶梯 (7 档):
          26.9->lv1  23.0->lv2  16.3->lv3  15.4->lv4
          14.4->lv5  13.4->lv6  12.5->lv7
        outline 实测标定 (4 级):
          16.3->lv1  14.4->lv2  12.5->lv3  11.5->lv4

    于是前置页的 13.4pt 标题（法律通告 / 摘要）被排名阶梯判成 **lv6** ——
    比所有正文章节（最深 lv4）都深，语义完全错误；它还会影响
    `_render_block()` 生成的 Markdown 标题层级（`######` vs `###`）。

    做法
    ----
    1. 对每个字号, 取它在 outline 中**实测出现过的层级**（多数票）作为标定值。
    2. 没有 outline 观测的字号, 按字号大小找**最近的已标定字号**插值
       （距离相同时取较浅的一级, 避免过度加深）。
    3. 完全没有 outline 时返回原排名阶梯 —— 退化为旧行为, 不会更差。

    这正是 RAGFlow `title_frequency` 里 `most_level` 的思路: **不要相信
    层级的绝对值, 用文档自身观测到的分布来定层级**。区别是 RAGFlow 用
    "最频繁的层级" 定一个截断点, 这里用 outline 做逐档标定, 更细粒度。

    注意: RAGFlow 的 `if lvl <= most_level` 截断规则**不适用**于本文档 ——
    实测本文档 `most_level` = 6（13.4pt 前置页命中最多）, 若照搬会把
    lv7 的真实章节（21 个 x.y.z 标题）全部截掉。
    """
    if not size_levels:
        return {}

    # 1) 直接标定: 有 outline 观测的字号
    calibrated: Dict[float, int] = {}
    for size, counter in observations.items():
        if size not in size_levels or not counter:
            continue
        # 多数票; 平票时取较浅层级
        level, _ = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        calibrated[size] = int(level)

    if not calibrated:
        return dict(size_levels)

    # 2) 插值: 未标定的字号找最近的已标定字号
    result: Dict[float, int] = {}
    known = sorted(calibrated)
    for size in size_levels:
        if size in calibrated:
            result[size] = calibrated[size]
            continue
        nearest = min(known, key=lambda k: (abs(k - size), calibrated[k]))
        result[size] = calibrated[nearest]
    return result


def _parse_page_range(spec: Optional[str], total: int) -> List[int]:
    if not spec:
        return list(range(total))
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a.strip() else 1
            end = int(b) if b.strip() else total
            pages.extend(range(max(1, start) - 1, min(total, end)))
        else:
            pages.append(int(part) - 1)
    return sorted({p for p in pages if 0 <= p < total})


# --------------------------------------------------------------------------
# 主构建流程
# --------------------------------------------------------------------------

def build_artifact(
    src: PdfSource,
    source: SourcePayload,
    pages: Optional[str] = None,
    parser_name: str = "pdf",
) -> DocumentArtifact:
    """从任意 PdfSource 构建 DocumentArtifact。"""
    outline_index = _build_outline_index(src)
    outline_keys = set(outline_index)
    headers, footers, page_number_y = _collect_zones(src)
    body_size = _body_font_size(src)
    size_levels = _heading_size_levels(src, body_size)
    wanted = set(_parse_page_range(pages, src.page_count))

    # ---- outline 匹配器 + 字号层级标定 ----
    # outline 是权威层级来源; 字号排名只是它的兜底。标定把兜底从
    # "字号排第几" 修正为 "字号在本文档里实际对应第几级"。
    try:
        outline_entries = list(src.outline())
    except Exception:
        outline_entries = []
    matcher = _OutlineMatcher(outline_entries)
    font_observations = _collect_font_level_observations(
        src, matcher, size_levels, wanted,
    )
    font_levels = _calibrate_font_levels(size_levels, font_observations)
    stats_calibration = {
        "font_levels": {str(k): v for k, v in sorted(font_levels.items(), reverse=True)},
        "font_rank_levels": {str(k): v for k, v in sorted(size_levels.items(), reverse=True)},
    }

    blocks: List[DocumentBlock] = []
    # (rank, text) 二元组: rank 用于弹栈比较, text 用于拼 section_path。
    # 见下方 heading 分支的注释 (2026-09-19 嵌套 bug 修复)。
    heading_stack: List[Tuple[int, str]] = []
    order = 0
    stats: Counter = Counter()
    # 与 html / plain / markdown 路径保持一致: block 级 url 从 final_uri 取。
    # 本地 PDF 时 final_uri 是文件路径 (见 source.py), 引用展示同样需要。
    url = source.final_uri or source.uri or None

    for pno in range(src.page_count):
        if pno not in wanted:
            continue
        height = src.page_height(pno) or 1.0
        page_no = pno + 1

        lines = []
        for ln in src.page_lines(pno):
            if not ln.text.strip():
                continue
            if _is_header_footer(
                ln.text, ln.bbox[1], height, headers, footers,
                page_number_y, outline_keys, page_no,
            ):
                stats["dropped_header_footer"] += 1
                continue
            lines.append(ln)
        lines.sort(key=lambda x: (round(x.bbox[1]), x.bbox[0]))

        buffer: List[PdfLine] = []

        def flush() -> None:
            nonlocal order, buffer
            if not buffer:
                return
            text = "\n".join(x.text for x in buffer).strip()
            if not text:
                buffer = []
                return
            size = max(x.size for x in buffer)
            bbox = [
                min(x.bbox[0] for x in buffer),
                min(x.bbox[1] for x in buffer),
                max(x.bbox[2] for x in buffer),
                max(x.bbox[3] for x in buffer),
            ]
            order += 1
            btype = "list" if _looks_like_list(text) else "text"
            stats[btype] += 1
            blocks.append(DocumentBlock(
                block_id=f"b{order:04d}",
                type=btype,
                text=text,
                markdown=text,
                section_path=[t for _, t in heading_stack],
                order=order,
                page=page_no,
                bbox=bbox,
                url=url,
                metadata={
                    "font": buffer[0].font,
                    "font_size": size,
                    "page": page_no,
                    "line_count": len(buffer),
                },
            ))
            buffer = []

        for ln in lines:
            # outline 优先 (精确 -> 二元组模糊兜底); 字号只作兜底,
            # 且用**标定后**的层级而非原始字号排名。
            outline_level = matcher.match(ln.text, page_no)
            is_font_heading = ln.size in size_levels and len(ln.text) <= 120

            if outline_level is not None or is_font_heading:
                flush()
                level = (int(outline_level) if outline_level is not None
                         else font_levels.get(ln.size, size_levels[ln.size]))

                # 去重: 章首页的页眉与正文标题同文本同页, 会产出两个 heading。
                # 保留字号更大的那个（正文标题通常大于页眉）。
                if (blocks and blocks[-1].type == "heading"
                        and blocks[-1].page == page_no
                        and _normalize(blocks[-1].text) == _normalize(ln.text)):
                    if ln.size > blocks[-1].metadata.get("font_size", 0.0):
                        blocks.pop()
                        order -= 1
                    else:
                        stats["dedup_heading"] += 1
                        continue

                stats["heading_from_outline" if outline_level is not None
                      else "heading_from_font"] += 1
                # 用 (rank, text) 建栈: rank 是层级依据, text 是 section_path 内容。
                #
                # 历史 bug (2026-09-19 修复)
                # --------------------------
                # 旧实现只存 text, 用 `while len(stack) >= level` 决定弹栈。
                # 这把"栈深度"和"字号排名"两种量纲相比, 语义是错的:
                # 同字号(同 rank)的连续标题不会被 pop, 于是层层嵌套。
                # 实测该 PDF 三个 13.4pt 的前置标题排名都是 lv6, 而栈只有 2 层,
                # `len(stack) >= 6` 永不成立 -> 「摘要」错成「法律通告」的子节点,
                # 正文 b0010 的 section_path 变成 [..., '法律通告', '摘要']。
                #
                # 正确语义: 新标题 rank 必须严格深于栈顶, 否则先把不浅于它的弹出。
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                order += 1
                blocks.append(DocumentBlock(
                    block_id=f"b{order:04d}",
                    type="heading",
                    text=ln.text,
                    markdown=ln.text,
                    section_path=[t for _, t in heading_stack],
                    order=order,
                    page=page_no,
                    bbox=list(ln.bbox),
                    url=url,
                    metadata={
                        "font": ln.font,
                        "font_size": ln.size,
                        "page": page_no,
                        "level": level,
                        "from_outline": outline_level is not None,
                    },
                ))
                heading_stack.append((level, ln.text))
                continue

            # 段落续接: 同字号 且 行距连续
            if buffer:
                prev = buffer[-1]
                gap = ln.bbox[1] - prev.bbox[3]
                if ln.size != prev.size or gap > max(ln.size * 1.6, 6.0):
                    flush()
            buffer.append(ln)

        flush()

    meta = src.metadata() or {}
    title = str(meta.get("title") or "").strip() or None
    if not title:
        title = next((b.text for b in blocks if b.type == "heading"), None)

    return DocumentArtifact(
        schema_version="1.0",
        artifact_id=stable_artifact_id(source),
        source_uri=source.final_uri,
        source_type=source.source_type,
        mime_type=source.mime_type or "application/pdf",
        parser=parser_name,
        parser_version=f"{parser_name}-adapter-v1",
        title=title,
        language=infer_language(" ".join(b.text for b in blocks[:200])),
        blocks=blocks,
        metadata={
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
            "fetched_at": source.fetched_at,
            "pdf_page_count": src.page_count,
            "pdf_pages_parsed": len(wanted),
            "pdf_title_meta": meta.get("title"),
            "pdf_producer": meta.get("producer"),
            "outline_entries": len(outline_index),
            "body_font_size": body_size,
            "heading_size_levels": {str(k): v for k, v in size_levels.items()},
            # 标定后的字号层级 (outline 实测校正) + 观测样本, 便于诊断
            "font_levels_calibrated": stats_calibration["font_levels"],
            "font_level_observations": {
                str(k): dict(v) for k, v in sorted(
                    font_observations.items(), reverse=True)
            },
            "headers_detected": sorted(headers)[:10],
            "stats": dict(stats),
        },
    )
