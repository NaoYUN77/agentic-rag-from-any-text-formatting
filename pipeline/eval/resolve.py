r"""把 gold 的文本锚点解析成当前索引里的 chunk_id 集合。

这是整个评估 harness 的地基: gold 存的是原文片段, 评估时必须把它映射到
"当前这份索引里, 哪几个 chunk 包含这段文字"。

三个必须守住的规则(都是实测踩坑后得出)
--------------------------------------
1. **键必须带文档标识**。`block_id` 是文档内位置序号(b0001), 每篇文档都
   从 b0001 开始。实测当前 10 篇语料里有 99 个 block_id 跨文档碰撞 ——
   只用 block_id 建索引会把不同文档的块混在一起, gold 反查会命中错误的
   chunk, 而且**不报错**, 只给出错误的指标。
   → 一律用 (artifact_id, block_id)。

2. **跳过 image 的空 fragment**。实测 729 个 fragment 里有 24 个
   text='' span=[0,0), 全是 image 块(图片只有 URI)。参与反查会污染结果。

3. **解析失败必须报错, 不能静默跳过**。静默跳过会让 recall 虚高
   (分母变小), 是最隐蔽的错误来源。

4. **匹配前两侧都要做文本归一化**。实测: 去掉 Markdown 中转层后,
   `**Context** refers to ...` 变成 `Context refers to ...` ——
   同一句话, 只是星号没了。如果 gold 片段里带着 `**`, 直接子串匹配
   会**误报"标注失效"**。这不是内容丢失, 是标记层被移除。
   → 归一化: 去掉 Markdown 强调标记 + 折叠空白。两侧都做。

5. **链接语法与标点前空格也是解析层差异**(2026-09-19 实测, 39 条 qrels
   扩到后 OLD 语料 3 条 FAIL 的根因):
   - OLD(Markdown 中转)把超链接写成 `[simple definition](https://...)`,
     NEW(DOM 直产)只有 `simple definition` —— URL 不是正文内容;
   - NEW 的 DOM 行内元素边界会挤出 `Agents , on the other hand` 这样的
     **标点前空格**, OLD 是 `Agents, on the other hand`。
   → 归一化追加: 剥离 `[anchor](url)` 只留 anchor + 折叠标点前空白。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .qrels import GoldSpan


# Markdown 强调标记: **bold** / __bold__ / *it* / _it_ / `code`
# 归一化时一律去掉 —— 这些是**渲染标记**, 不是正文内容。
# 去掉 Markdown 中转层后, 正文里不再有这些字符, 而旧标注里可能有。
#
# 关键: 必须**成对**匹配, 不能见一个删一个。
# 否则会把标识符里的下划线删掉: `doc_a` -> `doca`, `snake_case` -> `snakecase`。
# (这正是第一版写成 [*_`] 删除后踩的坑。)
_MD_EMPHASIS = re.compile(
    r"\*{1,3}(?=\S)(.+?)(?<=\S)\*{1,3}"
    r"|(?<![\w])_{1,3}(?=\S)(.+?)(?<=\S)_{1,3}(?![\w])"
    r"|`{1,3}(?=\S)(.+?)(?<=\S)`{1,3}",
    re.DOTALL,
)
# Markdown 链接 [anchor](url): 保留 anchor, 丢掉 URL —— URL 不是正文内容。
# 放在强调标记之前处理, 这样 **[x](url)** 会先变 **x** 再被强调规则清掉。
_MD_LINK = re.compile(r"\[([^\]\[]+)\]\([^)\s]*\)")
# 标点前的空白: DOM 行内元素边界会挤出 "Agents , on" 这样的伪影,
# Markdown 中转层则是正常的 "Agents, on"。两侧统一折叠掉。
_PRE_PUNCT_WS = re.compile(r"\s+([,.;:!?])")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """把文本归一化到"只比较正文"的形式。

    做四件事, 顺序不能反:
      1. 剥离 Markdown 链接语法（`[anchor](url)` -> `anchor`）
      2. 去掉 Markdown 强调标记（成对的 **  __  *  _  `）
      3. 折叠标点前空白（`Agents , on` -> `Agents, on`）
      4. 折叠所有空白为单个空格

    两侧（gold 片段 / 索引 fragment）都要调用, 否则归一化没有意义。
    """
    text = _MD_LINK.sub(r"\1", text)
    # 只保留捕获组里真正的内容, 丢掉标记本身。
    text = _MD_EMPHASIS.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _PRE_PUNCT_WS.sub(r"\1", text)
    return _WS.sub(" ", text).strip()


class GoldResolveError(RuntimeError):
    """gold 无法映射到当前索引 —— 通常意味着标注已失效或索引不匹配。"""


@dataclass
class FragmentRef:
    """索引里的一个 fragment 及其归属。"""
    chunk_id: str
    point_id: Optional[int]
    artifact_id: str
    block_id: str
    block_type: str
    start_char: int
    end_char: int
    text: str


@dataclass
class ResolveReport:
    """逐条 gold 的解析结果, 供人工核对。"""
    query_id: str
    gold_index: int
    text: str
    chunk_ids: Set[str] = field(default_factory=set)
    matched_fragments: int = 0
    resolved: bool = False
    note: str = ""


class FragmentIndex:
    """(artifact_id, block_id) -> [FragmentRef] 的索引。

    刻意不用裸 block_id 作键 —— 见模块 docstring 规则 1。
    """

    def __init__(self) -> None:
        self._by_key: Dict[Tuple[str, str], List[FragmentRef]] = {}
        self._all: List[FragmentRef] = []
        self.skipped_empty = 0
        self.skipped_image = 0

    def add(self, ref: FragmentRef) -> None:
        if ref.block_type == "image" or not ref.text.strip():
            if ref.block_type == "image":
                self.skipped_image += 1
            else:
                self.skipped_empty += 1
            return
        self._by_key.setdefault((ref.artifact_id, ref.block_id), []).append(ref)
        self._all.append(ref)

    @property
    def size(self) -> int:
        return len(self._all)

    @property
    def keys(self) -> int:
        return len(self._by_key)

    def search(self, snippet: str) -> List[FragmentRef]:
        """按文本子串查找所有包含它的 fragment。

        对整库线性扫描。当前语料规模(千级 fragment)下足够快,
        且不引入任何近似匹配的误差 —— 评估的正确性优先于速度。

        两侧都必须走 normalize_text: 索引侧在 build_fragment_index 存入时归一化,
        查询侧在这里归一化。只归一化一边等于没归一化。
        """
        needle = normalize_text(snippet)
        if not needle:
            return []
        return [ref for ref in self._all if needle in ref.text]


def _iter_chunk_records(chunks_path: Path) -> Iterable[Dict[str, Any]]:
    """读 chunks.jsonl。

    兼容两种格式:
      - 原始 chunk dump:  字段在顶层 (chunk_id / fragments / ...)
      - 稀疏索引产物:     业务字段在 metadata 里
    """
    with io.open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata")
            if isinstance(meta, dict) and "chunk_id" in meta:
                merged = dict(meta)
                merged.setdefault("point_id", rec.get("point_id"))
                yield merged
            else:
                yield rec


def fragment_ref_from_record(
    rec: Dict[str, Any],
    fr: Dict[str, Any],
    chunk_id: str,
) -> FragmentRef:
    """把一条 chunk record 里的一个 fragment 转成 FragmentRef。

    唯一的构造点 —— 任何"自己拼 FragmentRef"的地方都必须走这里,
    否则 text 归一化会被绕过, gold 解析就会莫名其妙地失败。
    (踩过的坑: check_qrels_resolution.py 自己拼了一份, 结果修了 build_fragment_index
     却依然 FAIL。)
    """
    pid = rec.get("point_id")
    return FragmentRef(
        chunk_id=chunk_id,
        point_id=pid if isinstance(pid, int) else None,
        artifact_id=str(rec.get("artifact_id") or ""),
        block_id=str(fr.get("block_id") or ""),
        block_type=str(fr.get("block_type") or ""),
        start_char=int(fr.get("start_char") or 0),
        end_char=int(fr.get("end_char") or 0),
        # 存入时即归一化, 见 normalize_text。
        # gold 片段是标注者在**旧语料**(含 Markdown 标记)上写的,
        # 去掉 Markdown 中转层后正文不再有 ** 之类的字符。
        # 两边统一归一化, 标注才不会随解析层变动而失效。
        text=normalize_text(str(fr.get("text") or "")),
    )


def build_fragment_index(chunks_path: str | Path) -> FragmentIndex:
    """从 chunks.jsonl 构建 fragment 索引。"""
    p = Path(chunks_path)
    if not p.exists():
        raise GoldResolveError(f"chunks.jsonl 不存在: {p}")

    index = FragmentIndex()
    for rec in _iter_chunk_records(p):
        chunk_id = str(rec.get("chunk_id") or "")
        if not chunk_id:
            continue
        for fr in rec.get("fragments") or []:
            index.add(fragment_ref_from_record(rec, fr, chunk_id))
    return index


def build_fragment_index_multi(paths: Iterable[str | Path]) -> FragmentIndex:
    """把多篇文档的 chunks.jsonl 合成一个索引。

    评估集是跨文档的, 所以需要一个全库索引。
    用 (artifact_id, block_id) 做键, 不同文档的同名 block 不会互相覆盖。
    """
    index = FragmentIndex()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for rec in _iter_chunk_records(p):
            chunk_id = str(rec.get("chunk_id") or "")
            if not chunk_id:
                continue
            for fr in rec.get("fragments") or []:
                index.add(fragment_ref_from_record(rec, fr, chunk_id))
    return index


def resolve_gold(
    gold: GoldSpan,
    index: FragmentIndex,
    query_id: str = "",
    gold_index: int = 0,
    min_snippet_chars: int = 8,
) -> ResolveReport:
    """把一条 gold 解析成 chunk_id 集合。

    主路径是文本子串匹配 —— 它对 block_id 失效免疫。
    匹配不到时**抛错**, 不静默返回空集。
    """
    snippet = (gold.text or "").strip()
    report = ResolveReport(query_id=query_id, gold_index=gold_index, text=snippet)

    if len(snippet) < min_snippet_chars:
        raise GoldResolveError(
            f"{query_id} gold[{gold_index}]: 片段只有 {len(snippet)} 字符, "
            f"低于 {min_snippet_chars} —— 太短会多命中, 请给更长的原文片段"
        )

    hits = index.search(snippet)
    if not hits:
        raise GoldResolveError(
            f"{query_id} gold[{gold_index}]: 文本反查无结果, 标注可能已失效。\n"
            f"  片段: {snippet[:80]!r}\n"
            f"  索引: {index.size} 个 fragment / {index.keys} 个 (文档,块) 键\n"
            f"  排查: 该片段是否真的存在于当前语料? 是否被 cleaner 过滤掉了?"
        )

    report.chunk_ids = {h.chunk_id for h in hits}
    report.matched_fragments = len(hits)
    report.resolved = True
    if len(report.chunk_ids) > 1:
        report.note = f"命中 {len(report.chunk_ids)} 个 chunk(overlap 导致, 正常)"
    return report


def resolve_all(
    items: Iterable[Any],
    index: FragmentIndex,
) -> Tuple[Dict[str, Set[str]], List[ResolveReport]]:
    """批量解析。返回 {query_id: gold_chunk_ids} 与逐条报告。

    任何一条失败都会抛出 —— 不跳过。
    """
    gold_map: Dict[str, Set[str]] = {}
    reports: List[ResolveReport] = []
    for item in items:
        merged: Set[str] = set()
        for gi, g in enumerate(item.gold):
            rep = resolve_gold(g, index, query_id=item.query_id, gold_index=gi)
            reports.append(rep)
            merged |= rep.chunk_ids
        gold_map[item.query_id] = merged
    return gold_map, reports


def chunks_sha256(chunks_path: str | Path) -> str:
    """产物指纹, 写进 manifest 保证指标可比。"""
    p = Path(chunks_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
