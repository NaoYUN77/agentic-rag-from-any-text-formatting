r"""单句检索单元 + metadata 窗口。

设计意图
--------
LlamaIndex 的 SentenceWindowNodeParser 思路: **检索用单句, 上下文用 metadata**。

    embedding   <- 只嵌单句正文        (向量聚焦, 语义单元小)
    metadata    <- 前后 N 句 + 页码 + bbox + section_path
                  (检索命中后再取出来拼上下文 / 做引用)

与 Contextual Retrieval 的区别
------------------------------
Contextual Retrieval 用 LLM 生成"这段在全文中的角色"作为前缀, 成本
∝ 索引单元数 × 文档长度。本模块**不调 LLM** —— 上下文来自同章节的
相邻句子, 成本为零。

两者可以叠加: 先跑本模块确认单句粒度本身是否有效, 再决定要不要
为每句额外生成 LLM 前缀。

与 BlockAwareHierarchicalChunkBuilder 的关系
-------------------------------------------
并存, 不替代。前者产出 800-token chunk(粗粒度), 本模块产出单句(细粒度),
用于 A/B 对比。

几个刻意的设计取舍
------------------
1. **窗口不跨 section**。跨章节的句子主题不同, 拼进窗口会稀释语义。
2. **标题计入窗口但不单独成句**。标题没有检索价值, 但它给窗口提供
   章节语义, 所以作为上下文参与窗口, 不产出独立 chunk。
3. **原子块(code/formula/table/image)整体成 chunk, 不切句**。
   代码块切句会破坏语法。
4. **句子切分用规则而非模型**。避免新依赖; 中英混排用句末标点 + 常见
   缩写保护。误差可控, 且比"按固定长度切"更贴合语义边界。
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunker import (
    ATOMIC_TYPES,
    IndexReadyChunk,
    ParentNode,
    _render_block,
)
from .models import DocumentArtifact, DocumentBlock


# --------------------------------------------------------------------------
# 句子切分
# --------------------------------------------------------------------------

# 常见缩写, 避免在 "e.g." / "Fig." 处误切
_ABBREV = {
    "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "no", "vol", "pp",
    "mr", "mrs", "ms", "dr", "st", "inc", "ltd", "co", "approx",
}

# 仅由列表标记构成的片段, 如 "1." / "a." / "•" / "(2)"
_LIST_MARKER_ONLY = re.compile(r"^\s*(?:[(（]?(?:\d+|[a-zA-Z]|[ivxIVX]+)[)）.、]|[-*+•·])\s*$")

_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?。！？；;])(?=[\s\u4e00-\u9fff]|$)"
)


def _has_content(text: str, min_chars: int = 3) -> bool:
    """判断片段是否有实质内容（排除纯标记、纯符号）。

    实测坑: 有序列表 "1. Break down..." 会被切成 "1." 和正文两段,
    其中 "1." 只有 2 个字符却会成为一个独立的检索单元。
    """
    core = re.sub(r"[\s\W_]", "", text, flags=re.UNICODE)
    return len(core) >= min_chars


def split_sentences(text: str, min_chars: int = 3) -> List[str]:
    """把一段文本切成句子。

    规则: 在句末标点(中英)之后、且后面紧跟空白或 CJK 字符处切开。
    两类保护:
      - 常见缩写(e.g. / Fig.)不切
      - 列表标记("1." / "a.")不单独成句, 与后续正文合并
    """
    if not text or not text.strip():
        return []

    pieces: List[str] = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.end()
        chunk = text[start:end]
        stripped = chunk.rstrip()

        # 缩写保护
        if stripped.endswith("."):
            last_word = re.split(r"[\s(]", stripped[:-1])[-1].lower()
            if last_word in _ABBREV or (len(last_word) == 1 and last_word.isalpha()):
                continue
        # 列表标记保护: 到这里还只是个标记, 不切
        if _LIST_MARKER_ONLY.match(stripped):
            continue
        if not _has_content(chunk, min_chars):
            continue

        pieces.append(chunk.strip())
        start = end

    tail = text[start:].strip()
    if tail and _has_content(tail, min_chars):
        pieces.append(tail)
    return pieces or ([text.strip()] if _has_content(text, min_chars) else [])


# --------------------------------------------------------------------------
# 单元: 一个句子或一个原子块
# --------------------------------------------------------------------------

class _Unit:
    __slots__ = ("text", "block", "start_char", "end_char", "is_heading",
                 "is_atomic", "token_count")

    def __init__(self, text: str, block: DocumentBlock, start: int, end: int,
                 is_heading: bool, is_atomic: bool, token_count: int) -> None:
        self.text = text
        self.block = block
        self.start_char = start
        self.end_char = end
        self.is_heading = is_heading
        self.is_atomic = is_atomic
        self.token_count = token_count


class SentenceWindowChunkBuilder:
    """单句检索单元 + metadata 窗口。"""

    def __init__(
        self,
        window_sentences: int = 3,
        max_window_tokens: int = 800,
        min_sentence_chars: int = 3,
    ) -> None:
        if window_sentences < 0:
            raise ValueError("window_sentences 不能为负")
        self.window_sentences = window_sentences
        self.max_window_tokens = max_window_tokens
        self.min_sentence_chars = min_sentence_chars

    # ---- 分词: 复用 llama-index 的 SentenceSplitter 做 token 估算 ----
    def _token_size(self, text: str) -> int:
        if not hasattr(self, "_splitter"):
            from llama_index.core.node_parser import SentenceSplitter
            self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=0)
        return int(self._splitter._token_size(text))

    # ---- 把 block 展开成单元序列 ----
    def _block_units(self, block: DocumentBlock) -> List[_Unit]:
        text = block.text or ""
        if not text.strip():
            return []

        if block.type in ATOMIC_TYPES:
            return [_Unit(text, block, 0, len(text), False, True,
                          self._token_size(text))]

        if block.type == "heading":
            return [_Unit(text, block, 0, len(text), True, False,
                          self._token_size(text))]

        units: List[_Unit] = []
        cursor = 0
        for sent in split_sentences(text, self.min_sentence_chars):
            if len(sent) < self.min_sentence_chars:
                continue
            # 在原文里定位这一句, 保证 fragment 坐标可溯源
            start = text.find(sent, cursor)
            if start < 0:
                start = cursor
            end = start + len(sent)
            cursor = max(cursor, end)
            units.append(_Unit(sent, block, start, end, False, False,
                               self._token_size(sent)))
        return units

    # ---- 分组: 与 BlockAwareHierarchicalChunkBuilder 保持一致的 _scope 语义 ----
    @staticmethod
    def _scope(block: DocumentBlock, fallback: str) -> str:
        if block.section_path:
            return block.section_path[0]
        if block.type == "heading":
            return block.text
        return fallback or "文档开头"

    def build(
        self,
        artifact: DocumentArtifact,
    ) -> Tuple[List[ParentNode], List[IndexReadyChunk]]:
        groups: "OrderedDict[str, List[DocumentBlock]]" = OrderedDict()
        fallback = artifact.title or "文档开头"
        for block in sorted(artifact.blocks, key=lambda b: b.order):
            groups.setdefault(self._scope(block, fallback), []).append(block)

        parents: List[ParentNode] = []
        chunks: List[IndexReadyChunk] = []
        parent_counter = 0
        chunk_counter = 0

        for scope, blocks in groups.items():
            parent_counter += 1
            parent_id = f"{artifact.artifact_id}_p{parent_counter:04d}"

            # 组内展开成单元序列(含标题, 标题只作上下文)
            units: List[_Unit] = []
            for block in blocks:
                units.extend(self._block_units(block))
            if not units:
                continue

            parent_chunks: List[IndexReadyChunk] = []

            for idx, unit in enumerate(units):
                if unit.is_heading:
                    continue          # 标题不单独成 chunk, 只参与窗口

                chunk_counter += 1
                window = self._build_window(units, idx)
                section_path = list(unit.block.section_path)
                if unit.is_atomic:
                    section_path = self._atomic_section_path(units, idx, unit)

                chunk = IndexReadyChunk(
                    chunk_id=f"{artifact.artifact_id}_s{chunk_counter:04d}",
                    artifact_id=artifact.artifact_id,
                    parent_id=parent_id,
                    section_block_id=unit.block.block_id,
                    section_path=section_path,
                    fragments=[self._fragment(unit)],
                    full_text=unit.text,
                    sparse_text=unit.text,
                    token_count=unit.token_count,
                    char_count=len(unit.text),
                    page_start=unit.block.page,
                    page_end=unit.block.page,
                    overlap_from_previous=0,
                    metadata={
                        "unit_kind": "atomic" if unit.is_atomic else "sentence",
                        "window_sentences": window["texts"],
                        "window_text": window["text"],
                        "window_token_count": window["token_count"],
                        "window_size": len(window["texts"]),
                        # 引用信息: 供检索后展示, 不参与 embedding
                        "page": unit.block.page,
                        "bbox": unit.block.bbox,
                        "block_id": unit.block.block_id,
                        "url": unit.block.url,
                        "scope": scope,
                    },
                )
                chunks.append(chunk)
                parent_chunks.append(chunk)

            if not parent_chunks:
                parent_counter -= 1
                continue

            parents.append(ParentNode(
                parent_id=parent_id,
                artifact_id=artifact.artifact_id,
                scope_title=scope,
                section_path=list(parent_chunks[0].section_path) or [scope],
                child_chunk_ids=[c.chunk_id for c in parent_chunks],
                full_text="\n\n".join(c.full_text for c in parent_chunks),
                token_count=sum(c.token_count for c in parent_chunks),
                page_start=min((c.page_start for c in parent_chunks
                                if c.page_start), default=None),
                page_end=max((c.page_end for c in parent_chunks
                              if c.page_end), default=None),
            ))

        return parents, chunks

    # ---- 窗口: 前后 N 句, 遇标题边界即停 ----
    def _build_window(self, units: Sequence[_Unit], idx: int) -> Dict[str, Any]:
        n = self.window_sentences
        texts: List[str] = []
        # 往前取
        for j in range(idx - 1, max(-1, idx - 1 - n), -1):
            u = units[j]
            if u.is_heading:
                texts.append(u.text)
                break          # 到标题就停, 不跨章节
            texts.append(u.text)
        texts.reverse()
        # 中心句
        texts.append(units[idx].text)
        # 往后取
        for j in range(idx + 1, min(len(units), idx + 1 + n)):
            u = units[j]
            if u.is_heading:
                break
            texts.append(u.text)

        # 按 token 上限截断(从两端收)
        while len(texts) > 1 and self._token_size("\n".join(texts)) > self.max_window_tokens:
            if len(texts) > 2 and self._token_size("\n".join(texts[1:])) <= self.max_window_tokens:
                texts = texts[1:]
            else:
                texts = texts[:-1]

        joined = "\n".join(texts)
        return {
            "texts": texts,
            "text": joined,
            "token_count": self._token_size(joined),
        }

    # ---- 原子块的 section_path: 用最近的前置标题补全 ----
    @staticmethod
    def _atomic_section_path(
        units: Sequence[_Unit], idx: int, unit: _Unit,
    ) -> List[str]:
        if unit.block.section_path:
            return list(unit.block.section_path)
        for j in range(idx - 1, -1, -1):
            if units[j].is_heading:
                return list(units[j].block.section_path) + [units[j].text]
        return []

    @staticmethod
    def _fragment(unit: _Unit):
        from .chunker import ChunkFragment
        return ChunkFragment(
            block_id=unit.block.block_id,
            start_char=unit.start_char,
            end_char=unit.end_char,
            block_type=unit.block.type,
            text=unit.text,
            page=unit.block.page,
            bbox=list(unit.block.bbox) if unit.block.bbox else None,
        )
