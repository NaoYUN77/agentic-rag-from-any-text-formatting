r"""Block-aware Hierarchical ChunkBuilder。"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from .models import DocumentArtifact, DocumentBlock, IndexDecision, QualityReport


ATOMIC_TYPES = {"code", "formula", "table", "image"}
MERGEABLE_TYPES = {"heading", "text", "list"}
SPARSE_TYPES = {"text", "list", "code", "formula", "table"}


@dataclass
class ChunkFragment:
    """chunk 里的一段连续原文, 以及它在源文档中的位置。

    page / bbox 是为**引用展示**准备的: 生成阶段要把答案里的 [Cn]
    还原成"哪篇文档的哪一页哪一块"。它们不进 embedding, 只随 payload 走。

    一个 chunk 可能跨页, 所以位置放在 fragment 级而不是 chunk 级 ——
    chunk 级只能给一个包围盒, 跨页时那个盒子没有意义。
    """
    block_id: str
    start_char: int
    end_char: int
    block_type: str
    text: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParentNode:
    parent_id: str
    artifact_id: str
    scope_title: str
    section_path: List[str]
    child_chunk_ids: List[str] = field(default_factory=list)
    full_text: str = ""
    token_count: int = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexReadyChunk:
    chunk_id: str
    artifact_id: str
    parent_id: str
    section_block_id: Optional[str]
    section_path: List[str]
    fragments: List[ChunkFragment]
    full_text: str
    sparse_text: str
    token_count: int
    char_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    quality: Optional[QualityReport] = None
    index_decision: Optional[IndexDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["fragments"] = [asdict(f) for f in self.fragments]
        data["quality"] = self.quality.to_dict() if self.quality else None
        data["index_decision"] = self.index_decision.to_dict() if self.index_decision else None
        return data

    def to_llama_text_node(self) -> TextNode:
        node = TextNode(
            id_=self.chunk_id,
            text=self.full_text,
            metadata={
                "artifact_id": self.artifact_id,
                "parent_id": self.parent_id,
                "section_path": " > ".join(self.section_path),
                "token_count": self.token_count,
                "char_count": self.char_count,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "block_ids": [f.block_id for f in self.fragments],
            },
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=self.parent_id),
            },
        )
        return node


@dataclass
class _Piece:
    block: DocumentBlock
    start_char: int
    end_char: int
    text: str
    token_count: int
    is_heading: bool = False
    is_atomic: bool = False
    oversized: bool = False


def _render_block(
    block: DocumentBlock,
    text: str,
    strip_headings: bool = True,
) -> str:
    """把一个 block 渲染成 chunk 正文里的一段。

    strip_headings=True（默认）
    ---------------------------
    标题**不进正文**，只作 metadata。理由:

    1. 标题是"这段来自哪一章"的**引用信息**，不是内容本身 ——
       它已由 `chunk.section_path` + `section_block_id` + heading fragment
       完整承载（引用头 `[C1] 文档: x | 章节: y | 第 N 页` 全部取自 metadata）。
    2. 标题块与正文块被分到不同 section 时（如 PDF 的封面标题），
       旧的渲染方式会产出"只有标题、没有正文"的空壳 chunk。
    3. 实测代价很小: 标题行只占全部 token 的 **1.39%**
       （601 / 43109，用 chunker 自己的 tokenizer 统计）。

    注意: 这只影响 `full_text`（dense embedding 的输入）。
    `dense_text` 字段已删除（与 `full_text` 100% 相同，纯冗余，见 qdrant_indexer._chunk_payload）。
    `sparse_text` 本来就不含 heading（见 SPARSE_TYPES）。
    `fragments[].text` 也**不受影响** —— 它一直是裸标题（不含 `#`），
    评测的 gold 反查依赖它，所以剥离标题不会影响 gold 解析。
    """
    if block.type == "heading":
        if strip_headings:
            return ""
        level = int(block.metadata.get("level", 1))
        return f"{'#' * max(1, min(level, 6))} {text}"
    if block.type == "code":
        return f"```{block.code_language or ''}\n{text}\n```"
    if block.type == "formula":
        return f"$${text}$$"
    if block.type == "image":
        caption = block.caption or block.text
        return f"![{caption}]({block.image_uri or ''})"
    return text


class BlockAwareHierarchicalChunkBuilder:
    """Block 感知的**固定大小**切块器。

    当前策略：固定大小 + 结构边界
    ------------------------------
    切分只由两件事决定：

    1. **结构边界** —— 遇到新 heading 且已有正文时收口。标题是作者自己做过的
       语义切分，免费且高信噪比，所以拿它当边界。
    2. **长度上限** —— 累积到 `chunk_tokens` 就收口；超大原子块
       (`{code,formula,table,image}`) 不拆，独占一个 chunk。

    ⚠️ **overlap 已取消**（2026-09-21）
    --------------------------------
    此前 `overlap_tokens=400` 会在收口时把上一块尾部带回新块。实测它的分布是
    **双峰**的，且与 heading 精确互补：

        有 heading 的文档  -> 章节本身即边界, overlap 几乎不触发 (0~1 个)
        无 heading 的文档  -> 唯一信号就是长度+overlap, 重叠占该文 token 43%~47%

    即近一半 token 预算花在重复内容上，且重叠文本会同时进两个 chunk 的向量
    → 同一内容可被双命中，挤占 `candidate_k`。而它真正起作用的只是
    "接住跨块指代"，实测携带的 2~6 句中只有第 1 句在起作用。
    结论：overlap 收益远小于代价，**移除**。

    ⚠️ **滑动窗口 (`window_tokens`) 已取消**（2026-09-21）
    --------------------------------------------------
    该参数从未参与任何计算（只被接收并保存，后续代码从未读取）。它属于
    "语义切片"的规划范围（用滑动窗口算 embedding 语义距离、在距离峰值处切分），
    该方向暂缓，因此把参数一并删除，避免留下"看起来能用、实际是死参数"的坑。

    → 若将来要做语义边界微调，应先在**无结构文档**上做（有 heading 的地方
      再用语义切分会与作者边界打架），且必须先有评估集验证收益。
    """

    def __init__(
        self,
        chunk_tokens: int = 800,
        strip_headings: bool = True,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be greater than 0")
        self.chunk_tokens = chunk_tokens
        # 标题是否只作 metadata 而不进 chunk 正文（见 _render_block）
        self.strip_headings = strip_headings
        self._splitter = SentenceSplitter(
            chunk_size=chunk_tokens,
            chunk_overlap=0,
        )

    def _token_size(self, text: str) -> int:
        return int(self._splitter._token_size(text))

    def _split_text_block(self, block: DocumentBlock) -> List[_Piece]:
        text = block.text
        total = self._token_size(text)
        if total <= self.chunk_tokens:
            return [_Piece(block, 0, len(text), text, total)]

        # 超限的文本块按句子边界切成 <= chunk_tokens 的 piece。
        # 此前这里用 chunk_tokens - overlap_tokens 预留 overlap 空间,
        # overlap 取消后直接用 chunk_tokens —— 让句子尽量吃满预算,
        # 避免产出偏小的 chunk。
        splitter = SentenceSplitter(
            chunk_size=self.chunk_tokens,
            chunk_overlap=0,
        )
        parts = splitter.split_text(text)
        pieces: List[_Piece] = []
        cursor = 0
        for part in parts:
            start = text.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            pieces.append(_Piece(
                block=block,
                start_char=start,
                end_char=end,
                text=part,
                token_count=self._token_size(part),
            ))
            cursor = max(cursor, end)
        return pieces or [_Piece(block, 0, len(text), text, total)]

    def _block_pieces(self, block: DocumentBlock) -> List[_Piece]:
        tokens = self._token_size(block.text)
        if block.type in ATOMIC_TYPES:
            return [_Piece(
                block=block,
                start_char=0,
                end_char=len(block.text),
                text=block.text,
                token_count=tokens,
                is_atomic=True,
                oversized=tokens > self.chunk_tokens,
            )]
        if block.type == "heading":
            return [_Piece(
                block=block,
                start_char=0,
                end_char=len(block.text),
                text=block.text,
                token_count=tokens,
                is_heading=True,
            )]
        return self._split_text_block(block)

    @staticmethod
    def _scope(block: DocumentBlock, fallback: str) -> str:
        """决定一个 block 归属哪个 parent 分组。

        ⚠️ 用**完整 section_path** 分组，而不是只取 `section_path[0]`。
        ----------------------------------------------------------------
        旧实现只取第一层，实测导致 parent 严重退化（见 issues/10）：

            openai_scaling_storage  9 个章节全都挂在同一个 lv1 标题下
                                    -> section_path[0] 恒为文档标题 -> 1 个 parent
            openai_agents_api       完全无 heading
                                    -> section_path 为空 -> 全部落同一 fallback

        前者是"顶层只有一个值"，后者是"根本没有结构"。取完整路径能修掉前者。

        注意 heading 的特殊性: 标题块的 `section_path` 是【祖先链, 不含自己】
        (见 markdown.py 里 `heading_stack[:-1]` 的语义), 所以:

            第 1 章 (lv1)          -> section_path = []          <- 祖先为空
            第 1 章下的正文块       -> section_path = ['第 1 章']  <- 含自身章节

        因此标题块要**把自己的文本补进路径末尾**，才能与自己章节下的正文块
        落到同一组（正文块的 section_path 正是 `祖先链 + [该标题]`）。
        若不做这一步，标题会与自己的正文分家 —— 实测会让 lv1 标题塌缩成一个
        "章节目录"chunk（只有标题、无正文、sparse_text 为空）。

        无 section_path 且非 heading 的块（即整篇无 heading 的文档）仍落到
        fallback —— 这类文档没有可用的结构信号，本函数无法改善，
        需要靠 parent 尺寸上限兜底（见 issues/10 的待办）。
        """
        if block.type == "heading":
            path = list(block.section_path) + [block.text]
            return " > ".join(str(x) for x in path)
        if block.section_path:
            return " > ".join(str(x) for x in block.section_path)
        return fallback or "文档开头"

    def _group_blocks(self, artifact: DocumentArtifact) -> "OrderedDict[str, List[DocumentBlock]]":
        groups: "OrderedDict[str, List[DocumentBlock]]" = OrderedDict()
        fallback = artifact.title or "文档开头"
        for block in sorted(artifact.blocks, key=lambda b: b.order):
            scope = self._scope(block, fallback)
            groups.setdefault(scope, []).append(block)
        return groups

    def _make_chunk(
        self,
        artifact: DocumentArtifact,
        parent_id: str,
        pieces: Sequence[_Piece],
        chunk_index: int,
    ) -> Optional[IndexReadyChunk]:
        """返回 None 表示该 chunk 正文为空（纯 heading），应被丢弃。"""
        fragments = [ChunkFragment(
            block_id=p.block.block_id,
            start_char=p.start_char,
            end_char=p.end_char,
            block_type=p.block.type,
            text=p.text,
            page=p.block.page,
            bbox=list(p.block.bbox) if p.block.bbox else None,
        ) for p in pieces]
        # strip_headings 下 heading 渲染成 ""，必须先过滤再 join，
        # 否则 full_text 会拼出多余的 "\n\n"。
        full_parts = [
            rendered for rendered in (
                _render_block(p.block, p.text, self.strip_headings) for p in pieces
            ) if rendered.strip()
        ]
        sparse_parts: List[str] = []
        for p in pieces:
            if p.block.type in SPARSE_TYPES:
                sparse_parts.append(p.text)
            elif p.block.type == "image" and (p.block.caption or p.block.text):
                sparse_parts.append(p.block.caption or p.block.text)

        # 纯 heading chunk（如 PDF 的封面标题）在 strip 模式下正文为空。
        # 显式丢弃: 它没有可检索内容, 保留只会造成
        # chunks_loaded 与 dense_points/sparse_points 口径分裂,
        # 并让 gold 反查命中一个"没进索引"的 chunk。
        if not full_parts:
            return None

        token_count = sum(p.token_count for p in pieces)
        char_count = sum(len(p.text) for p in pieces)
        pages = [p.block.page for p in pieces if p.block.page is not None]

        # 引用定位: 只有当整块落在同一页时才给包围盒 ——
        # 跨页的单一 bbox 没有意义, 那种情况由 fragment 级位置承担。
        boxes = [p.block.bbox for p in pieces if p.block.bbox]
        bbox = None
        if boxes and pages and min(pages) == max(pages):
            bbox = [
                round(min(b[0] for b in boxes), 1),
                round(min(b[1] for b in boxes), 1),
                round(max(b[2] for b in boxes), 1),
                round(max(b[3] for b in boxes), 1),
            ]
        chunk_id = f"{artifact.artifact_id}_c{chunk_index:04d}"
        headings = [p for p in pieces if p.block.type == "heading"]
        if headings:
            last_heading = headings[-1]
            section_path = list(last_heading.block.section_path) + [last_heading.text]
            section_block_id = last_heading.block.block_id
        else:
            section_path = list(pieces[0].block.section_path) if pieces else []
            section_block_id = None

        quality_values = []
        total_weight = 0
        flags: List[str] = []
        for p in pieces:
            weight = max(p.token_count, 1)
            q = p.block.quality_score if p.block.quality_score is not None else 1.0
            quality_values.append(q * weight)
            total_weight += weight
            flags.extend(p.block.quality_flags)
            if p.oversized:
                flags.append("oversized_atomic_block")
        chunk_score = sum(quality_values) / max(total_weight, 1)
        quality = QualityReport(
            score=round(chunk_score, 4),
            status="high" if chunk_score >= 0.85 else "medium" if chunk_score >= 0.6 else "low",
            flags=sorted(set(flags)),
            metrics={
                "token_count": token_count,
                "char_count": char_count,
                "fragment_count": len(fragments),
            },
        )
        index_decision = self._decision_for_quality(quality)
        return IndexReadyChunk(
            chunk_id=chunk_id,
            artifact_id=artifact.artifact_id,
            parent_id=parent_id,
            section_block_id=section_block_id,
            section_path=section_path,
            fragments=fragments,
            full_text="\n\n".join(full_parts),
            sparse_text="\n".join(sparse_parts),
            token_count=token_count,
            char_count=char_count,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            quality=quality,
            index_decision=index_decision,
            metadata={
                # 引用信息: 随 payload 走, 不进 embedding
                "page": min(pages) if pages else None,
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "bbox": bbox,
                "block_ids": [p.block.block_id for p in pieces],
                "unit_kind": "block",
                # 块级来源 URL: 取第一个有 url 的 block（HTML 路径会填）
                "url": next((p.block.url for p in pieces if p.block.url), None),
                # 记录本次是否把标题剥出正文 —— 让产物可追溯,
                # 否则"同一份 chunks.jsonl 为什么 token 变少了"无法判断。
                "strip_headings": self.strip_headings,
                # 标题清单: 标题不再进正文后, 这里保留"本 chunk 覆盖了哪些标题",
                # 供引用/上下文补全使用（section_path 只给最后一级的祖先链）。
                "headings": [p.text for p in pieces if p.block.type == "heading"],
            },
        )

    @staticmethod
    def _decision_for_quality(quality: QualityReport) -> IndexDecision:
        if quality.status == "high":
            return IndexDecision("high", True, True, 1.0, 0.0, False, "high quality")
        if quality.status == "medium":
            return IndexDecision("medium", True, True, 0.6, 0.0, False, "medium quality")
        if quality.status == "low":
            return IndexDecision("low", True, False, 0.0, 0.2, True, "low quality")
        return IndexDecision("reject", False, False, 0.0, 1.0, True, "reject quality")

    def build(self, artifact: DocumentArtifact) -> Tuple[List[ParentNode], List[IndexReadyChunk]]:
        parents: List[ParentNode] = []
        chunks: List[IndexReadyChunk] = []
        chunk_counter = 0

        for scope, blocks in self._group_blocks(artifact).items():
            # parent_id 用"已产出的 parent 数 + 1"命名, 而不是循环序号 ——
            # 因为整组 chunk 都可能因"正文为空"被丢弃(strip_headings 下的
            # 纯标题分组), 那种分组不该产出空壳 parent, 也不该在编号上留洞。
            parent_id = f"{artifact.artifact_id}_p{len(parents) + 1:04d}"
            parent_chunks: List[IndexReadyChunk] = []
            current: List[_Piece] = []

            def emit() -> None:
                """收口当前累积的 piece 成一个 chunk, 然后清空累积。

                overlap 取消后收口就是纯粹的"造块 + 清空", 不再有
                "回填上一块尾部"的分支。
                """
                nonlocal current, chunk_counter
                if not current:
                    return
                # 先试造再递增计数: _make_chunk 在"正文为空"(纯 heading)时
                # 返回 None, 该 chunk 被丢弃。若先递增, chunk_id 会出现空洞。
                candidate = self._make_chunk(
                    artifact, parent_id, current, chunk_counter + 1,
                )
                if candidate is not None:
                    chunk_counter += 1
                    chunks.append(candidate)
                    parent_chunks.append(candidate)
                current = []

            for block in blocks:
                for piece in self._block_pieces(block):
                    current_tokens = sum(p.token_count for p in current)
                    has_body = any(not p.is_heading for p in current)

                    # 结构边界: 遇到新标题且已有正文 -> 收口。
                    # 标题是作者自己做过的语义切分, 拿它当边界免费且高信噪比。
                    if piece.is_heading and has_body:
                        emit()
                    # 长度上限: 装不下就先收口, 再让这个 piece 起新块。
                    elif current and current_tokens + piece.token_count > self.chunk_tokens:
                        emit()

                    current.append(piece)

                    # 超大原子块不拆, 单独形成一个 chunk。
                    if piece.is_atomic and piece.oversized:
                        emit()

            if current:
                emit()

            # 注: 这里原本有一段"去重重叠产生的重复 chunk"，已删除。
            # 它的 key 是 (chunk_id, block_ids, full_text)，而 chunk_id 含
            # 单调递增的 chunk_counter、永远唯一，所以那个 key 永不重复，
            # 去重永不生效 —— 是死代码。
            # 实测: 去掉 chunk_id 后 88/88 仍全唯一，当前语料本就无重复可删。
            if not parent_chunks:
                # 该分组的 chunk 全部因正文为空被丢弃 —— 不产出空壳 parent。
                # 实测语料里这类分组 = 纯标题分组（如 PDF 的封面标题）。
                continue

            for chunk in parent_chunks:
                chunk.parent_id = parent_id

            parents.append(ParentNode(
                parent_id=parent_id,
                artifact_id=artifact.artifact_id,
                scope_title=scope,
                section_path=list(parent_chunks[0].section_path) if parent_chunks else [scope],
                child_chunk_ids=[c.chunk_id for c in parent_chunks],
                full_text="\n\n".join(c.full_text for c in parent_chunks),
                token_count=sum(c.token_count for c in parent_chunks),
                page_start=min((c.page_start for c in parent_chunks if c.page_start), default=None),
                page_end=max((c.page_end for c in parent_chunks if c.page_end), default=None),
            ))

        return parents, chunks
