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
    block_id: str
    start_char: int
    end_char: int
    block_type: str
    text: str

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
    dense_text: str
    sparse_text: str
    token_count: int
    char_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    overlap_from_previous: int = 0
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


def _render_block(block: DocumentBlock, text: str) -> str:
    if block.type == "heading":
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
    def __init__(
        self,
        chunk_tokens: int = 800,
        overlap_tokens: int = 400,
        window_tokens: int = 400,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be greater than 0")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be greater than or equal to 0")
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.window_tokens = window_tokens
        self._text_piece_tokens = max(1, chunk_tokens - overlap_tokens)
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

        splitter = SentenceSplitter(
            chunk_size=self._text_piece_tokens,
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
        if block.section_path:
            return block.section_path[0]
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
        overlap_from_previous: int,
    ) -> IndexReadyChunk:
        fragments = [ChunkFragment(
            block_id=p.block.block_id,
            start_char=p.start_char,
            end_char=p.end_char,
            block_type=p.block.type,
            text=p.text,
        ) for p in pieces]
        full_parts = [_render_block(p.block, p.text) for p in pieces]
        dense_parts = [part for part in full_parts if part.strip()]
        sparse_parts: List[str] = []
        for p in pieces:
            if p.block.type in SPARSE_TYPES:
                sparse_parts.append(p.text)
            elif p.block.type == "image" and (p.block.caption or p.block.text):
                sparse_parts.append(p.block.caption or p.block.text)

        token_count = sum(p.token_count for p in pieces)
        char_count = sum(len(p.text) for p in pieces)
        pages = [p.block.page for p in pieces if p.block.page is not None]
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
            dense_text="\n\n".join(dense_parts),
            sparse_text="\n".join(sparse_parts),
            token_count=token_count,
            char_count=char_count,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            overlap_from_previous=overlap_from_previous,
            quality=quality,
            index_decision=index_decision,
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

    def _slice_tail(self, piece: _Piece, target_tokens: int) -> Optional[_Piece]:
        if target_tokens <= 0:
            return None
        if piece.is_atomic or piece.is_heading:
            return None
        if piece.token_count <= target_tokens:
            return piece

        text = piece.text
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high) // 2
            if self._token_size(text[mid:]) <= target_tokens:
                high = mid
            else:
                low = mid + 1
        start = low
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return None
        sliced = text[start:]
        token_count = self._token_size(sliced)
        if token_count <= 0:
            return None
        return _Piece(
            block=piece.block,
            start_char=piece.start_char + start,
            end_char=piece.end_char,
            text=sliced,
            token_count=token_count,
            is_heading=piece.is_heading,
            is_atomic=piece.is_atomic,
            oversized=piece.oversized,
        )

    def _tail_overlap(
        self,
        pieces: Sequence[_Piece],
        target_tokens: Optional[int] = None,
    ) -> Tuple[List[_Piece], int]:
        target = self.overlap_tokens if target_tokens is None else target_tokens
        if target <= 0:
            return [], 0
        tail: List[_Piece] = []
        total = 0
        for piece in reversed(pieces):
            if piece.is_heading:
                break
            remaining = target - total
            if remaining <= 0:
                break
            if piece.token_count <= remaining:
                tail.append(piece)
                total += piece.token_count
                continue
            sliced = self._slice_tail(piece, remaining)
            if sliced is not None:
                tail.append(sliced)
                total += sliced.token_count
            break
        tail.reverse()
        return tail, total

    def build(self, artifact: DocumentArtifact) -> Tuple[List[ParentNode], List[IndexReadyChunk]]:
        parents: List[ParentNode] = []
        chunks: List[IndexReadyChunk] = []
        parent_counter = 0
        chunk_counter = 0

        for scope, blocks in self._group_blocks(artifact).items():
            parent_counter += 1
            parent_id = f"{artifact.artifact_id}_p{parent_counter:04d}"
            parent_chunks: List[IndexReadyChunk] = []
            current: List[_Piece] = []
            overlap_from_previous = 0

            def emit(carry_overlap: bool = True) -> None:
                nonlocal current, overlap_from_previous, chunk_counter
                if not current:
                    return
                chunk_counter += 1
                chunk = self._make_chunk(
                    artifact, parent_id, current, chunk_counter, overlap_from_previous
                )
                chunks.append(chunk)
                parent_chunks.append(chunk)
                if carry_overlap:
                    overlap, overlap_tokens = self._tail_overlap(current)
                    current = list(overlap)
                    overlap_from_previous = overlap_tokens
                else:
                    current = []
                    overlap_from_previous = 0

            for block in blocks:
                for piece in self._block_pieces(block):
                    current_tokens = sum(p.token_count for p in current)
                    has_body = any(not p.is_heading for p in current)

                    if piece.is_heading and has_body:
                        emit(carry_overlap=False)
                    elif current and current_tokens + piece.token_count > self.chunk_tokens:
                        emit(carry_overlap=has_body)
                        if current and sum(p.token_count for p in current) + piece.token_count > self.chunk_tokens:
                            budget = max(0, self.chunk_tokens - piece.token_count)
                            current, overlap_from_previous = self._tail_overlap(
                                current,
                                target_tokens=budget,
                            )
                        if piece.is_atomic and piece.oversized:
                            current = []

                    current.append(piece)

                    # 超大原子块不拆, 单独形成一个 chunk。
                    if piece.is_atomic and piece.oversized:
                        emit()
                        current = []

            if current:
                emit()

            # 去重重叠产生的重复 chunk。
            unique_chunks: List[IndexReadyChunk] = []
            seen = set()
            for chunk in parent_chunks:
                key = (chunk.chunk_id, tuple(f.block_id for f in chunk.fragments), chunk.full_text)
                if key not in seen:
                    unique_chunks.append(chunk)
                    seen.add(key)
            for chunk in unique_chunks:
                chunk.parent_id = parent_id

            parents.append(ParentNode(
                parent_id=parent_id,
                artifact_id=artifact.artifact_id,
                scope_title=scope,
                section_path=list(unique_chunks[0].section_path) if unique_chunks else [scope],
                child_chunk_ids=[c.chunk_id for c in unique_chunks],
                full_text="\n\n".join(c.full_text for c in unique_chunks),
                token_count=sum(c.token_count for c in unique_chunks),
                page_start=min((c.page_start for c in unique_chunks if c.page_start), default=None),
                page_end=max((c.page_end for c in unique_chunks if c.page_end), default=None),
            ))

        return parents, chunks
