r"""LlamaIndex NodeParser: 800 token + 400 token overlap + 边界语义微调。

输出标准 llama_index TextNode, 但切分算法保持:
    1. 在 section 内按句子累积到接近 chunk_size
    2. 在末尾 window_size 的 token 窗口内计算相邻句向量距离
    3. 选择语义距离最大的位置作为切点
    4. 下一块回退约 chunk_overlap tokens 作为重叠
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, List, Sequence

import numpy as np
from llama_index.core.node_parser import NodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode, Document, TextNode
from pydantic import Field

from preprocess import parse_markdown
from semantic_chunker_demo import CN_SENTENCE_SPLIT_REGEX


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(CN_SENTENCE_SPLIT_REGEX, text) if s.strip()]


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(1.0 - (va @ vb) / (na * nb))


def build_section_documents(markdown: str, source_name: str) -> List[Document]:
    """按顶层章节聚合, 同时记录每句的完整 section 路径。"""
    blocks = parse_markdown(markdown)
    grouped: OrderedDict[str, List[tuple[str, str]]] = OrderedDict()
    for block in blocks:
        if block.kind != "body":
            continue
        top_section = (block.section or "文档开头").split(" > ")[0].strip()
        grouped.setdefault(top_section, []).append((block.text, block.section))

    docs: List[Document] = []
    for top_section, parts in grouped.items():
        body_parts: List[str] = []
        sentence_sections: List[str] = []
        for text, full_section in parts:
            for sentence in split_sentences(text):
                body_parts.append(sentence)
                sentence_sections.append(full_section)
        body = "".join(body_parts)
        if not body.strip():
            continue
        docs.append(Document(
            text=body,
            metadata={
                "file_name": source_name,
                "section": top_section,
                "sentence_sections": sentence_sections,
                "content_type": "body",
            },
        ))
    return docs


class FixedSemanticNodeParser(NodeParser):
    """LlamaIndex NodeParser: 固定 token 目标 + overlap + 语义边界微调。"""

    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=400, ge=0)
    window_size: int = Field(default=400, ge=1)
    embed_model: Any = Field(default=None, exclude=True, repr=False)

    def _token_size(self, text: str) -> int:
        tokenizer = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return int(tokenizer._token_size(text))

    @staticmethod
    def _semantic_cut(
        start: int,
        hard_end: int,
        token_counts: Sequence[int],
        distances: Sequence[float],
        window_size: int,
    ) -> int:
        """在 [start, hard_end) 的尾部窗口里寻找语义断点。"""
        if hard_end <= start + 1 or hard_end > len(distances) + 1:
            return hard_end

        window_start = max(start, hard_end - 1)
        used = token_counts[window_start]
        while window_start > start and used + token_counts[window_start - 1] <= window_size:
            window_start -= 1
            used += token_counts[window_start]

        best_cut = hard_end
        best_distance = -1.0
        # boundary i 的切点位于句子 i 和 i+1 之间, 即 cut=i+1
        for i in range(window_start, hard_end - 1):
            if i < len(distances) and distances[i] > best_distance:
                best_distance = distances[i]
                best_cut = i + 1
        return best_cut

    @staticmethod
    def _overlap_start(
        current_start: int,
        cut: int,
        token_counts: Sequence[int],
        overlap: int,
    ) -> int:
        """从切点向前回退约 overlap tokens, 返回下一块起点。"""
        next_start = cut
        used = 0
        while next_start > current_start and used + token_counts[next_start - 1] <= overlap:
            next_start -= 1
            used += token_counts[next_start]
        return next_start

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        if self.embed_model is None:
            raise ValueError("embed_model is required for semantic boundary adjustment")

        output: List[BaseNode] = []
        for doc in nodes:
            sentences = split_sentences(doc.text)
            if not sentences:
                continue
            sentence_sections = list(
                doc.metadata.get("sentence_sections")
                or [doc.metadata.get("section", "文档开头")] * len(sentences)
            )

            # 超长配置块/代码块可能被当作单个句子。先用 token 窗口硬切,
            # 保证后续长度控制和语义窗口不会被一个巨型单元绕过。
            hard_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,
            )
            normalized_sentences: List[str] = []
            normalized_sections: List[str] = []
            for sentence, section in zip(sentences, sentence_sections):
                if self._token_size(sentence) <= self.chunk_size:
                    normalized_sentences.append(sentence)
                    normalized_sections.append(section)
                    continue
                for piece in hard_splitter.split_text(sentence):
                    if piece.strip():
                        normalized_sentences.append(piece)
                        normalized_sections.append(section)

            sentences = normalized_sentences
            sentence_sections = normalized_sections
            if not sentences:
                continue

            token_counts = [self._token_size(s) for s in sentences]
            sentence_vectors = self.embed_model.embed_documents(sentences)
            distances = [
                cosine_distance(sentence_vectors[i], sentence_vectors[i + 1])
                for i in range(len(sentence_vectors) - 1)
            ]

            start = 0
            chunk_index = 0
            overlap_from_prev = 0
            while start < len(sentences):
                hard_end = start
                used = 0
                while hard_end < len(sentences):
                    next_tokens = token_counts[hard_end]
                    if used + next_tokens > self.chunk_size and hard_end > start:
                        break
                    used += next_tokens
                    hard_end += 1

                if hard_end >= len(sentences):
                    cut = len(sentences)
                else:
                    cut = self._semantic_cut(
                        start=start,
                        hard_end=hard_end,
                        token_counts=token_counts,
                        distances=distances,
                        window_size=self.window_size,
                    )
                    if cut <= start:
                        cut = hard_end

                text = "".join(sentences[start:cut])
                chunk_section = (
                    sentence_sections[start]
                    if start < len(sentence_sections)
                    else doc.metadata.get("section", "文档开头")
                )
                metadata = dict(doc.metadata)
                metadata.pop("sentence_sections", None)
                node = TextNode(
                    text=text,
                    metadata={
                        **metadata,
                        "section": chunk_section,
                        "chunk_index": chunk_index,
                        "char_count": len(text),
                        "token_count": sum(token_counts[start:cut]),
                        "n_sentences": cut - start,
                        "overlap_tokens_from_prev": overlap_from_prev,
                        "split_method": "llamaindex_fixed_semantic",
                    },
                )
                output.append(node)
                chunk_index += 1

                if cut >= len(sentences):
                    break
                next_start = self._overlap_start(
                    current_start=start,
                    cut=cut,
                    token_counts=token_counts,
                    overlap=self.chunk_overlap,
                )
                if next_start <= start:
                    next_start = cut
                overlap_from_prev = sum(token_counts[next_start:cut])
                start = next_start

        return output
