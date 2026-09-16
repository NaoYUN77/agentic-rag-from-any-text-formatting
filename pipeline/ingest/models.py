r"""统一的文档摄取数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourcePayload:
    source_type: str
    uri: str
    final_uri: str
    mime_type: str
    content_type: str
    data: bytes
    encoding: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    fetched_at: Optional[str] = None
    size_bytes: int = 0
    sha256: str = ""
    local_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    source_type: str
    mime_type: str
    parser: str
    confidence: float
    reason: str
    fallbacks: List[str] = field(default_factory=list)
    mode: Optional[str] = None


@dataclass
class DocumentBlock:
    block_id: str
    type: str
    text: str
    markdown: str = ""
    parent_id: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    order: int = 0
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    code_language: Optional[str] = None
    latex: Optional[str] = None
    image_uri: Optional[str] = None
    caption: Optional[str] = None
    ocr_confidence: Optional[float] = None
    quality_score: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    score: float
    status: str
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexDecision:
    tier: str
    dense_index: bool
    sparse_index: bool
    sparse_weight: float = 1.0
    rerank_penalty: float = 0.0
    review_required: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentArtifact:
    schema_version: str
    artifact_id: str
    source_uri: str
    source_type: str
    mime_type: str
    parser: str
    parser_version: str = ""
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_at: Optional[str] = None
    language: str = "unknown"
    blocks: List[DocumentBlock] = field(default_factory=list)
    assets: List[Dict[str, Any]] = field(default_factory=list)
    quality: Optional[QualityReport] = None
    index_decision: Optional[IndexDecision] = None
    route: Optional[RouteDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        parts: List[str] = []
        for block in self.blocks:
            if block.type == "heading":
                level = int(block.metadata.get("level", 1))
                parts.append(f"{'#' * max(1, min(level, 6))} {block.text}")
            elif block.type == "code":
                lang = block.code_language or ""
                parts.append(f"```{lang}\n{block.text}\n```")
            elif block.type == "formula" and block.latex:
                parts.append(f"$${block.latex}$$")
            elif block.type == "image":
                alt = block.caption or block.text or "image"
                parts.append(f"![{alt}]({block.image_uri or ''})")
            else:
                parts.append(block.markdown or block.text)
        return "\n\n".join(p for p in parts if p is not None).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "mime_type": self.mime_type,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "title": self.title,
            "authors": self.authors,
            "published_at": self.published_at,
            "language": self.language,
            "blocks": [b.to_dict() for b in self.blocks],
            "assets": self.assets,
            "quality": self.quality.to_dict() if self.quality else None,
            "index_decision": self.index_decision.to_dict() if self.index_decision else None,
            "route": asdict(self.route) if self.route else None,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
