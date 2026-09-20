r"""统一的文档摄取数据模型。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def stable_artifact_id(source: "SourcePayload") -> str:
    """从来源派生**确定性**的 artifact_id。

    为什么不用 uuid4
    ----------------
    原实现 4 个 parser 都写 `f"doc_{uuid.uuid4().hex[:12]}"`, 每次重建同一份
    语料都得到不同的 id。实测两次重建: `doc_049627f36e74` vs `doc_0e5e3c88e501`。

    后果:
      - `chunk_id = f"{artifact_id}_c{index:04d}"` 不稳定
        → 增量更新无法对齐 (无法按 artifact 删除旧 chunk)
        → 跨 run 无法按 chunk_id 做逐 chunk 对比
      - 依赖 chunk_id 的历史产物 (如 eval/runs/*/per_query.jsonl) 会失效
      - point_id 也由 chunk_id 派生 → 重建产生全新 point 而非覆盖

    派生规则
    --------
    优先用 `final_uri` (文档身份), 无则用 `uri`, 再无则用内容 `sha256`。
    取 sha256 前 12 位十六进制, 保持与原格式一致的 `doc_<12hex>`。

    为什么优先用 URI 而不是内容哈希
    ------------------------------
    URI 是**文档身份**, 内容哈希是**版本指纹**。
    用 URI 时, 换 parser / 换切块参数重建同一篇文档 → artifact_id 不变
    → chunk_id 按位置对齐, 可以逐 chunk 对比"改了什么"。
    若用内容哈希, 内容一变全部 id 就变, 对比只能靠全文匹配。

    代价 (需要知道): 同一 URI 内容更新后 artifact_id 不变,
    旧 chunk 不会自动消失 —— 正确的增量更新需要**按 artifact_id 删除**,
    本函数不负责那一步。
    """
    basis = (source.final_uri or source.uri or "").strip()
    if not basis:
        basis = (source.sha256 or "").strip()
    if not basis:
        raise ValueError(
            "无法派生 artifact_id: source 既无 final_uri/uri 也无 sha256"
        )
    return "doc_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


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
    # 来源 URL。HTML 路径填入, 用于引用时标注"这段来自哪个页面"。
    # 与 DocumentArtifact.source_uri 的区别: 后者是整篇文档的入口,
    # 前者是块级的来源 —— 当一篇文档由多个页面聚合而成时两者不同。
    url: Optional[str] = None
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
