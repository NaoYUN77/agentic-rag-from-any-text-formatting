r"""格式路由与摄取编排。"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List

from .cleaner import clean_artifact
from .chunker import BlockAwareHierarchicalChunkBuilder, IndexReadyChunk, ParentNode
from .models import DocumentArtifact, RouteDecision, SourcePayload
from .parser_registry import get_parser
from .quality import assess_raw_quality, decide_index
from .router import FormatRouter
from .source import load_source


@dataclass
class IngestResult:
    artifact: DocumentArtifact
    route: RouteDecision
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    parents: List[ParentNode] = field(default_factory=list)
    chunks: List[IndexReadyChunk] = field(default_factory=list)


class IngestPipeline:
    def __init__(self, router: FormatRouter | None = None) -> None:
        self.router = router or FormatRouter()

    def ingest(
        self,
        value: str,
        timeout: int = 30,
        max_bytes: int = 20 * 1024 * 1024,
        allow_private: bool = False,
        parser_kwargs: Dict[str, Any] | None = None,
        build_chunks: bool = True,
        chunk_tokens: int = 800,
        strip_headings: bool = True,
    ) -> IngestResult:
        source: SourcePayload = load_source(
            value,
            timeout=timeout,
            max_bytes=max_bytes,
            allow_private=allow_private,
        )
        route = self.router.route(source)
        kwargs = dict(parser_kwargs or {})
        attempts: List[Dict[str, Any]] = []
        candidates = [route.parser] + list(route.fallbacks)
        last_error: Exception | None = None

        for parser_name in candidates:
            try:
                parser = get_parser(parser_name)
                allowed = set(inspect.signature(parser).parameters)
                parser_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                artifact = parser(source, **parser_kwargs)
                artifact.route = route
                artifact.quality = assess_raw_quality(artifact)
                artifact = clean_artifact(artifact)
                artifact.quality = assess_raw_quality(artifact)
                artifact.index_decision = decide_index(artifact)
                attempts.append({
                    "parser": parser_name,
                    "status": "ok",
                    "quality": artifact.quality.status,
                })
                if artifact.quality.status != "reject":
                    parents = []
                    chunks = []
                    if build_chunks:
                        # 固定大小 + 结构边界切块（见 chunker 类 docstring）。
                        # 原先并存的 "sentence"（单句检索单元 + metadata 窗口）
                        # 策略已于 2026-09-21 随"滑动窗口机制取消"一并删除。
                        builder = BlockAwareHierarchicalChunkBuilder(
                            chunk_tokens=chunk_tokens,
                            strip_headings=strip_headings,
                        )
                        parents, chunks = builder.build(artifact)
                    return IngestResult(
                        artifact=artifact,
                        route=route,
                        attempts=attempts,
                        parents=parents,
                        chunks=chunks,
                    )
                last_error = RuntimeError(f"parser {parser_name} quality reject")
            except Exception as exc:
                attempts.append({
                    "parser": parser_name,
                    "status": "error",
                    "error": str(exc),
                })
                last_error = exc

        raise RuntimeError(f"所有 parser 均失败: {last_error}; attempts={attempts}")
