r"""用 LlamaIndex 切分器重建 dense + sparse 全量索引。

注意:
    Qdrant local 有文件锁。运行前必须停止 FastAPI 服务。

流程:
    Red Hat + NGINX Markdown
      -> LlamaIndex FixedSemanticNodeParser
      -> TextNode
      -> dense embedding + Qdrant
      -> 基于新 chunk 重建 BM25 sparse index
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from llamaindex_fixed_semantic_splitter import (
    FixedSemanticNodeParser,
    build_section_documents,
)
from semantic_chunker_demo import build_embeddings
from sparse_pipeline_demo import (
    Chunk,
    SparseBM25Index,
    save_artifacts,
    store_sparse_collection,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QDRANT_PATH = Path("qdrant_data")
DENSE_COLLECTION = "redhat"
SPARSE_COLLECTION = "redhat_sparse"
ARTIFACT_DIR = Path(__file__).resolve().parent / "index_artifacts"
VECTOR_SIZE = 1024
CHARS_PER_PAGE = 900


def default_sources() -> List[tuple[Path, str]]:
    root = Path(__file__).resolve().parent
    return [
        (root / "corpus" / "redhat_p1-20.md", "redhat_p1-20.md"),
        (root / "corpus" / "nginx_guide.md", "nginx_guide.pdf"),
    ]


def build_chunks_for_source(
    path: Path,
    source_name: str,
    parser: FixedSemanticNodeParser,
    embeddings: Any,
    start_order: int,
) -> tuple[List[Chunk], List[List[float]], Dict[str, Any]]:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    docs = build_section_documents(markdown, source_name)
    nodes = parser.get_nodes_from_documents(docs)
    texts = [node.text for node in nodes]
    vectors = embeddings.embed_documents(texts)

    chunks: List[Chunk] = []
    position = 0
    for offset, (node, text) in enumerate(zip(nodes, texts)):
        meta = dict(node.metadata)
        meta["file_name"] = source_name
        meta["page"] = position // CHARS_PER_PAGE + 1
        meta["chunk_index"] = start_order + offset
        position += len(text)
        section = str(meta.get("section") or "文档开头")
        chunks.append(Chunk(
            id=str(start_order + offset),
            source=source_name,
            section=section,
            order=start_order + offset,
            text=text,
            index_text=f"{section}\n{text}" if section != "文档开头" else text,
            metadata=meta,
            point_id=start_order + offset,
        ))

    stat = {
        "path": str(path),
        "source_name": source_name,
        "documents": len(docs),
        "chunks": len(chunks),
        "tokens_min": min(int(c.metadata["token_count"]) for c in chunks) if chunks else 0,
        "tokens_max": max(int(c.metadata["token_count"]) for c in chunks) if chunks else 0,
        "chars_min": min(len(c.text) for c in chunks) if chunks else 0,
        "chars_max": max(len(c.text) for c in chunks) if chunks else 0,
    }
    return chunks, vectors, stat


def rebuild_dense(
    chunks: List[Chunk],
    vectors: List[List[float]],
) -> None:
    if QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        client.create_collection(
            collection_name=DENSE_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = dict(chunk.metadata)
            payload["text"] = chunk.text
            points.append(PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload=payload,
            ))
        client.upsert(
            collection_name=DENSE_COLLECTION,
            points=points,
            wait=True,
        )
    finally:
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="LlamaIndex 全量重建")
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=400)
    ap.add_argument("--window-tokens", type=int, default=400)
    ap.add_argument("--source", action="append", default=[],
                    help="path|source_name, 可重复")
    args = ap.parse_args()

    sources = default_sources()
    if args.source:
        sources = []
        for item in args.source:
            path_str, source_name = item.split("|", 1)
            sources.append((Path(path_str), source_name))

    embeddings = build_embeddings("qwen")
    parser = FixedSemanticNodeParser(
        chunk_size=args.chunk_tokens,
        chunk_overlap=args.overlap_tokens,
        window_size=args.window_tokens,
        embed_model=embeddings,
    )

    all_chunks: List[Chunk] = []
    all_vectors: List[List[float]] = []
    source_stats: List[Dict[str, Any]] = []
    for path, source_name in sources:
        print(f"切分 {source_name}: {path}")
        chunks, vectors, stat = build_chunks_for_source(
            path=path,
            source_name=source_name,
            parser=parser,
            embeddings=embeddings,
            start_order=len(all_chunks),
        )
        print(
            "  documents=%d chunks=%d tokens=%d~%d chars=%d~%d"
            % (
                stat["documents"], stat["chunks"], stat["tokens_min"],
                stat["tokens_max"], stat["chars_min"], stat["chars_max"],
            )
        )
        all_chunks.extend(chunks)
        all_vectors.extend(vectors)
        source_stats.append(stat)

    rebuild_dense(all_chunks, all_vectors)
    print(f"dense 重建完成: {len(all_chunks)} points")

    sparse_index = SparseBM25Index(all_chunks)
    save_artifacts(
        out_dir=ARTIFACT_DIR,
        source="llamaindex_fixed_semantic",
        chunks=all_chunks,
        index=sparse_index,
        query_count=0,
    )
    stored = store_sparse_collection(
        qdrant_path=str(QDRANT_PATH),
        collection=SPARSE_COLLECTION,
        chunks=all_chunks,
        index=sparse_index,
        rebuild=True,
    )
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunker": "llamaindex_fixed_semantic",
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap_tokens,
        "window_tokens": args.window_tokens,
        "dense_points": len(all_chunks),
        "sparse_points": stored,
        "sources": source_stats,
    }
    (ARTIFACT_DIR / "ingest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"sparse 重建完成: {stored} points")
    print(f"manifest: {ARTIFACT_DIR / 'ingest_manifest.json'}")


if __name__ == "__main__":
    main()
