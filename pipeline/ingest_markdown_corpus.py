r"""增量入库: 保留已有 dense point, 追加 Markdown 语料并重建 sparse。

用途:
    当前 Qdrant collection 已有 Red Hat chunk。
    本脚本读取现有 dense points, 只为新 Markdown 生成 chunk 和 embedding,
    然后基于全部 chunk 统一重建 BM25 sparse index。

运行前必须停止 FastAPI 服务, 因为 Qdrant local 有文件锁。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag_pipeline import build_metadatas, chunk_by_section, prepare_sentences
from semantic_chunker_demo import build_embeddings
from llamaindex_fixed_semantic_splitter import (
    FixedSemanticNodeParser,
    build_section_documents,
)
from sparse_pipeline_demo import (
    Chunk,
    SparseBM25Index,
    save_artifacts,
    store_sparse_collection,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QDRANT_PATH = "qdrant_data"
DENSE_COLLECTION = "redhat"
SPARSE_COLLECTION = "redhat_sparse"
ARTIFACT_DIR = Path(__file__).resolve().parent / "index_artifacts"


def read_existing_points(client: QdrantClient) -> List[Any]:
    points, _ = client.scroll(
        collection_name=DENSE_COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=True,
    )
    return sorted(points, key=lambda p: int(p.id) if isinstance(p.id, int) else str(p.id))


def to_existing_chunks(points: List[Any], start_order: int = 0) -> List[Chunk]:
    chunks: List[Chunk] = []
    for order, point in enumerate(points, start_order):
        payload: Dict[str, Any] = dict(point.payload or {})
        text = str(payload.get("text", "")).strip()
        section = str(payload.get("section") or "文档开头")
        metadata = dict(payload)
        metadata.pop("text", None)
        chunks.append(Chunk(
            id=str(point.id),
            source=str(payload.get("file_name") or DENSE_COLLECTION),
            section=section,
            order=order,
            text=text,
            index_text=f"{section}\n{text}" if section != "文档开头" else text,
            metadata=metadata,
            point_id=point.id,
        ))
    return chunks


def build_new_chunks(
    md: str,
    source_name: str,
    max_chars: int,
    window_chars: int,
    embeddings: Any,
    start_order: int,
) -> tuple[List[Chunk], List[List[float]], Dict[str, Any]]:
    pairs, stat = prepare_sentences(md, preprocess=True)
    sents = [s for s, _ in pairs]
    sections = [sec for _, sec in pairs]
    print(f"  分句: {len(sents)}")
    if not sents:
        raise RuntimeError("新文档没有解析出可用正文")

    sentence_vectors = np.array(
        embeddings.embed_documents(sents), dtype=np.float64
    )
    spans = chunk_by_section(
        sents, sections, sentence_vectors, max_chars, window_chars
    )
    metas = build_metadatas(sents, spans, source_name, sections)
    texts = ["".join(sents[a:b]) for a, b in spans]
    vectors = embeddings.embed_documents(texts)

    chunks: List[Chunk] = []
    for offset, (text, meta, _span) in enumerate(zip(texts, metas, spans)):
        section = str(meta.get("section") or "文档开头")
        chunks.append(Chunk(
            id="pending",
            source=source_name,
            section=section,
            order=start_order + offset,
            text=text,
            index_text=f"{section}\n{text}" if section != "文档开头" else text,
            metadata=dict(meta),
            point_id=None,
        ))

    return chunks, vectors, stat or {}


def build_new_chunks_llamaindex(
    md: str,
    source_name: str,
    chunk_tokens: int,
    overlap_tokens: int,
    window_tokens: int,
    embeddings: Any,
    start_order: int,
) -> tuple[List[Chunk], List[List[float]], Dict[str, Any]]:
    docs = build_section_documents(md, source_name)
    parser = FixedSemanticNodeParser(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        window_size=window_tokens,
        embed_model=embeddings,
    )
    nodes = parser.get_nodes_from_documents(docs)
    texts = [node.text for node in nodes]
    vectors = embeddings.embed_documents(texts)

    chunks: List[Chunk] = []
    position = 0
    for offset, (node, text) in enumerate(zip(nodes, texts)):
        meta = dict(node.metadata)
        meta["chunk_index"] = start_order + offset
        meta["page"] = position // 900 + 1
        position += len(text)
        section = str(meta.get("section") or "文档开头")
        chunks.append(Chunk(
            id="pending",
            source=source_name,
            section=section,
            order=start_order + offset,
            text=text,
            index_text=f"{section}\n{text}" if section != "文档开头" else text,
            metadata=meta,
            point_id=None,
        ))
    return chunks, vectors, {
        "documents": len(docs),
        "nodes": len(nodes),
        "split_method": "llamaindex_fixed_semantic",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="增量追加 Markdown 语料")
    ap.add_argument("--md", required=True, help="新语料 Markdown")
    ap.add_argument("--source-name", required=True, help="写入 payload 的 file_name")
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--window-chars", type=int, default=400)
    ap.add_argument("--chunker", choices=["legacy", "llamaindex"], default="legacy")
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=400)
    ap.add_argument("--window-tokens", type=int, default=400)
    ap.add_argument("--qdrant-path", default=QDRANT_PATH)
    args = ap.parse_args()

    path = Path(args.md)
    md = path.read_text(encoding="utf-8", errors="replace")
    print(f"读取新语料: {path} ({len(md)} 字)")

    client = QdrantClient(path=args.qdrant_path)
    try:
        info = client.get_collection(DENSE_COLLECTION)
        print(f"现有 dense points: {info.points_count}")
        old_points = read_existing_points(client)
        old_chunks = to_existing_chunks(old_points)
        max_id = max(int(p.id) for p in old_points if isinstance(p.id, int))
    finally:
        client.close()

    embeddings = build_embeddings("qwen")
    if args.chunker == "llamaindex":
        new_chunks, new_vectors, stat = build_new_chunks_llamaindex(
            md=md,
            source_name=args.source_name,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            window_tokens=args.window_tokens,
            embeddings=embeddings,
            start_order=len(old_chunks),
        )
    else:
        new_chunks, new_vectors, stat = build_new_chunks(
            md=md,
            source_name=args.source_name,
            max_chars=args.max_chars,
            window_chars=args.window_chars,
            embeddings=embeddings,
            start_order=len(old_chunks),
        )

    print(f"新 chunk 数: {len(new_chunks)}")
    print(f"预处理: {stat.get('by_kind', {})}")

    next_id = max_id + 1
    new_points: List[PointStruct] = []
    for idx, (chunk, vector, meta) in enumerate(
        zip(new_chunks, new_vectors, [c.metadata for c in new_chunks])
    ):
        point_id = next_id + idx
        chunk.point_id = point_id
        chunk.id = str(point_id)
        payload = dict(meta)
        payload["text"] = chunk.text
        new_points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        ))

    client = QdrantClient(path=args.qdrant_path)
    try:
        client.upsert(
            collection_name=DENSE_COLLECTION,
            points=new_points,
            wait=True,
        )
        total_dense = client.count(
            collection_name=DENSE_COLLECTION, exact=True
        ).count
        print(f"dense 写入完成: {len(new_points)} 新 point, 总数 {total_dense}")
    finally:
        client.close()

    all_chunks = old_chunks + new_chunks
    sparse_index = SparseBM25Index(all_chunks)
    save_artifacts(
        out_dir=ARTIFACT_DIR,
        source=f"{len(old_chunks)} existing + {len(new_chunks)} new",
        chunks=all_chunks,
        index=sparse_index,
        query_count=0,
    )
    stored = store_sparse_collection(
        qdrant_path=args.qdrant_path,
        collection=SPARSE_COLLECTION,
        chunks=all_chunks,
        index=sparse_index,
        rebuild=True,
    )
    manifest = {
        "source_name": args.source_name,
        "markdown": str(path),
        "old_chunks": len(old_chunks),
        "new_chunks": len(new_chunks),
        "dense_points": total_dense,
        "sparse_points": stored,
        "max_chars": args.max_chars,
        "window_chars": args.window_chars,
        "chunker": args.chunker,
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.overlap_tokens,
    }
    (ARTIFACT_DIR / "ingest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"sparse 重建完成: {stored} points")
    print(f"统一 sidecar 写入: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
